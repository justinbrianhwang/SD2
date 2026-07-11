# Recorder bug audit — 2026-07-10

Verifying the counterfactual-intervention work surfaced nine defects in the CARLA recording
pipeline. Two of them silently corrupted the driving itself and the metric used to report it.
Verifying *those* fixes surfaced five more, of a different kind: command-line flags that were
accepted and then silently ignored. Diagnosing why TransFuser would not drive surfaced three more,
the worst of the lot: the sensors were never configured, so every model was fed a 2x zoomed camera
and a 25x sparser LiDAR than its checkpoint expects. Pushing that diagnosis further found an
eighteenth — a LiDAR y-sign bug that emptied TransFuser's BEV — and, past it, a model–environment
limit no recorder fix resolves. Eighteen defects in total, all listed below.

Each round of verification found the next round's bugs. The recurring cause is a silent fallback —
`has_attribute` skipping a misspelled key, `getattr(args, ..., default)` swallowing a missing flag,
a copy-pasted helper drifting from the one that was fixed. Every fix below therefore ships with a
guard that turns the silence into an error.

**Every live CARLA result recorded before this audit is invalid and must be re-recorded.** That
includes the robustness fingerprints, the cross-stress and cross-town tables, the anti-crawl
ablation, and every route-completion figure previously reported.

This document records what was wrong, how it was found, and how each fix was verified, so the
correction is auditable rather than asserted.

---

## 1. The global plan was mirrored about y = 0

`_location_to_gps` converts route waypoints into the lat/lon frame the model's `RoutePlanner`
consumes. It was ported from the CARLA 0.9.10 leaderboard, which computes

```python
my -= location.y
```

CARLA 0.9.16's GNSS sensor reports **latitude increasing with +y**. Keeping the negation reflects
the entire global plan about the y = 0 axis. `next_wp` — and therefore the `target_point` fed to the
model — pointed at a mirrored goal for the whole run.

### Evidence

The ego was spawned on route node 0 (Town10HD_Opt, spawn 0), a GNSS sample was read, and both were
scaled into metres with the model's own `RoutePlanner.scale`:

| | scaled north | scaled east |
| --- | --- | --- |
| GNSS sensor | **+24.48** | −64.67 |
| plan node 0 (before fix) | **−24.51** | −64.83 |
| ego world location | y = +24.47 | x = −64.64 |

The delta was 48.99 m, exactly `2 × 24.47`: a sign flip, not an offset. After the fix the same probe
reported a delta of **0.16 m**.

### Blast radius

The function was copy-pasted into three files. The copy in `_carla_e2e_common.py` is shared by AIM,
CILRS, NEAT and TCP, so **all six models** were steering toward reflected goals.

### Consequence for the paper

After the fix, InterFuser's `target_speed` went from 150 zero-frames out of 300 to **0 out of 300**,
and the brake limit-cycle disappeared. This raised the hypothesis that the "cold-start crawl", and
the entire **anti-crawl evaluation protocol** built to work around it, were artifacts of this bug.

**That hypothesis was tested and is false for two of the three models.** See
[the anti-crawl re-evaluation](#anti-crawl-re-evaluation-after-the-coordinate-fix) below.

---

## 2. Route completion was fabricated by an index teleport

`RouteProgressTracker._nearest_route_index` searched from `last_index - 5` to the **end of the
route** and took the geometrically nearest point. `last_index` is monotonic. Where a route passes
near itself — an opposite lane, a loop, a parallel street — the nearest point lies hundreds of
indices ahead, and a single frame permanently teleports the tracker down the route.

### Evidence

Replaying a real InterFuser recording (Town10HD_Opt, spawn 0 → dest 77, 287.4 m route) through the
tracker:

```
frame  45  idx= 11  remaining=281.1  progress=0.0190
frame  53  idx= 12 -> 254            (+242 indices in one frame)
frame  60  idx=254  remaining= 43.1  progress=0.8497
frame 119  idx=254  remaining= 55.1  progress=0.8079
```

The ego had driven ~22 m of a 287 m route, roughly **7.7 %**. The reported 0.8079 is an artifact.
Note also that progress *decreased* after the jump, from 0.8497 to 0.8079, as the ego drove away
from the snapped point — completion must never go down.

On a longer route the failure inverted: the index stuck at 8 while `remaining` grew from 358 m to
415 m, so progress clamped to 0.0000 for all 600 frames even though the ego moved 64 m. The tracker
could not distinguish "made no progress" from "abandoned the route".

Replaying the same trajectory through the fixed tracker gives **0.0200** and no index jumps.

### Fix

One shared implementation (the class had been copy-pasted three times), a bounded forward search
window, a corridor gate that refuses to advance the index when the ego is more than 15 m from every
candidate point, monotone completion, and an `off_route` flag surfaced in the recorded outcome so an
abandoned route is not silently reported as 0 % progress.

---

## 3. IMU and GNSS `sensor_tick` equalled the world delta

CARLA fires a sensor once its accumulated elapsed time reaches `sensor_tick`. With
`sensor_tick == delta == 0.05`, floating-point accumulation eventually falls short and the sensor
skips a tick. In synchronous mode the recorder then blocks in `SensorBuffer.read` waiting for a
reading that can never arrive, because no further tick is issued while it waits.

Runs died at roughly frame 400–500 with `TimeoutError: timed out waiting for sensor 'imu'`. Short
120-frame runs — everything recorded until now — happened to finish first. All six recorders had
`sensor_tick: 0.05` on the IMU and `0.01` on the GNSS; the GNSS value is the same trap for any
`--delta 0.01`.

Both are now `0.0`, enforced by an AST test over all six recorders and by a runtime guard that
rejects any `sensor_tick >= delta` at startup.

Isolation, restarting the server before every case so a poisoned world could not fake a result:

| frames | dest-index | result |
| --- | --- | --- |
| 120 | 50 | ok |
| 300 | 50 | ok |
| 600 | none | `TimeoutError` |
| 600 | 50 | ok, after the fix |

---

## 4. NPC teardown aborted the process and poisoned the server

Destroying vehicles still registered with the TrafficManager aborts the process. The abort left the
world in **synchronous mode**, and a world in synchronous mode with no client to tick it makes every
later run fail with `RuntimeError: time-out of 30000ms while waiting for the simulator`. A batch of
15 runs would have lost the fourteen after the first failure.

Isolation, one CARLA restart per case, 30 vehicles:

| case | configuration | rc | aborted |
| --- | --- | --- | --- |
| A | TrafficManager, 0 vehicles | 0 | no |
| D | 30 vehicles, no TrafficManager | 0 | no |
| B | TM + 30 autopilot vehicles, destroyed directly | 139 | **yes** |
| C | as B, plus `tm.shut_down()` | 3 | **yes, earlier** |
| E | TM + 30 autopilot vehicles, `set_autopilot(False)` before destroy | 0 | no |
| F | as E, plus `tm.shut_down()` | 0 | no |

`tm.shut_down()` is neither the cause nor the cure: it only moves the abort earlier, before the
world settings are restored, which is what poisons the server. Unregistering each vehicle from the
traffic manager before destroying it fixes it outright.

---

## 5. Controller state leaked through the intervention forwards

The dual-forward intervention recorder calls `InterfuserController.run_step` three times per tick:
one candidate control per forward, plus the applied control. The candidate calls were wrapped in a
helper that deep-copied only `turn_controller` and `speed_controller`, while `run_step` also mutates
`stop_steps`, `forced_forward_steps`, `red_light_steps`, `block_red_light`, `in_stop_sign_effect`,
`block_stop_sign_distance` and `stop_sign_trigger_times`.

A single "protected" call leaked four of them:

```
stop_steps              0 -> 1
in_stop_sign_effect     False -> True
block_stop_sign_distance  0 -> 2.0
stop_sign_trigger_times   0 -> 3
```

Because the dual forward runs regardless of `--intervene-stage`, this also corrupted plain
clean/stress recordings: `stop_steps` advanced three times per tick, changing the stop and creep
logic. Candidate controls are now computed on a throwaway `deepcopy` of the controller. Verified
live: the real controller receives exactly one `run_step` per tick (7 calls over 7 ticks) and the
clones receive exactly two per tick (14).

---

## 6. Remaining fixes

- **Hardcoded checkpoint path.** `interfuser_record.py` defaulted `--checkpoint` to an absolute path
  on one developer's machine. It now reads `$INTERFUSER_CKPT` and requires the flag when unset.
- **Total data loss on a mid-run crash.** Frames were accumulated in memory and written once at the
  end, so the 600-frame run that died at frame ~450 wrote nothing. Recordings now stream through
  `Sd2JsonlWriter` into `<output>.jsonl.partial`, flushed per frame and renamed to the final path
  only on success. Verified live by killing a running recorder: **208 frames survived**, and the
  final path was correctly absent.

---

## Anti-crawl re-evaluation after the coordinate fix

Fixing the mirrored global plan removed InterFuser's brake limit-cycle, which suggested the crawl —
and therefore the anti-crawl protocol — might have been an artifact. It was measured rather than
assumed. Each run: Town10HD_Opt, spawn 0, 300 frames (15 s), seed 42, no stress, no traffic, CARLA
restarted before every run. Speed is the ego's own speed from the recorded state, not a proxy.

| run | route progress | off_route | mean speed | frames moving | displacement |
| --- | --- | --- | --- | --- | --- |
| InterFuser, no anti-crawl | 0.2551 | false | 4.95 m/s | 300/300 | 72.8 m |
| AIM, no anti-crawl | 0.0245 | false | 0.66 m/s | 178/300 | 9.8 m |
| AIM, with anti-crawl | 0.0391 | **true** | 3.48 m/s | 276/300 | 46.4 m |
| TransFuser, default | 0.0073 | false | 0.12 m/s | 36/300 | 1.8 m |

Three conclusions, all of which contradict what we assumed before measuring:

1. **InterFuser needs no anti-crawl at all.** It drives at 4.95 m/s and stays on route. Its previous
   crawling was entirely an artifact of the mirrored plan.
2. **AIM and TransFuser genuinely crawl.** The coordinate fix did not help them: AIM covers 9.8 m in
   15 s, TransFuser 1.8 m at 0.12 m/s. The crawl is a property of those checkpoints, not of our bug.
   Anti-crawl remains necessary to obtain any driving from them.
3. **Anti-crawl is not innocuous.** It gets AIM moving, but the run goes `off_route`. The protocol
   trades a stationary ego for one that departs the planned route. This was invisible before the
   `off_route` flag existed, and it is a threat to validity that must be reported, not buried: a
   "driving" AIM run under anti-crawl is not following the route it is being scored on.

Anti-crawl therefore stays, scoped to the models that need it, and must be reported per model rather
than as a global protocol. Claims about AIM's and TransFuser's route completion under anti-crawl are
not yet defensible.

**Gaps found while measuring, all since closed.** See
[the copy-paste gaps](#the-copy-paste-gaps-flags-that-parsed-and-did-nothing) below.

---

## The copy-paste gaps: flags that parsed and did nothing

Measuring the anti-crawl protocol exposed a second family of defects. Three recorders
(`interfuser`, `tcp`, `transfuser`) kept their own `argparse` parser and their own drive loop
instead of the shared `parse_record_args` / `run_recording`. The copies had drifted. The failure
mode is the worst kind: a flag is **accepted** on the command line and then **silently ignored**,
so a run appears configured and is not.

The coverage before the fix, established by asking each recorder's own `--help`:

| recorder | parser | `--num-vehicles` | `--anti-crawl` | nudge runs? | NPCs spawn? | teardown |
| --- | --- | --- | --- | --- | --- | --- |
| aim, cilrs, neat | shared | yes | yes | yes | yes | shared |
| interfuser | own | yes | **no** | — | yes | shared |
| tcp | own | **no** | yes | yes | via shared loop | shared |
| transfuser | own | **no** | **no** | — | **no** | **own, stale copy** |

Five distinct defects fell out of this.

1. **`transfuser_record.py` accepted no NPC-traffic or anti-crawl flags at all.** It could not be
   run under the protocol the other recorders use.
2. **The anti-crawl marker never reached the JSONL.** `run_recording` writes
   `anti_crawl_applied` and `applied_throttle` onto the control record, but every adapter's
   `_control_state` returned only `{steer, throttle, brake}` and dropped them. The number of nudged
   frames was unrecoverable from a recording, so no anti-crawl ablation could cite it.
3. **`--anti-crawl` parsed and did nothing on `interfuser` and `transfuser`.** Only four of the six
   recorders call `e2e.run_recording`; those two have their own drive loops, which contained no
   anti-crawl logic. Adding the flag to their parsers — the obvious fix — would have made the
   problem worse by making it look configured.
4. **`--num-vehicles` parsed and spawned nothing on `transfuser`.** It never called
   `spawn_npc_traffic` and never wrote a `.scene.json` sidecar, so its runs were also the only ones
   that did not record requested-vs-spawned actor counts.
5. **`interfuser` and `transfuser` carried byte-identical stale copies of `cleanup`.** Both predate
   the NPC-teardown fix in §4: they destroy vehicles without first calling
   `set_autopilot(False, tm_port)`. InterFuser's copy was dead code (zero call sites; it correctly
   calls `e2e.cleanup`), but TransFuser's was live. **Wiring TransFuser's NPC spawning without also
   fixing its teardown would have reintroduced the server-poisoning bug of §4 outright.** The spawn
   and the teardown had to be fixed in one change.

### A silent collision that nearly landed

TransFuser already used `--creep-speed`, `--creep-duration` and `--creep-threshold` to override its
**own** native stuck/creep controller. The shared parser uses `--creep-speed` and `--creep-duration`
for the completely unrelated **anti-crawl applied-throttle nudge**, with different defaults. Merging
TransFuser onto the shared parser as-is would have silently changed `crawl_speed` from `0.1` to
`2.0` and overwritten `config.creep_duration` with `40` — on every run, including runs with no
`--anti-crawl` flag. The README documented both meanings under the same flag name.

TransFuser's native flags are now `--tf-creep-speed`, `--tf-creep-duration`, `--tf-stuck-threshold`.
The old spellings were removed rather than aliased, so a stale command line fails loudly:
`transfuser_record.py --creep-threshold 5` now exits with status 2.

### Fixes

One parser (`parse_record_args`, with an `extra_args` hook for model-specific flags), one nudge
(`AntiCrawlNudger`, called from all three drive loops), one teardown (`e2e.cleanup`; both stale
copies deleted). `ControlState` now declares `anti_crawl_applied` and `applied_throttle`, and every
adapter passes them through — while the recorded `throttle` stays the **model's raw output**, so the
diagnosis never sees the protocol's intervention.

### Verification

`AntiCrawlNudger` was checked frame-by-frame against a verbatim transcription of the inline block it
replaced, over 9 configurations × 30 trials × 200 frames: **identical**, including the un-nudged
counter updates and the `getattr` fallbacks that fire for a parser without the flags.

Live, 120 frames each, one CARLA server for all five runs:

| run | nudged frames | `applied_throttle` present |
| --- | --- | --- |
| aim, no `--anti-crawl` | 0 | 0 |
| aim, `--anti-crawl` | 95 | 95 |
| interfuser, `--anti-crawl` | 119 | 119 |
| transfuser, `--anti-crawl` | 93 | 93 |
| transfuser, native creep only | 0 | 0 |

The InterFuser row used `--creep-speed 100` so that every frame counts as a crawl. That is a probe of
the wiring, not a driving protocol: InterFuser drives at ~5 m/s and needs no nudge.

NPC spawning and teardown, four consecutive runs on one server (a single run proves nothing about
poisoning — the bug in §4 only appears on the *next* run):

| run | scene sidecar | actors left on server | poisoned |
| --- | --- | --- | --- |
| transfuser, 5 vehicles + 5 walkers | `veh 5/5  walk 5/5` | 0/0 | no |
| transfuser, 5 vehicles + 5 walkers | `veh 5/5  walk 5/5` | 0/0 | no |
| interfuser, 5 vehicles + 5 walkers | `veh 5/5  walk 5/5` | 0/0 | no |
| transfuser, no traffic | `veh 0/0  walk 0/0` | 0/0 | no |

`--tf-creep-speed 2.5 --tf-stuck-threshold 5` was confirmed to still drive TransFuser's own creep
(`Detected TransFuser stuck state; forced_move=0..6`), independently of the outer nudge.

Three AST regression tests now fail any recorder that accepts `--num-vehicles` without reaching
`spawn_npc_traffic`, `write_scene_sidecar` and `e2e.cleanup`, and any recorder that re-introduces a
local `_cleanup`. That is the general form of the defect, not just its five instances.

---

## The sensors were never configured

Every recorder builds a `sensors()` spec copied from its upstream agent. Those agents run under the
CARLA leaderboard, whose `agent_wrapper.setup_sensors` translates the spec onto the blueprint and
injects a large set of attributes. Our recorders bypass the leaderboard, and
`attach_model_sensors` forwarded exactly five keys:

```python
for attr in ("width", "height", "fov", "sensor_tick", "reading_frequency"):
    if attr in spec and blueprint.has_attribute(attr):
        blueprint.set_attribute(attr, str(spec[attr]))
```

**`width` and `height` are not blueprint attributes.** The camera blueprint calls them
`image_size_x` and `image_size_y`. `has_attribute` returned `False`, so the assignment was a silent
no-op and every camera ran at CARLA's default 800x600 regardless of what the recorder asked for.
The `if ... has_attribute` idiom turned a wrong name into silence instead of an error.

This is not cosmetic. The agents feed the raw frame to `scale_and_crop_image(image, crop=256)`,
which centre-crops 256 pixels. On the intended 400x300 frame that crop spans 64% of the width; on
the 800x600 frame we actually delivered it spans 32%. At `fov=100` the models saw an effective
horizontal field of view of roughly 32 degrees instead of 64 — a 2x zoom, with the vanishing point
and road geometry displaced accordingly.

No LiDAR attribute was set at all, so the LiDAR fell back to CARLA's defaults. Measured live at the
same pose over 20 ticks:

| | channels | points_per_second | points/frame |
| --- | --- | --- | --- |
| ours (CARLA defaults) | 32 | 56000 | **626** |
| leaderboard | 64 | 600000 | **15546** |

The leaderboard's camera lens attributes (`lens_circle_multiplier`, `lens_circle_falloff`,
`chromatic_aberration_intensity`, `chromatic_aberration_offset`) were likewise never applied.

Measured resolutions before and after the fix, read from each recorder's own first-packet log:

| recorder | declared | delivered before | delivered after |
| --- | --- | --- | --- |
| aim, cilrs, neat | 400x300 | 800x600 | 400x300 |
| tcp | 900x256 | 800x600 | 900x256 |
| transfuser | 960x480 | 800x600 | 960x480 |
| interfuser (side cameras) | 400x300 | 800x600 | 400x300 |
| lidar (interfuser, transfuser) | — | 626 pts | 16159 pts |

**Every model had been driving on inputs materially different from the ones its checkpoint was
trained and evaluated on.** All live results predating this fix are invalid, including the
"post-fix" InterFuser figures reported earlier in this document.

### Fix

`configure_sensor_blueprint` now mirrors `agent_wrapper.setup_sensors`: it translates
`width`/`height` to `image_size_x`/`image_size_y`, applies the leaderboard defaults per sensor
family, and lets an explicit spec value override a default.

Two guards make the class of defect impossible to reintroduce. A spec key that maps to no blueprint
attribute now raises instead of being skipped, and every written attribute is read back and compared.
The read-back compares floats at binary32 precision, because CARLA stores them as 32-bit floats and
`0.45` returns as `0.44999998807907104` — an exact comparison rejects a correctly applied value. That
was caught by a three-frame smoke test before any long run.

Separately, `validate_sensor_ticks` had only ever run on the four recorders that go through
`attach_model_sensors`; `interfuser` and `transfuser` hand-roll their attach loops and never reached
it. All six now do, enforced by an AST wiring test.

---

## The cold-start crawl is real, and anti-crawl does not fix TransFuser

With the sensors corrected, the crawl was re-measured. 300 frames (15 s), Town10HD_Opt, spawn 0,
seed 42, no stress, no traffic, **no anti-crawl**, CARLA restarted before each run. Speed is the
ego's own speed from `states.planning.ego`.

| model | route progress | mean speed | frames moving | collisions |
| --- | --- | --- | --- | --- |
| InterFuser | 0.2462 | 4.95 m/s | 300/300 | 0 |
| NEAT | 0.1221 | 2.38 m/s | 152/300 | **69** |
| AIM | 0.0248 | 0.71 m/s | 188/300 | 0 |
| TCP | 0.0029 | 1.09 m/s | 298/300 | 0 |
| TransFuser | 0.0000 | 0.00 m/s | 0/300 | 0 |
| CILRS | 0.0000 | 0.00 m/s | 0/300 | 0 |

The hypothesis that the crawl was an artifact of the 2x zoomed camera is **refuted**. AIM barely
moved (0.66 -> 0.71 m/s) and TransFuser went from 0.12 m/s to a dead stop. Only InterFuser drives.

TransFuser's own creep controller cannot fire on these routes: `config.stuck_threshold` is
`1100/action_repeat = 550` processed frames, about 55 s, far beyond a 300-frame run. Sweeping the
creep and anti-crawl parameters:

| configuration | progress | speed after frame 150 | **frames the model braked** | collisions |
| --- | --- | --- | --- | --- |
| default | 0.0000 | 0.00 | 300/300 | 0 |
| `--tf-stuck-threshold 5 --tf-creep-duration 60` | 0.0889 | 0.94 | 129/300 | 12 |
| `--tf-stuck-threshold 5 --tf-creep-duration 120` | 0.0949 | 3.16 | 27/300 | 13, off route |
| `--anti-crawl --creep-duration 40` | 0.1553 | 3.15 | 291/300 | 0 |
| `--anti-crawl --creep-duration 100` | **0.3163** | 6.61 | **300/300** | 0 |
| both | 0.0799 | 3.10 | 86/300 | 22, off route |

Read the last two columns together. The best-looking row completes **more of the route than
InterFuser** while the model commanded a full brake on **every single frame**. The anti-crawl nudge
overrode the applied throttle on 243 of 300 frames and pushed a braking model 93 m down the road;
the only thing TransFuser contributed was steering. Forcing motion with its native creep instead
produces 12-22 collisions and leaves the route.

**There is no creep parameter that makes TransFuser drive.** Its checkpoint, under this harness,
outputs `brake = 1.0` on 100% of frames with a predicted `target_speed` of 0.02 m/s, while its
`target_point` is a sane ~35 m ahead and its RGB assembly matches the reference agent's. CILRS is in
the same state. Reporting a route-completion number for either model under anti-crawl would be
reporting the protocol's number, not the model's.

The remaining known divergence from the reference agent is `action_repeat`: `submission_agent.run_step`
returns the previous control on every second frame (`if self.step % 2 == 1`), so the network decides
at 10 Hz, while our recorder runs it every frame at 20 Hz. That is the next thing to test. Until
TransFuser and CILRS drive without being pushed, no outcome-based claim about them is defensible.

---

## Verification status

All seventeen fixes are covered by the pure test suite (258 tests) and, where the defect only
manifests against a live simulator, by a live CARLA check. The first coherent recording with
correctly configured sensors:

```
InterFuser, Town10HD_Opt, spawn 0, 300 frames, no stress, no traffic, no anti-crawl
route_progress  0.0000 -> 0.2462
off_route       never true
mean speed      4.95 m/s, moving on 300/300 frames
model braked    5/300 frames
```

The earlier figure of 0.2551 in this document came from the same route recorded through the
crippled 800x600 camera and 626-point LiDAR. It is superseded, not corrected — the two runs are not
comparable.

Signatures now checked directly: a run that "drives" while `route_progress` never advances is
defect 1; a run whose `route_progress` jumps discontinuously is defect 2; a sensor packet whose
shape does not match the recorder's declared spec is the sensor defect; and a route-completion
figure recorded while the model braked on most frames is the anti-crawl protocol reporting its own
number rather than the model's.

### What is still not defensible

InterFuser drives. NEAT drives but collides 69 times in 15 seconds. AIM and TCP crawl. TransFuser
and CILRS output a full brake on every frame and do not move at all. Any outcome-based comparison
across this set would currently be a comparison of how hard the evaluation protocol pushed each
model, not of the models.

---

## Why TransFuser will not drive — a LiDAR sign bug, then a deeper limit

TransFuser's dead stop was traced input by input, by dumping the exact tensors the model receives.

The camera panorama is a clean, correctly framed road. The **LiDAR BEV was almost empty**: 21 of
256×256×2 cells occupied, out of 16159 raw points. TransFuser reads a near-empty occupancy grid as
"boxed in" and predicts a stationary trajectory, so `control_pid` brakes every frame.

The cause is a coordinate-sign bug of the same family as the mirrored GNSS (§1). `lidar_to_histogram_features`
splats points into a window `x∈[-16,16], y∈[-32,0]`, i.e. forward is −y. CARLA 0.9.16's ray-cast
LiDAR already reports forward as −y (raw y in `[-83, 0]`). TransFuser, written for 0.9.10 whose LiDAR
reported forward as +y, negates y (`lidar[:, 1] *= -1`) to land points in that window. Under 0.9.16
the negation pushes every forward point to +y, **outside** the window: measured, 26 of 16159 points
survive instead of 11600. Dropping the negation restores the BEV to **7381 occupied cells** and a
top-down scatter shows the road correctly ahead. The same negation was in `_safety_box` and the
(unused) point-pillars branch; all three are fixed.

**This is a real bug, and fixing it is necessary — but it is not sufficient.** With the BEV restored,
the emergency brake disabled, and a prototype `action_repeat` (the LiDAR sweeps only 180° per frame,
so odd frames carry a rear-half BEV — the reference agent reuses the previous control on those),
TransFuser **still brakes on every frame** and does not move.

What the tensors then show is that the model is not broken: it predicts a *different* trajectory in
Town01 (first waypoint 1.66 m ahead) than in Town10HD (0.02 m), so it is reading its inputs. It is
simply extremely conservative from a standstill. TransFuser is designed this way: it holds station
until its own creep controller fires, and that controller is gated on `stuck_threshold =
1100/action_repeat = 550` processed frames — about **55 seconds**. Leaderboard episodes run for
minutes, so the creep eventually engages; a 300-frame (15 s) recording ends long before it can.
Forcing the creep early gets motion but no steering, hence the 12–22 collisions in the sweep above.

Town01 is one of TransFuser's own training towns and it stalls there too, so this is not a
Town10 mismatch. The residual cause is a model–environment limit: a 0.9.10-era checkpoint evaluated
in 0.9.16, in a window shorter than its creep gate. That is a property of the evaluation setup, not
a recorder defect, and no pipeline fix changes it. It is recorded here so the LiDAR-sign fix is not
mistaken for a fix to the stall.
