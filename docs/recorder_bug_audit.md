# Recorder bug audit — 2026-07-10

Verifying the counterfactual-intervention work surfaced nine defects in the CARLA recording
pipeline. Two of them silently corrupted the driving itself and the metric used to report it.
Verifying *those* fixes surfaced five more, of a different kind: command-line flags that were
accepted and then silently ignored. Fourteen in total, all listed below.

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

## Verification status

All fourteen fixes are covered by the pure test suite (240 tests) and, where the defect only
manifests against a live simulator, by a live CARLA check. The first coherent recording after the
fixes:

```
InterFuser, Town10HD_Opt, spawn 0, 300 frames, no stress, no traffic
route_progress  0.0000 -> 0.2551
off_route       never true
target_speed    mean 5.00, zero frames 0/300
```

A run that "drives" while `route_progress` never advances is the signature of defect 1. A run whose
`route_progress` jumps discontinuously is the signature of defect 2. Both are now checked directly.
