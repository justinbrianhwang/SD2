# Recorder bug audit — 2026-07-10

Verifying the counterfactual-intervention work surfaced nine defects in the CARLA recording
pipeline. Two of them silently corrupted the driving itself and the metric used to report it.

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

**Gap found while measuring:** `transfuser_record.py` keeps its own argument parser and accepts
neither `--num-vehicles`/`--num-walkers` nor `--anti-crawl`, so it cannot yet be run with NPC traffic
or under the shared anti-crawl protocol. Also, the adapters store only `steer`/`throttle`/`brake`, so
the recorder's `anti_crawl_applied` marker never reaches the JSONL and the number of nudged frames is
not recoverable from a recording.

---

## Verification status

All nine fixes are covered by the pure test suite (194 tests) and, where the defect only manifests
against a live simulator, by a live CARLA check. The first coherent recording after the fixes:

```
InterFuser, Town10HD_Opt, spawn 0, 300 frames, no stress, no traffic
route_progress  0.0000 -> 0.2551
off_route       never true
target_speed    mean 5.00, zero frames 0/300
```

A run that "drives" while `route_progress` never advances is the signature of defect 1. A run whose
`route_progress` jumps discontinuously is the signature of defect 2. Both are now checked directly.
