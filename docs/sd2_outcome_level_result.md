# SD2 outcome-level causal confirmation — InterFuser, semantic stage (2026-07-11)

This document supersedes the "there is no stable route / outcome-level
confirmation is structurally unavailable under 0.9.16" conclusion in
[sd2_pilot_results.md](sd2_pilot_results.md). That conclusion was an artifact of
the route selection, not of CARLA 0.9.16 or of SD2. The pilot only ever scanned
**long opposite-spawn routes** (destination = `spawn + count//2`), which cross
many junctions where a 0.9.10-trained InterFuser goes out of distribution and
stalls. On **short, straight, spawn-aligned routes** InterFuser drives stably to
near-completion, and on those routes the counterfactual causal chain closes at
the route-completion (outcome) level.

## 1. Stable drivable routes exist

A geometry scan of Town10HD_Opt's 155 spawn points (trace every spawn->spawn
pair 40-130 m apart, keep curvature ~0deg and spawn-yaw-aligned) found **56
straight, aligned candidate routes**. Screening InterFuser (clean, 300 frames,
no traffic) on 12 of them:

| route | length | clean completion | mean speed | collisions |
| --- | ---: | ---: | ---: | ---: |
| spawn 31 -> dest 36 | 67 m | **0.998** | 4.45 | 0 |
| spawn 33 -> dest 36 | 52 m | **0.998** | 3.45 | 0 |
| spawn 53 -> dest 107 | 59 m | **0.992** | 3.92 | 0 |
| spawn 114 -> dest 51 | 48 m | 0.97-0.99 | 3.08 | 0 |
| spawn 134 -> dest 52 | 48 m | 0.949 | 3.23 | 3 |
| spawn 47 -> dest 52 | 61 m | 0.688 | 3.10 | 2 |
| spawn 0 -> dest 53 | 63 m | 0.657 | 3.52 | 1 |
| spawn 105 -> dest 134 | 50 m | 0.325 | 1.39 | 1 |
| 102->116, 130->137, 99->26, 102->27 | 50-64 m | ~0.000 (stall) | ~0 | 0 |

Drivability is **scene-dependent, not purely geometric**: several equally
straight routes stall completely (the model brakes on the scene content, not the
geometry). But stable low-variance routes plainly exist. Why the old scan missed
them: the reference RoutePlanner returns `route[1]`, the next sparse waypoint. On
routes that begin near a junction that first waypoint is only ~5 m ahead, and an
OOD InterFuser reads a near goal as "arriving", brakes, and never advances (a
self-reinforcing stall). Straight routes place the first waypoint ~50 m ahead, so
the model drives.

## 2. The causal chain closes at the outcome level (two independent routes)

Protocol: CARLA restarted before **every** run; 300 frames; no traffic. Clean x5
(noise floor), each gaussian_noise s5 condition x3 (seeds 42/43/44). Route
completion is the closed-loop outcome.

### Route 31 -> 36 (67 m)

| condition | route completion | mean speed |
| --- | ---: | ---: |
| clean | 0.9990 +/- 0.0005 | 4.46 |
| gaussian_noise s5 | 0.0000 +/- 0.0000 | 0.01 |
| gn s5 + **semantic-restore** | **0.9778 +/- 0.0001** | 4.42 |
| gn s5 + planning-restore | 0.0000 +/- 0.0000 | 0.01 |

degradation 0.999; **semantic recovery 0.978 (97.9%)**; planning recovery 0.000
(0%); clean noise floor 0.0005.

### Route 53 -> 107 (59 m), independent replication

| condition | route completion | mean speed |
| --- | ---: | ---: |
| clean | 0.9920 +/- 0.0003 | 3.90 |
| gaussian_noise s5 | 0.0001 +/- 0.0001 | 0.01 |
| gn s5 + **semantic-restore** | **0.9832 +/- 0.0002** | 3.89 |
| gn s5 + planning-restore | 0.0101 +/- 0.0176 | 0.04 |

degradation 0.992; **semantic recovery 0.983 (99.1%)**; planning recovery 0.010
(1.0%); clean noise floor 0.0003.

A clean **double dissociation on both routes**: restoring the clean semantic map
revives the totally-stalled ego to near-complete route completion; restoring the
clean planning waypoints does essentially nothing. The recovery is three orders
of magnitude above the clean noise floor.

## 3. Necessity AND sufficiency (the 2x2)

Restore (fix one stage, keep the rest stressed) tests necessity; inject (break
one stage, keep the rest clean) tests sufficiency. Route 31->36, gaussian_noise
s5, x3 seeds each:

| stage | inject (break only this) | restore (fix only this) |
| --- | ---: | ---: |
| **semantic** | 0.010 (stalls) -> **sufficient** | 0.978 (recovers) -> **necessary** |
| planning | 0.977 (drives) -> not sufficient | 0.000 (stalls) -> not necessary |

Corrupting the semantic stage alone reproduces the total stall; restoring it
alone recovers driving. Corrupting or restoring planning does neither. Semantic
is both necessary and sufficient for the failure; planning is neither. This is a
complete causal identification, not a correlation.

## 4. Dose-response: recovery scales with severity

gaussian_noise s1..s5, x{none, semantic-restore, planning-restore} x2 seeds,
route 31->36 (clean 0.999):

| severity | none | semantic-restore | planning-restore | degradation | semantic recovery | planning recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.999 | 0.999 | 0.999 | 0.000 | 0.000 | 0.000 |
| 2 | 0.999 | 0.999 | 0.999 | 0.000 | 0.000 | 0.000 |
| 3 | 0.196 | 0.996 | 0.197 | 0.803 | **0.801** | 0.001 |
| 4 | 0.085 | 0.998 | 0.062 | 0.914 | **0.913** | -0.023 |
| 5 | 0.000 | 0.974 | 0.000 | 0.999 | **0.974** | 0.000 |

Below the threshold (s1-s2) nothing degrades. Once it bites (s3-s5) the
degradation rises monotonically, and **semantic-restore recovers almost all of it
at every severity** (0.80 -> 0.91 -> 0.97) while **planning-restore recovers ~0
throughout**. The semantic localization is not a single-severity artifact; it is
graded across the whole dynamic range.

## 5. Mechanism: which control channel, and when

From the same-pose dual-forward logged on a gn_s5 stress run (offline, no CARLA),
the per-control-channel effect of each stage (mean |Δ| vs the clean forward):

| channel | semantic effect | planning effect |
| --- | ---: | ---: |
| steer | 0.000 | 0.000 |
| throttle | 0.550 | 0.000 |
| **brake** | **0.733** | 0.000 |

The corrupted semantic map drives the **brake** (and throttle) channel, not
steering; planning has zero effect on any channel. The stressed forward commands
brake>0.5 on 222/300 frames starting at frame 0, while the clean forward and the
semantic-restored control brake on **0/300**. The mechanism is concrete: noise
corrupts InterFuser's object-density map, the controller reads phantom obstacles,
and it slams the brake -- restoring the clean semantic map removes the phantom
braking entirely.

## 6. The same result across levels (SD2's own pipeline)

Run through `sd2 intervention` (official analysis), route 31 -> 36:

- **Control level** (same-pose dual-forward, `--none-run`): semantic control
  share **1.000**, planning share **0.009**.
- **Outcome level** (`--baseline-clean/--stress/--intervened` with clean
  replicates): semantic-restore raw recovery **0.980**, planning-restore **0.000**,
  effect threshold 0.001.

Route 53 -> 107: semantic control share ~1.0 (planning 0.000); semantic-restore
outcome recovery **0.991**.

## 7. Counterfactual beats correlational (the methodological contribution)

The correlational diagnosis (SD2's `diagnosis.json`) is **route-dependent**:

| route | correlational primary stage | counterfactual (outcome-confirmed) | agree? |
| --- | --- | --- | :---: |
| 31 -> 36 | **planning** | semantic (0.98 recovery) | **No** |
| 53 -> 107 | semantic | semantic (0.99 recovery) | Yes |

On route 31 -> 36 the correlational label would have **misdiagnosed the stall as
planning**. SD2's same-pose counterfactual localizes it to **semantic on both
routes** and *proves* the localization by recovering the closed-loop outcome
(semantic-restore recovers ~98-99%, planning-restore ~0%). The counterfactual is
stable where the correlational label is not, and it is validated at the outcome
level. This is the core methodological claim of SD2.

## 8. Multi-stressor breadth: different corruptions localize to different stages

Same route (31 -> 36), same protocol, all four visual corruptions at severity 5,
each intervention x2 seeds. Clean baseline 0.9990, noise floor 0.0005.

| stressor s5 | no intervene | semantic-restore | planning-restore | degradation | recovering stage |
| --- | ---: | ---: | ---: | ---: | :---: |
| gaussian_noise | 0.000 | **0.978** | 0.000 | 0.999 | **semantic** |
| motion_blur | 0.900 | 0.900 | **0.998** | 0.099 | **planning** |
| brightness | 0.991 | 0.992 | 0.999 | 0.008 | none (ineffective) |
| fog | 0.994 | 0.995 | 0.998 | 0.005 | none (ineffective) |

Two corruptions degrade InterFuser on this route, and they localize to
**different stages**: gaussian_noise causes a total stall recovered only by
restoring the clean **semantic** map (planning-restore does nothing), while
motion_blur causes a smaller degradation (0.10, with collisions) recovered only
by restoring the clean **planning** waypoints (semantic-restore does nothing).
brightness and fog do not meaningfully degrade InterFuser here (completion stays
>0.99), so there is nothing to attribute. This stressor x stage double
dissociation is what a purely correlational method cannot produce: SD2 tells you
not just that the model failed, but which internal stage each corruption breaks.

## 9. Reproduce

Stable route + outcome chain (CARLA on Town10HD_Opt, InterFuser checkpoint in
`INTERFUSER_CKPT`), one condition shown; restart CARLA before each run:

```bash
# clean baseline (noise floor: repeat with several --seed)
python experiments/interfuser_record.py --town Town10HD_Opt --frames 300 --warmup 20 \
  --spawn-index 31 --dest-index 36 --num-vehicles 0 --num-walkers 0 \
  --stress none --intervene-stage none --output data/carla/clean.jsonl
# total-stall degradation
python experiments/interfuser_record.py --town Town10HD_Opt --frames 300 --warmup 20 \
  --spawn-index 31 --dest-index 36 --num-vehicles 0 --num-walkers 0 \
  --stress gaussian_noise --stress-severity 5 --intervene-stage none --output data/carla/gn5_none.jsonl
# semantic-restore (recovers) and planning-restore (does not)
python experiments/interfuser_record.py ... --stress gaussian_noise --stress-severity 5 \
  --intervene-stage semantic --intervene-direction restore --output data/carla/gn5_semantic.jsonl
python experiments/interfuser_record.py ... --stress gaussian_noise --stress-severity 5 \
  --intervene-stage planning --intervene-direction restore --output data/carla/gn5_planning.jsonl
# official analysis (control share + outcome recovery vs clean-replicate noise floor)
sd2 intervention --none-run data/carla/gn5_none.jsonl --output outputs/share
sd2 intervention --baseline-clean data/carla/clean.jsonl --stress data/carla/gn5_none.jsonl \
  --intervened data/carla/gn5_semantic.jsonl --config configs/mvp.yaml \
  --clean-replicates data/carla/clean_seed*.jsonl --output outputs/semantic
```

Straight, spawn-aligned routes are found by tracing every spawn->spawn pair
40-130 m apart and keeping curvature ~0deg with the spawn yaw aligned to the
route's initial heading (the default destination = `spawn + count//2` gives long
junction-crossing routes that stall). Verified stable pairs on Town10HD_Opt:
`31->36`, `33->36`, `53->107`, `114->51`.

## 10. Scope and honesty

- **InterFuser only.** TransFuser and CILRS command full brake every frame,
  AIM and TCP crawl, NEAT collides — all in 0.9.16 with 0.9.10-era checkpoints
  (out of distribution). Unchanged.
- These are still 0.9.10 checkpoints in 0.9.16. The claim is **not** that
  InterFuser drives 0.9.16 robustly (it does not — most routes stall). The claim
  is that on the routes where it does drive, an effective input stressor
  (gaussian_noise s5) induces a total stall, and SD2's counterfactual localizes
  that stall to the semantic stage and confirms the localization at the
  route-completion level, on two independent routes and across three analysis
  levels (control, behavioral, outcome).
- Every number here is read from a JSON/JSONL artifact under
  `data/carla/sd2_outcome_s31_d36/` and `.../sd2_outcome_s53_d107/`, produced
  with CARLA restarted before each run.
