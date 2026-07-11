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

## 2. The causal chain closes at the outcome level (four independent routes)

Protocol: restart-each for routes 31->36 and 53->107 (CARLA restarted before
**every** run); back-to-back for the two expansion routes 33->36 and 114->51
(clean x3 local floor per route). 300 frames; no traffic. Each gaussian_noise s5
condition x3 (seeds 42/43/44). Route completion is the closed-loop outcome.

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

### Route 33 -> 36, second replication

| condition | route completion | mean speed |
| --- | ---: | ---: |
| clean | 0.9983 +/- 0.0008 | 3.44 |
| gaussian_noise s5 | 0.0000 +/- 0.0000 | 0.00 |
| gn s5 + **semantic-restore** | **0.9660 +/- 0.0031** | 3.65 |
| gn s5 + planning-restore | 0.0000 +/- 0.0000 | 0.00 |

degradation 0.998; **semantic recovery 0.966 (96.8%)**; planning recovery 0.000
(0%).

### Route 114 -> 51, third replication

| condition | route completion | mean speed |
| --- | ---: | ---: |
| clean | 0.9744 +/- 0.0210 | 3.06 |
| gaussian_noise s5 | 0.0778 +/- 0.0221 | 0.22 |
| gn s5 + **semantic-restore** | **0.9588 +/- 0.0006** | 3.08 |
| gn s5 + planning-restore | 0.0938 +/- 0.0389 | 0.29 |

degradation 0.897; **semantic recovery 0.881 (relative to the stall floor)**;
planning recovery 0.016.

A clean **double dissociation on all four routes**: restoring the clean semantic
map revives the totally-stalled ego to near-complete route completion; restoring
the clean planning waypoints does essentially nothing. The recovery is three
orders of magnitude above the clean noise floor.

### Bootstrap 95% confidence intervals (n=4 routes)

Recovery = mean(intervened) - mean(gn5_none), bootstrapped over runs (20,000
resamples, deterministic seed):

| route | semantic recovery [95% CI] | planning recovery [95% CI] |
| --- | --- | --- |
| 31 -> 36 | 0.978 [0.978, 0.978] | 0.000 [0.000, 0.000] |
| 53 -> 107 | 0.983 [0.983, 0.983] | 0.010 [-0.000, 0.030] |
| 33 -> 36 | 0.966 [0.964, 0.970] | 0.000 [0.000, 0.000] |
| 114 -> 51 | 0.881 [0.868, 0.907] | 0.016 [-0.028, 0.054] |

**Across the four routes: mean semantic recovery = 0.952 (sd 0.048); mean
planning recovery = 0.006 (sd 0.008).** Every route's semantic-recovery CI
excludes zero by a wide margin and does not overlap its planning-recovery CI; the
planning CIs all bracket zero. The dissociation is not a single-route artifact --
it holds on four routes spanning three spawn regions of Town10HD, under both
restart-each and back-to-back protocols.

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

## 9. Ecological validity: the result holds with real traffic

Sections 2-8 are on empty roads, where the only objects are the noise-induced
phantoms. To check the semantic localization is not an artifact of an empty scene,
route 31->36 was rerun with **60 vehicles + 40 walkers**, raising the mean
detected-object density to ~1.9/frame (up from ~0.4 on the empty road) -- above
SD2's 1.0 semantic-evidence gate. Clean/stress/restore x3 seeds, restart-each.

| condition (60veh/40walk) | route completion | obj density |
| --- | ---: | ---: |
| clean | 0.842 +/- 0.186 | 1.85 |
| gaussian_noise s5 | 0.037 +/- 0.041 | 2.24 |
| + **semantic-restore** | **0.817 +/- 0.163** | 0.70 |
| + planning-restore | 0.010 +/- 0.018 | 2.27 |

Real traffic makes the clean run noisier -- InterFuser legitimately slows and
stops behind vehicles, so clean completion varies 0.64-1.00 -- but the
dissociation holds: gaussian_noise s5 still causes a near-total stall (0.037)
**with real objects in view**, semantic-restore recovers it to the clean-traffic
level (0.817, **97% of the degradation**, ~4x the clean noise floor), and
planning-restore does not (0.010). The semantic localization is a property of the
model's response to the corruption, not an artifact of an empty road; and because
the object density clears the 1.0 gate, the correlational semantic evidence is now
sufficient too.

## 10. Cross-architecture: NEAT is robust where InterFuser is brittle

To test whether the semantic brittleness is InterFuser-specific, NEAT (the other
multi-input-controller model, which also exposes a semantic BEV-occupancy stage)
was run on the same routes. NEAT drives 6 of 8 straight routes (completion
0.79-0.97, with 1-3 wall scrapes). Under severity-5 stress, NEAT is robust to
**all four** visual corruptions (route 31->36):

| stressor s5 | NEAT route completion |
| --- | ---: |
| clean | 0.933 |
| gaussian_noise | 0.937 |
| motion_blur | 0.914 |
| brightness | 0.988 |
| fog | 0.950 |

The gaussian_noise s5 that totally stalls InterFuser leaves NEAT unaffected, so
SD2's stress-effectiveness gate correctly reports **no localizable NEAT failure**
under these stressors -- the method does not overclaim.

SD2 also explains *why*, at the control level. On the same gaussian_noise s5, the
same-pose single-run share (3 recordings each) is:

| model | semantic control share | planning control share | outcome |
| --- | ---: | ---: | --- |
| InterFuser | 1.000 | 0.009 | stalls (0.000) |
| NEAT | **0.000** | 1.000 | robust (0.937) |

The identical noise routes through **opposite stages** in the two architectures:
it corrupts InterFuser's object-density semantic map (phantom brake, stall), but
NEAT's BEV-occupancy semantic stage is noise-immune (0.000 semantic effect on
every control channel), and the small residual control change is entirely
planning, which NEAT absorbs without failing. This is the cross-architecture
contribution: SD2 does not merely say "one model failed and one did not" -- it
localizes to the same stage that is brittle in one architecture and robust in the
other, and explains the robustness difference mechanistically.

### 10.1 A full closed-loop replication on NEAT: the failure localizes to planning

The robustness above is to the four default stressors. A stronger corruption --
**contrast_shift s5** (RGB contrast x2.0 around the midpoint), a stressor NEAT was
not previously exposed to -- does degrade NEAT on two of the four routes: route
completion drops from ~0.93 to a ~0.68-0.75 floor with mean speed falling from
~4.6 to ~3.35 m/s (NEAT drives markedly slower and under-completes in the fixed
300-frame budget; no collisions). This gives a NEAT failure to localize, so we
ran the **same closed-loop restore/inject counterfactual** used for InterFuser
(routes 31->36 and 33->36, n=6 seeds per condition, back-to-back):

| route | clean | contrast none | **planning-restore** | planning-inject |
| --- | ---: | ---: | ---: | ---: |
| 31 -> 36 | 0.928 | 0.813 (4/6 degraded) | **0.941** | 0.751 |
| 33 -> 36 | 0.934 | 0.795 (3/6 degraded) | **0.939** | 0.901 |

Bootstrap 95% CI (20,000 resamples):

| route | planning-restore recovery [95% CI] | planning-inject degradation [95% CI] |
| --- | --- | --- |
| 31 -> 36 | **+0.128 [0.059, 0.195]** | **+0.177 [0.169, 0.183]** |
| 33 -> 36 | **+0.144 [0.043, 0.233]** | +0.032 [-0.015, 0.122] |

**Necessity holds on both routes:** restoring the clean planning waypoints
recovers NEAT to clean-level completion (all 12 restore runs land at 0.93-0.95;
the recovery CI excludes zero on both routes). **Sufficiency holds on route
31->36** (breaking only the waypoints reproduces the degradation, CI
[0.169, 0.183]) but is **not established on 33->36** (the stochastic degradation
basin triggered on only 1/6 inject runs, CI includes zero). The contrast
degradation localizes to NEAT's **planning (waypoint) stage** -- a *different*
stage than InterFuser's semantic stall.

**Two honesty caveats, stated plainly:**

1. **NEAT's semantic arm is scenario-vacuous here, so this is not a symmetric
   2x2.** SD2's NEAT "semantic" intervention swaps `red_light_occ` (the only
   NEAT semantic signal on the causal path to `control_pid`; the BEV-occupancy
   map SD2 records is a `decode(...)` side output, off-path). On these routes
   `red_light_occ` is identically 0 on all 300 frames, so the semantic swap
   changes control on **0/300 frames** and cannot matter -- an empty
   counterfactual, not evidence that "semantic is not necessary". By contrast the
   planning swap changes control on **93/300 frames**, confirming the planning arm
   is genuinely on the causal path. We therefore claim only a planning
   localization for NEAT, not a semantic dissociation.
2. **The NEAT degradation is a slowdown, not a stall.** It is a real, cleanly
   localizable behavioral change (~27% speed loss, cleanly recovered by
   planning-restore), but weaker than InterFuser's total 0.999 -> 0.000 stall, and
   its onset is stochastic (bites 3-4 of 6 seeds).

The honest cross-architecture statement is thus: SD2 localizes **distinct
failures in two architectures to their respective correct internal stages** --
InterFuser's gaussian-noise stall to its semantic object-density map (full
necessity+sufficiency 2x2), and NEAT's contrast slowdown to its planning
waypoints (necessity on two routes, sufficiency on one) -- each validated by a
closed-loop counterfactual, with no overclaiming where a channel is vacuous.

## 11. Reproduce

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

## 12. Scope and honesty

- **Two architectures with closed-loop localizations; InterFuser is the deep
  case.** InterFuser carries the full result (necessity+sufficiency 2x2,
  dose-response, mechanism, traffic, four-route distribution). NEAT adds a second
  closed-loop replication (Section 10.1) that localizes to a *different* stage
  (planning), but it is narrower: single causal arm (its semantic channel is
  scenario-vacuous), a slowdown rather than a stall, and sufficiency on one of two
  routes. TransFuser and CILRS command full brake every frame, AIM and TCP crawl
  — all in 0.9.16 with 0.9.10-era checkpoints (out of distribution), so they
  provide no localizable stressor-induced failure to analyze.
- These are still 0.9.10 checkpoints in 0.9.16. The claim is **not** that
  InterFuser drives 0.9.16 robustly (it does not — most routes stall). The claim
  is that on the routes where it does drive, an effective input stressor
  (gaussian_noise s5) induces a total stall, and SD2's counterfactual localizes
  that stall to the semantic stage and confirms the localization at the
  route-completion level, on two independent routes and across three analysis
  levels (control, behavioral, outcome).
- **Where a channel is vacuous, we say so rather than presenting a null as
  evidence.** NEAT's semantic intervention swaps a signal (`red_light_occ`) that
  is identically zero on the tested routes (0/300 control-frame effect), so its
  null recovery is structural, not empirical; we claim only the planning arm,
  which changes control on 93/300 frames.
- Every number here is read from a JSON/JSONL artifact under
  `data/carla/sd2_outcome_s*/` and `.../sd2_neat_contrast_s*/`, produced with the
  restart-each or back-to-back protocol noted per section.
