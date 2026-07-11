# SD2 pilot analysis — InterFuser under input stress (2026-07-11)

This is the first analysis run on recordings made after the eighteen recorder fixes and the sensor
configuration fix (see [recorder_bug_audit.md](recorder_bug_audit.md)). It is a **pilot**: one model,
one route, one stressor family. Its purpose is to show the pipeline end to end and to establish what
is and is not yet defensible. Every number below is read from a JSON/JSONL artifact, not scraped from
a log.

## Which models can even be analyzed

Under CARLA 0.9.16, only **InterFuser drives on its own model prediction**. TransFuser and CILRS
output a full brake every frame (0.9.10-era checkpoints, out of distribution in 0.9.16; see the
audit). AIM and TCP crawl. NEAT drives but pins against a wall. So this analysis is InterFuser only,
and any "cross-architecture" framing is not currently available. That is a project-level constraint,
not a tuning problem.

## Closed-loop CARLA is non-deterministic — the clean noise floor

Town10HD_Opt, spawn 0, 300 frames, no stress, no traffic, run five times with an identical command
(the stressor RNG is inert with `--stress none`, so this isolates simulator nondeterminism):

| metric | value |
| --- | --- |
| route completion | mean 0.176, sd **0.063**, range [0.122, 0.258] |
| collision events | mean 1.2 |

The spread is large because InterFuser is *marginal* at one curve (~frame 165): the simulator's
own nondeterminism decides whether it clears the curve (~0.26) or clips it (~0.12), so clean is
effectively bimodal. **A single clean-vs-single-stress comparison is meaningless against this
spread** — which is exactly why SD2's same-pose dual-forward intervention exists: it runs the clean
and stress forward at the *same* pose every tick, so it never pays the closed-loop divergence.

## The stress-effectiveness gate: only gaussian_noise s5 degrades outcome

Three replicates each (seeds 42/43/44), same route:

| condition | route completion (mean±sd) | collision events | mean speed |
| --- | --- | --- | --- |
| clean | 0.176 ± 0.063 | 1.2 | 3.55 |
| gaussian_noise s3 | 0.253 ± 0.003 | 0 | 4.93 |
| gaussian_noise **s5** | **0.000 ± 0.000** | 0 | **0.01** |
| motion_blur s5 | 0.125 ± 0.004 | 1.7 | 2.63 |

gaussian_noise s3 does **not** degrade — it is *more* stable than clean (sd 0.003) and gets further,
because the noise happens to carry InterFuser cleanly through the marginal curve. This was the
"stress looks better than clean" artifact seen in the first pilot. motion_blur s5 keeps route
completion near the clean level but raises collisions. **Only gaussian_noise s5 clears the clean
noise floor**, and it does so decisively and reproducibly: InterFuser stops dead (0.000, 0.01 m/s,
sd 0).

## The causal result: gaussian_noise s5 stalls InterFuser through the SEMANTIC stage

Under gaussian_noise s5, the same-pose intervention shows the model would brake on the stress image
and drive on the clean image:

- stress-forward control: throttle 0.21, **brake 0.72**
- clean-forward control: throttle 0.75, brake 0.00

Restoring one stage at a time isolates the cause. Three independent lines of evidence agree that it
is **semantic** (the object-density / traffic map the controller consumes), not planning:

| restore stage | share of control change | behavioral outcome | applied brake |
| --- | --- | --- | --- |
| planning | **0.0 %** | 0.000, stays stopped (3/300 moving) | 215/300 |
| **semantic** | **96.6 %** | **0.101, driving resumes (123/300 moving)** | **0/300** |

Restoring the clean semantic map revives the stalled ego (0.000 → 0.101 route, brake 215 → 0);
restoring the clean planning waypoints does nothing. The noise corrupts InterFuser's scene
understanding, the controller reads phantom obstacles, and it brakes — planning is intact throughout.

Note the contrast with the weaker stress: at gaussian_noise s3, where outcome did not degrade,
the counterfactual attributed most of the (small) control change to *planning* (57 %) while the
correlational diagnosis said semantic — and the semantic evidence was insufficient (mean detected
objects 0.47 < 1.0). At the severity that actually stalls the car, the counterfactual attribution
(96.6 % semantic) and the correlational diagnosis agree. The counterfactual is what pins the cause;
the correlational label alone was not trustworthy at low severity.

## What is honestly not yet established

The outcome-recovery metric is **null** ("effect_below_noise_floor"): the clean noise floor computed
from the five replicates is 0.137, larger than the 0.122 degradation, so a completion-level recovery
cannot be declared significant even though semantic-restore visibly revived driving. The causal claim
holds at the **control level** (96.6 % share, brake 0.72 → 0.00) and the **behavioral level**
(0.000 → 0.101, 0 → 123 moving frames); it does **not** yet hold at the **route-completion level**,
because this marginal route's variability swamps it.

The next step is to find a spawn/route where InterFuser drives stably (small clean variance) and
repeat the gaussian_noise s5 semantic intervention there. If completion-level recovery then clears
the noise floor, the full chain — stress → semantic corruption → stall → semantic-restore → recovery
— is established end to end. Until then, the defensible result is: *the counterfactual localizes the
stall to the semantic stage at the control and behavioral levels, and reports outcome recovery as
unmeasurable against the route's own noise, rather than overclaiming it.*
