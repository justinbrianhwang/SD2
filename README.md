# SD2

**SD2 (System Deviation Diagnosis)** is a robustness diagnosis framework for **end-to-end (E2E) autonomous driving models**. It decomposes a driving system into observable functional stages — **perception → scene representation → planning → control → outcome** — runs the same scenario under clean and stressed conditions, measures how much each stage deviates, and localizes the stage where robustness *first* collapses and how the error propagates downstream.

Instead of asking *"how well does this model drive?"*, SD2 asks *"where in the pipeline does robustness collapse, and how does the error propagate?"*

![SD2 concept: where does robustness collapse in the driving pipeline](assets/images/fig1_concept.png)

Diagnosis outputs are **temporal-correlational**: SD2 identifies the earliest stage whose deviation crosses calibrated thresholds and is temporally followed by downstream deviation and/or driving-failure evidence. It does not claim a mechanistic root cause.

## What SD2 observes

A modern E2E model is a black box from sensors to actuation. SD2 "opens" it into observable functional stages and reads the intermediate state at each one — raw image, neural features, the model's internal **scene representation** (object density, BEV detections, or BEV occupancy), the predicted trajectory, and the vehicle control signals:

![Opening the E2E black box into functional stages](assets/images/fig2_pipeline.png)

We deliberately say *scene representation* rather than *scene understanding*: InterFuser's object density, TransFuser's BEV detections, and NEAT's BEV occupancy are semantic **representations**, not human-style understanding. A `reasoning` stage also exists in the schema, but it is **optional** and used only by language-based driving agents — it plays no part in the E2E experiments reported here.

## How it works

SD2 pairs a **clean run** with a **stress run** frame by frame, computes a normalized deviation per stage, analyzes how deviations propagate between adjacent stages, and diagnoses the **primary failure stage** — then writes a Markdown report:

![SD2 method flow: clean/stress runs to diagnosis report](assets/images/fig3_method.png)

## Example Output

A real CARLA closed-loop **TransFuser** run (Town10HD_Opt, Gaussian noise severity 3, matched clean/stress pair) produces a per-stage **robustness fingerprint** — higher is more robust:

![Robustness fingerprint](docs/example/robustness_fingerprint.png)

The **deviation timeline** shows where robustness degrades first: TransFuser's planning deviation rises earliest and largest, with control drifting after it, while its semantic representation stays comparatively stable:

![Stage-wise deviation timeline](docs/example/deviation_timeline.png)

From this, the diagnosis module generates a natural-language summary:

> Under Gaussian Noise severity 3, the transfuser model completed 83.5% of the route and did not record a collision or lane invasion. No stage crossed the critical deviation threshold. Downstream deviation increases followed the Planning onset in the order Control (+0.020). The primary_failure_stage label is Planning. Planning had the highest observed mean deviation (0.234) across the pipeline stages.

`diagnosis.json` includes `"diagnosis_type": "temporal_correlational"` to make this framing explicit — SD2 localizes the earliest-collapsing stage by timing, not by mechanistic proof.

See the full generated report at [docs/example/example_report.md](docs/example/example_report.md).

## Cross-architecture comparison

Because SD2 is architecture-agnostic, it can diagnose different E2E models under the *same* stress and reveal that they fail at *different* stages. On real CARLA closed-loop runs under Gaussian noise, **InterFuser** keeps a robust visual encoder but collapses at the **scene-representation (semantic)** stage, while **TransFuser**'s fused feature is itself noise-sensitive so its collapse originates at the **vision/feature** stage and propagates into planning:

![Cross-architecture failure comparison: semantic-stage vs feature-stage collapse](assets/images/fig4_cross_model.png)

Details and the honest caveats are in [docs/example/cross_model_comparison.md](docs/example/cross_model_comparison.md).

### E2E models & source repositories

SD2 diagnoses six published E2E driving models. The model weights and code are
consumed read-only through gitignored `models/` junctions; nothing in this repo
redistributes them. Original sources:

| Model | Source repository | Paper (venue) | Sensors | SD2 stages observed |
| --- | --- | --- | --- | --- |
| **InterFuser** | [opendilab/InterFuser](https://github.com/opendilab/InterFuser) | Shao et al., CoRL 2022 | camera + LiDAR | vision, **semantic** (object density), planning, control |
| **TransFuser** | [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser) | Chitta et al., CVPR 2021 / TPAMI 2023 | camera + LiDAR | vision, **semantic** (BEV-seg / detections), planning, control |
| **AIM** | [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser) | Chitta et al., CVPR 2021 (baseline) | camera | vision, planning, control (no semantic head) |
| **CILRS** | [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser) | Codevilla et al., ICCV 2019 (reimpl.) | camera | vision, control (no semantic / waypoints) |
| **NEAT** | [autonomousvision/neat](https://github.com/autonomousvision/neat) | Chitta et al., ICCV 2021 | multi-camera | vision, **semantic** (BEV occupancy), planning, control |
| **TCP** | [OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP) · weights [Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) | Wu et al., NeurIPS 2022 | camera | vision, planning, control (no semantic head) |

## Results: live CARLA robustness diagnosis

All numbers below are from **real CARLA 0.9.16 closed-loop runs** (Town10HD_Opt,
120 frames, synchronous mode, matched spawn/route/seed per pair). Robustness is
`1 − mean normalized stage deviation` (higher = more robust, ∈ [0, 1]).
`—` = stage not observable for that architecture (e.g. no semantic head).

#### Two means, and when to use which

Different architectures expose different stages, so a single average is **not**
comparable across models. SD2 therefore reports two summaries:

- **Observed-stage mean** — averages whichever stages that model exposes. Use it
  for *within-model* diagnosis. It is **not** a cross-model ranking: a model is
  penalised simply for exposing a fragile stage that another model hides. (In our
  data InterFuser's observed mean is 0.902 only because it exposes a fragile
  semantic head; on the stages it shares with the camera baselines it scores
  0.957.)
- **Common-stage mean** — averages the stages **every** model exposes, i.e.
  `vision + control` (CILRS regresses control directly and predicts no waypoints,
  and AIM/CILRS/TCP have no semantic head). Use it for *cross-model* comparison.

`sd2 fingerprint` now emits both columns, and `fingerprint.json` carries
`common_stage_mean` alongside `mean_robustness`.

#### Evaluation protocol: anti-crawl

From a standstill these models fall into a cold-start crawl limit-cycle, which
would make every deviation a near-stationary artifact. We therefore use an
**anti-crawl moving-ego protocol**: TransFuser's own creep controller is allowed
to engage during the crawl, and AIM/CILRS/TCP get a throttle burst on the
*applied* actuation. It is applied **identically to the clean and the stress
run**, and SD2 records each model's **raw control output separately from the
applied actuation**, so the control-stage comparison remains a pure model
measurement. It is an evaluation protocol, not a driving-score aid — the ablation
is in [Anti-crawl ablation](#anti-crawl-ablation) below.

### Multi-seed statistical robustness (Gaussian noise, severity 3, seeds 42–46)

| Model | n | Common-stage mean (cross-model) | Observed-stage mean (within-model) | Primary failure stage (stability) |
| --- | --- | --- | --- | --- |
| CILRS | 5 | **0.973 ± 0.004** | 0.973 ± 0.004 | none crosses critical |
| AIM | 5 | 0.941 ± 0.005 | 0.947 ± 0.004 | control (5/5) |
| TCP | 5 | 0.898 ± 0.004 | 0.845 ± 0.009 | planning (4/5) |
| TransFuser | 5 | 0.898 ± 0.009 | 0.870 ± 0.006 | planning (5/5) |
| NEAT | 5 | 0.897 ± 0.022 | 0.890 ± 0.015 | planning (4/5) |

Across five seeds the per-model variance is small (std ≤ 0.022) and the primary
failure stage is the same in ≥4/5 runs, so SD2's diagnoses are stable — the
weakest stage is a property of the architecture, not of a lucky seed.

Note how the two means disagree: on the observed-stage mean **TCP looks worst
(0.845)**, but that is only because it exposes a fragile planning stage that
CILRS does not have. On the common stages TCP (0.898) is mid-field, tied with
TransFuser. **Rank models with the common-stage column.** Generated with
`sd2 aggregate` (see `outputs/multiseed/<model>/`).

### Cross-stress stage robustness (severity 3, seed 42, moving ego)

Same model under four input stresses — this is the architecture-level robustness
fingerprint (RQ3: *where* does each model first collapse?).

Per-stage scores are directly comparable across models; the two mean columns are
`observed` (within-model) and `common` = mean(vision, control) (cross-model).

| Model | Stress | Vision | Semantic | Planning | Control | Observed mean | Common mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIM | gaussian_noise | 0.961 | — | 0.959 | 0.920 | 0.947 | 0.940 |
| AIM | motion_blur | 0.877 | — | **0.672** | 0.809 | 0.786 | 0.843 |
| AIM | brightness | 0.978 | — | 0.971 | 0.942 | 0.964 | 0.960 |
| AIM | fog | 0.969 | — | 0.941 | 0.926 | 0.945 | 0.948 |
| CILRS | gaussian_noise | 0.987 | — | — | 0.967 | 0.977 | 0.977 |
| CILRS | motion_blur | **0.549** | — | — | 0.641 | **0.595** | **0.595** |
| CILRS | brightness | 0.996 | — | — | 0.976 | 0.986 | 0.986 |
| CILRS | fog | 0.957 | — | — | 0.897 | 0.927 | 0.927 |
| InterFuser | gaussian_noise | 0.974 | **0.767** | 0.927 | 0.940 | 0.902 | 0.957 |
| InterFuser | motion_blur | 0.952 | **0.773** | 0.885 | 0.981 | 0.898 | 0.966 |
| InterFuser | brightness | 0.989 | 0.873 | 0.954 | 0.956 | 0.943 | 0.973 |
| InterFuser | fog | 0.991 | 0.807 | 0.958 | 0.986 | 0.935 | 0.988 |
| NEAT | gaussian_noise | 0.929 | 0.944 | 0.838 | 0.910 | 0.905 | 0.919 |
| NEAT | motion_blur | 0.790 | 0.949 | **0.423** | 0.854 | 0.754 | 0.822 |
| NEAT | brightness | 0.954 | 0.963 | 0.876 | 0.908 | 0.925 | 0.931 |
| NEAT | fog | 0.948 | 0.955 | 0.854 | 0.878 | 0.909 | 0.913 |
| TCP | gaussian_noise | 0.898 | — | 0.764 | 0.908 | 0.857 | 0.903 |
| TCP | motion_blur | 0.874 | — | 0.753 | 0.956 | 0.861 | 0.915 |
| TCP | brightness | 0.924 | — | 0.796 | 0.963 | 0.894 | 0.943 |
| TCP | fog | 0.947 | — | 0.828 | 0.969 | 0.915 | 0.958 |
| TransFuser | gaussian_noise | 0.879 | 0.914 | 0.766 | 0.898 | 0.864 | 0.889 |
| TransFuser | motion_blur | **0.680** | 0.913 | **0.538** | 0.919 | 0.763 | 0.800 |
| TransFuser | brightness | 0.961 | 0.955 | 0.901 | 0.906 | 0.931 | 0.934 |
| TransFuser | fog | 0.965 | 0.953 | 0.909 | 0.965 | 0.948 | 0.965 |

**What the numbers show (RQ3 — architectures fail at different stages):**

- **Motion blur is the harshest stress** and it collapses the **planning** stage
  hardest: NEAT 0.423, TransFuser 0.538, AIM 0.672, TCP 0.753. Fog and
  brightness are mild (most stages > 0.9).
- **CILRS is the least robust to motion blur** — its single-frame image feature
  degrades (vision 0.549) and in closed loop the run crashed (47 collisions, 0%
  route). Under gaussian/brightness it is otherwise the *most* robust model.
- **InterFuser's weak point is semantic** (0.767–0.873 across all four stresses)
  on top of a robust encoder — its object-density decoder is the fragile link.
- **NEAT's BEV-seg semantic stays robust** (0.944–0.963) under every stress
  while its planning is sensitive — perception is stable, trajectory is not.
- **TransFuser** is vision- and planning-sensitive (both drop under noise/blur),
  consistent with a fused image+LiDAR feature that is itself perturbation-prone.
- **AIM/TCP** keep control robust; their fragility is in planning under blur.

### Cross-town robustness and generalization (Gaussian noise, severity 3, seed 42)

Running the same models on other maps exposes a second, model-level result: these
2021-era checkpoints **do not generalize across towns**. Driving is also strongly
**spawn-dependent** — a spawn on open road drives, a spawn facing a junction
crashes — so each town was probed with a NEAT spawn scout (`spawn_scout`) to find
a drivable start before recording.

| Model | Town10HD (spawn 0) | Town01 (spawn 128) | Town03 | Town05 |
| --- | ---: | ---: | ---: | ---: |
| AIM | 0.940 · ~88% | 0.935 · ~35% | 0.938 · OOD | 0.951 · OOD |
| CILRS | 0.977 · ~85% | 0.825 · ~54% | 0.781 · OOD | 0.924 · OOD |
| NEAT | 0.919 · ~85% | 0.909 · **~72%** | 0.929 · OOD | 0.818 · OOD |
| TCP | 0.903 · ~87% | 0.917 · ~41% | 0.900 · OOD | 0.920 · OOD |
| TransFuser | 0.889 · ~85% | 0.787 · ~46%† | 0.895 · OOD | 0.884 · OOD |

*Each cell is `common-stage mean robustness · clean route completion` — the
common-stage mean is used because this is a cross-model table. **OOD** = no
scouted spawn produced a drivable run (best NEAT probe ≤ 5% with frequent
collisions).*

- **Town10HD and Town01 drive** (with a scouted spawn): NEAT completes ~72% of
  Town01, and every model gives a real moving-ego closed-loop pair there.
- **Town03 and Town05 are out-of-distribution** — across six probed spawns even
  NEAT (the strongest driver) never exceeds ~5% and usually crashes or stalls, so
  no model drives them. This is a genuine **town-overfitting / generalization
  failure** of the checkpoints, not a recorder bug (the identical recorder drives
  Town10HD and Town01).
- **Where the ego drives, the failure *signature* is consistent**: NEAT and
  TransFuser stay semantic-robust and break first at *planning*; AIM/CILRS keep
  *vision* robust and break at *control*. SD2 reads an architecture-level
  signature, not a map artifact.
- **† TransFuser is the least town-robust**: it drives Town01 clean (~46%) but
  under Gaussian noise it crashes (62 collisions), its common-stage mean drops to
  0.787 (from 0.889 in Town10HD) and its semantic/planning robustness falls to
  0.52/0.58 — noise that is survivable in Town10HD is not in Town01.

**Reading the OOD cells honestly.** SD2 still computes stage deviations for
Town03/Town05, but on egos that never complete the route, so those numbers are
raw model-output sensitivity to the perturbation, **not** closed-loop robustness
— which is why they are marked OOD and kept out of the driving comparison. The
result here is diagnostic rather than a driving score: SD2 surfaces *that* these
checkpoints fall out of distribution on Town03/Town05, and — on the maps that do
drive — *where* each architecture first breaks. Note also that a single seed and
spawn is a thin sample per town; the driving percentages are indicative, not
leaderboard numbers. Recovering real closed-loop driving on the OOD maps needs
in-distribution checkpoints or the CARLA leaderboard scenario framework, not more
spawn or creep tuning.

### Anti-crawl ablation

Clean-run route completion on Town10HD_Opt (seed 42, 120 frames), with and
without the moving-ego protocol. NEAT needs no aid; every other model is
near-stationary without it, which is why the protocol exists.

| Model | Without anti-crawl / creep | With the protocol | Aid used |
| --- | ---: | ---: | --- |
| NEAT | ~85% | (not used) | none — drives natively |
| AIM | ~0.7% | ~89% | generic `--anti-crawl` throttle burst |
| CILRS | ~0.0% | ~85% | generic `--anti-crawl` throttle burst |
| TCP | ~1.2% | ~87% | generic `--anti-crawl` throttle burst |
| TransFuser | ~1.0% | ~85% | its **own** creep controller, engaged earlier |

Read this as: **without the protocol the ego barely moves, so stage deviations
would be measured on a near-stationary vehicle and would not describe pipeline
robustness at all.** With it, clean and stress runs share the same protocol and
the same seed/route, and the recorded `control` stage still holds each model's
raw output (the nudge only changes the *applied* actuation), so the clean/stress
comparison stays a model measurement. For TransFuser the protocol changes only
*when* the model's own creep controller engages, not its predictions.

Regenerate any of these with `sd2 analyze` + `sd2 fingerprint` / `sd2 aggregate`
(commands in the model sections below).

## Validation: Synthetic Fault Injection Benchmark

SD2 includes a synthetic fault-injection benchmark for validating the diagnosis
framework itself. This is a framework sanity check, not a real-model experiment:
it creates clean/stress JSONL run pairs where the primary failure stage is known
by construction, runs the full `run_analysis` pipeline, and scores
`diagnosis.json` against the label.

The five labeled fault classes are:

- `vision`: large visual embedding cosine deviation
- `semantic`: object-set collapse plus critical-object and traffic-light flips
- `reasoning`: intent/text/critical-object mention mismatch
- `planning`: waypoint and target-speed divergence
- `control`: steer/throttle/brake command spike

Run the benchmark with:

```powershell
sd2 benchmark --config configs/mvp.yaml --output outputs/fault_benchmark --n-per-class 20 --seed 42
```

Or run the demo wrapper:

```powershell
python experiments/run_fault_benchmark.py
```

The default demo writes `benchmark_result.json`, `benchmark_report.md`, and
`confusion_matrix.png`. The current example result is 100.0% overall accuracy
with 100.0% per-class accuracy on all five synthetic classes; this indicates
that the implemented diagnosis policy matches the controlled synthetic origins.
See [docs/example/benchmark_report.md](docs/example/benchmark_report.md) and
the embedded confusion heatmap:

![Synthetic benchmark confusion matrix](docs/example/confusion_matrix.png)

### Hard Benchmark Tier

The benchmark also has a hard profile with competing and ambiguous faults:
competing collapse, strong propagation, near-simultaneous adjacent collapse,
and noisy upstream distractors. These samples are labeled by the intended
origin stage, and near-simultaneous cases carry `ambiguous: true` in
`label.json`. The hard tier is meant to be discriminative, so accuracy below
100% is expected and should be reported honestly.

```powershell
sd2 benchmark --config configs/mvp.yaml --output outputs/fault_benchmark_hard --profile hard --n-per-class 20 --seed 42
```

Hard reports keep the confusion matrix and add per-ambiguity-type accuracy plus
an ambiguous-only accuracy slice.

### Reasoning Metric (optional stage): Ablations and Known Limitations

> The `reasoning` stage is **optional** and applies only to language-based
> driving agents that emit text. None of the E2E models benchmarked above expose
> it, and it is not part of the core `perception → scene representation →
> planning → control → outcome` pipeline. This section is retained for
> language-agent adapters.

The default reasoning metric remains `text_embedding_and_intent` with weights
`text_embedding=0.5`, `intent_mismatch=0.3`, and
`critical_object_mismatch=0.2`. Three ablation variants are registered for
review analysis: `reasoning_intent_only`, `reasoning_text_only`, and
`reasoning_critical_object_only`.

The current `text_embedding` component is a token-set Jaccard distance, not a
semantic embedding. It is intentionally documented as paraphrase-fragile:
same-meaning rewrites can score as large lexical deviations, while small word
edits can hide decision changes. Intent weighting mitigates this for the MVP,
and the ablation probe motivates a future embedding or judge-based upgrade.
See [docs/example/reasoning_ablation.md](docs/example/reasoning_ablation.md).

## Quickstart

Create and activate the conda environment:

```powershell
conda create -n sd2 python=3.12 -y
conda activate sd2
```

Install the package in editable mode:

```powershell
pip install -e .
```

Run the one-command MVP demo:

```powershell
python experiments/run_mvp.py
```

Or run analysis and report generation directly:

```powershell
python -m sd2.cli analyze --clean data/sample/clean_run.jsonl --stress data/sample/stress_run.jsonl --config configs/mvp.yaml --output outputs/sample_analysis --report
```

Calibrate warning/critical thresholds from repeated clean runs:

```powershell
python -m sd2.cli calibrate --clean clean_a.jsonl --clean clean_b.jsonl --clean clean_c.jsonl --config configs/mvp.yaml --output outputs/calibration
```

Consume calibrated per-stage thresholds during analysis:

```powershell
python -m sd2.cli analyze --clean data/sample/clean_run.jsonl --stress data/sample/stress_run.jsonl --config configs/mvp.yaml --thresholds outputs/calibration/calibrated_thresholds.json --output outputs/sample_analysis_calibrated --report
```

Generate a report from an existing analysis directory:

```powershell
python -m sd2.cli report --analysis-dir outputs/sample_analysis
```

Aggregate one or more fingerprint outputs:

```powershell
python -m sd2.cli fingerprint --analysis-dir outputs --output outputs/fingerprint_summary.md
```

Generate deterministic sample images and apply a stressor:

```powershell
python experiments/generate_sample_images.py
python -m sd2.cli stress --input data/sample/images --config configs/stress/gaussian_noise.yaml --output outputs/stress_demo --seed 42
```

Run tests:

```powershell
conda run -n sd2 python -m pytest -q
```

If Windows temp permissions interfere with pytest, use:

```powershell
conda run -n sd2 python -m pytest -q --basetemp .pytest_basetemp
```

## Pairing Anchors

Clean and stress runs are paired by `pairing.mode` in the YAML config. The
default is `frame_idx`, which preserves the original MVP behavior: only equal
frame indices are paired, and the pair key remains
`model_id:scenario_id:seed:<clean_frame_idx>`.

Two alternate clean-centric anchors are available for closed-loop runs where
stress can change the ego trajectory. `timestamp` pairs each clean frame with
the nearest stress timestamp within `pairing.timestamp_tolerance` seconds.
`route_progress` pairs each clean frame with the nearest stress
`outcome.route_progress` within `pairing.progress_tolerance`; this is useful
when heavy stress makes the ego lag, drift, or reach intersections at different
frame numbers. Route-progress mode requires `outcome.route_progress` on both
runs; otherwise use `frame_idx`.

For every mode, emitted pairs keep the clean frame's `frame_idx` and
`timestamp`, so deviation timelines, propagation, and onset logic remain on the
clean-run timeline. `pairing_summary.json` reports `mode`,
`mean_anchor_mismatch`, and `max_anchor_mismatch`; units are frame-index delta
for `frame_idx` (always `0.0`), seconds for `timestamp`, and route-progress
fraction for `route_progress`. This addresses the known caveat that pure
frame-index pairing can misalign comparable driving states under heavy stress.

## CARLA Logging (Real Closed-Loop)

CARLA is not a core package dependency because its wheel is local and
platform-specific. Install CARLA 0.9.16 manually inside the `sd2` environment:

```powershell
pip install external/Carla/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-*.whl
```

The recording client drives with CARLA's `BasicAgent`, which requires two extra
packages:

```powershell
pip install shapely networkx
```

Launch the CARLA server from the CARLA install directory:

```powershell
CarlaUE4.exe -quality-level=Low -RenderOffScreen -carla-rpc-port=2000
```

Record a clean run:

```powershell
python experiments/carla_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 200 --warmup 20 --seed 42 --delta 0.05 --stress none --output data/carla/town10_clean_seed42.jsonl --spawn-index 0
```

Record a matched `control_noise` stress run with the same seed, town, frame
count, and spawn index:

```powershell
python experiments/carla_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 200 --warmup 20 --seed 42 --delta 0.05 --stress control_noise --stress-severity 3 --output data/carla/town10_control_noise_s3_seed42.jsonl --spawn-index 0
```

Analyze the pair:

```powershell
sd2 analyze --clean data/carla/town10_clean_seed42.jsonl --stress data/carla/town10_control_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/carla_control_noise_s3 --report
```

The CARLA recorder currently populates only Planning, Control, and Outcome
states: waypoints, target speed, ego pose/speed, vehicle controls, collision,
lane invasion, route progress, and optional TTC. Vision, Semantic, and
Reasoning are intentionally absent until a model adapter is available, so these
logs are Observability Tier 0/1. Clean and stress runs are designed to pair by
`frame_idx`; severe weather or control noise can still make the closed-loop
trajectory diverge, so frame pairing is an alignment convention for analysis.

## E2E Model Diagnosis (InterFuser)

SD2 can record InterFuser, an E2E camera+lidar model, in CARLA and emit Tier
2/3 logs with Vision, Semantic, Planning, Control, and Outcome populated. The
recorder is [experiments/interfuser_record.py](experiments/interfuser_record.py);
the CARLA-free conversion module is
[src/sd2/adapters/interfuser_adapter.py](src/sd2/adapters/interfuser_adapter.py).

`models/InterFuser/` is expected to be a local junction to the InterFuser repo
and remains gitignored. The script applies the verified inference preamble
internally: it stubs `imgaug`, prepends `models/InterFuser/interfuser` so the
vendored `timm 0.4.13` wins, and prepends the InterFuser `leaderboard` and
`scenario_runner` paths. Point `--checkpoint` at your own InterFuser weights,
e.g. via an environment variable:

```bash
export INTERFUSER_CKPT=/path/to/interfuser.pth
```

The recorder attaches the InterFuser sensor rig from the leaderboard agent:
front RGB `800x600` fov `100`, left/right RGB `400x300` yaw `-60/+60`, lidar
`ray_cast` yaw `-90`, IMU, GNSS, and a speedometer measurement derived from the
ego velocity. Visual stressors are applied to RGB frames before InterFuser
preprocessing and inference, so the perturbation can propagate through
semantic prediction, planning, control, and outcome.

Record a clean InterFuser run:

```powershell
python experiments/interfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint "$INTERFUSER_CKPT" --stress none --output data/carla/interfuser_town10_clean_seed42.jsonl --spawn-index 0
```

Record a matched Gaussian-noise stress run with the same seed, town, frame
count, and spawn index:

```powershell
python experiments/interfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint "$INTERFUSER_CKPT" --stress gaussian_noise --stress-severity 3 --output data/carla/interfuser_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
```

Analyze the pair:

```powershell
sd2 analyze --clean data/carla/interfuser_town10_clean_seed42.jsonl --stress data/carla/interfuser_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/interfuser_town10_gaussian_noise_s3 --report
```

Stage mapping:

- `vision`: mean-pooled InterFuser `traffic_feature`/BEV feature as `feature`
  for `embedding_cosine`, plus front-camera `image_mean` and `image_std`
  fallback.
- `semantic`: tracked `traffic_meta` object counts/classes, occupied-cell
  density, junction probability, traffic-light score, and stop-sign score.
- `planning`: predicted waypoints, controller target speed, route command, and
  local target point.
- `control`: `InterfuserController` steer, throttle, and brake.
- `outcome`: CARLA collision and lane-invasion events, route progress, and
  optional TTC placeholder.

The script logs first-tick sensor shapes, model input tensor shapes, and model
output shapes before recording frames, which is the first place to look if a
live CARLA run has an input-shape mismatch.

### TransFuser

SD2 also records TransFuser as a second E2E architecture for RQ3-style
cross-architecture failure comparison: clean/stress runs can be collected with
the same CARLA town, seed, route, frame count, and visual stressor, then
compared against InterFuser fingerprints using the same SD2 stage schema.

The recorder is
[experiments/transfuser_record.py](experiments/transfuser_record.py); the
CARLA-free conversion module is
[src/sd2/adapters/transfuser_adapter.py](src/sd2/adapters/transfuser_adapter.py).
`models/TransFuser/` is expected to be a local gitignored junction to the
TransFuser checkout. The default checkpoint directory is:

```text
models/TransFuser/checkpoints/models_2022/transfuser
```

The script follows the verified TransFuser load recipe: it prepends only
`models/TransFuser/TransFuser_UI_V2/transfuser/team_code_transfuser`, imports
`GlobalConfig` and `LidarCenterNet`, reads `args.txt`, loads
`model_seed1_39.pth`, strips the `module.` DDP prefix, and uses the installed
standard `timm` package. It does not prepend InterFuser's vendored `timm` path;
`mmcv` and `torch_scatter` remain optional because the TransFuser model file has
fallbacks for this inference path.

The recorder attaches the TransFuser sensor rig from the submission agent:
front/left/right RGB cameras `960x480` fov `120` at yaw `0/-60/+60`, IMU, GNSS,
a speedometer measurement derived from ego velocity, and lidar `ray_cast` at the
configured lidar pose for the `transFuser` backbone. Visual stressors are
applied to the RGB camera frames before TransFuser preprocessing, target-point
image generation, `forward_ego`, and `control_pid`.

Record a clean TransFuser run:

```powershell
python experiments/transfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/TransFuser/checkpoints/models_2022/transfuser --stress none --output data/carla/transfuser_town10_clean_seed42.jsonl --spawn-index 0
```

Record a matched Gaussian-noise stress run with the same seed, town, frame
count, and spawn index:

```powershell
python experiments/transfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/TransFuser/checkpoints/models_2022/transfuser --stress gaussian_noise --stress-severity 3 --output data/carla/transfuser_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
```

Analyze the pair:

```powershell
sd2 analyze --clean data/carla/transfuser_town10_clean_seed42.jsonl --stress data/carla/transfuser_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/transfuser_town10_gaussian_noise_s3 --report
```

Aggregate InterFuser and TransFuser fingerprints for cross-model comparison:

```powershell
sd2 fingerprint --analysis-dir outputs --output outputs/e2e_fingerprint_summary.md
```

Stage mapping:

- `vision`: TransFuser fused image/LiDAR backbone embedding before the waypoint
  GRU as `feature` for `embedding_cosine`, plus three-camera `image_mean` and
  `image_std` fallback.
- `semantic`: `rotated_bb` detections from the CenterNet branch, converted to
  vehicle objects, per-class counts, occupancy/density summary, confidence
  summary, and optional BEV segmentation summary.
- `planning`: `pred_wp` waypoints, target-speed proxy from waypoint spacing,
  route command, local target point, and stuck-state flag.
- `control`: `LidarCenterNet.control_pid` steer, throttle, and brake after the
  TransFuser action-repeat/stuck/safety logic.
- `outcome`: CARLA collision and lane-invasion events, route progress, and
  optional TTC placeholder.

The TransFuser and InterFuser adapters both emit Vision, Semantic, Planning,
Control, and Outcome with the same SD2 stage names, so `sd2 analyze` and
`sd2 fingerprint` can compare the two architectures under matched visual stress.

#### Making TransFuser drive (anti-crawl creep)

Out of the box in open-world closed loop (no leaderboard scenario), TransFuser —
like the AIM/CILRS/TCP baselines — falls into a **cold-start crawl limit-cycle**:
from a standstill the model predicts short waypoints, so `control_pid` derives a
low desired speed and brakes; the ego briefly creeps, overshoots the tiny
desired speed, brakes hard, and stops again. Route progress stalls near zero and
the planning/control deviations end up measured on a near-stationary ego.

The `--debug-driving` diagnosis (per-tick `speed`, `is_stuck`, `stuck_detector`,
`forced_move`, `emergency_stop`, safety-box point count, `target_point`, and the
first/last predicted waypoint) confirmed the cause on a live run: the LiDAR
safety brake never fires (`emergency_stop=False`, `safety_pts=0`) and the
`target_point`/waypoint frames are correct — the ego is simply trapped by its own
low predicted speed. TransFuser already ships a **creep controller** (it forces
`default_speed ≈ 4 m/s` while `is_stuck`), but its stuck trigger
(`stuck_threshold = 1100` frames of near-zero speed) never fires during a crawl.

The recorder exposes the creep so it can engage in the crawl regime:

- `--creep-speed S` — count a frame toward the stuck detector when speed `< S`
  (default `0.1` = original "only truly stopped"; set `2.5` to treat crawling as
  stuck). The detector only resets once the ego is clearly moving above `S`.
- `--creep-threshold N` — engage the creep after `N` sub-`creep-speed` frames
  (overrides `config.stuck_threshold`).
- `--creep-duration N` — how many frames each forced-move creep lasts
  (overrides `config.creep_duration`).
- `--debug-driving` / `--no-lidar-safe-check` remain available for diagnosis.

With the settings below, TransFuser drives the route at a sustained ~4 m/s and
completes ~85% of it (matching NEAT), so its stage deviations are measured on a
properly moving ego:

```powershell
# clean + gaussian-noise, anti-crawl creep engaged
python experiments/transfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 120 --warmup 20 --seed 42 --checkpoint models/TransFuser/checkpoints/models_2022/transfuser --stress none --creep-speed 2.5 --creep-threshold 5 --creep-duration 60 --output data/carla/transfuser_town10_clean_seed42.jsonl --spawn-index 0
python experiments/transfuser_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 120 --warmup 20 --seed 42 --checkpoint models/TransFuser/checkpoints/models_2022/transfuser --stress gaussian_noise --stress-severity 3 --creep-speed 2.5 --creep-threshold 5 --creep-duration 60 --output data/carla/transfuser_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
sd2 analyze --clean data/carla/transfuser_town10_clean_seed42.jsonl --stress data/carla/transfuser_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/transfuser_town10_gaussian_noise_s3 --report
```

The creep is TransFuser's own mechanism; `--creep-speed`/`--creep-threshold` only
change *when* it engages, not the model's predictions. The same cold-start crawl
affects the AIM/CILRS/TCP camera baselines (NEAT escapes it on its own); those
recorders instead take a generic `--anti-crawl` flag that nudges the *applied*
throttle to give the ego a rolling start — see the next section.

If run (2) drives but (1) does not, the LiDAR safety box is the culprit; if
`emergency_stop=False` throughout but `brake` stays high with sane waypoints,
the fix is in the target-point/route frame rather than the safety logic.

### Classic TransFuser-CVPR'21 Baselines

SD2 also records three classic baselines from the TransFuser-CVPR'21 codebase
using the same clean/stress pairing and SD2 stage schema:

- **AIM**: camera-only imitation model; SD2 observes front-image encoder
  features, predicted waypoints, PID control, and outcome. AIM has no semantic
  head, so the semantic stage is absent/unobserved.
- **CILRS**: camera-only conditional imitation model; SD2 observes front-image
  encoder features, predicted velocity as the planning target-speed signal,
  direct control, and outcome. CILRS has no semantic head, so the semantic stage
  is absent/unobserved.
- **NEAT**: attention-field model; SD2 observes multi-camera encoder features,
  decoded BEV occupancy semantics (`bev_seg_summary`), predicted waypoints, PID
  control, and outcome. NEAT contributes a BEV-seg semantic signal like
  TransFuser.
- **TCP**: Bench2Drive trajectory+control dual-branch model; SD2 observes
  front-image backbone features, `pred_wp` waypoints, final gated control plus
  raw trajectory/control branch actions, and outcome. TCP is fed a single front
  camera resized/sized to `256x900` instead of the original three-camera mosaic,
  and has no semantic head, so the semantic stage is absent/unobserved.

`models/AIM/`, `models/CILRS/`, `models/NEAT/`, and `models/TCP/` are expected
to be local gitignored junctions/checkouts. The default checkpoints are:

```text
models/AIM/aim/best_model.pth
models/CILRS/cilrs/best_model.pth
models/NEAT/neat/best_encoder.pth
models/NEAT/neat/best_decoder.pth
models/NEAT/neat/args.txt
models/TCP/checkpoints/tcp_b2d.ckpt
```

**Anti-crawl (recommended).** AIM, CILRS, and TCP have no native creep
controller, so from a standstill they fall into the same cold-start crawl
limit-cycle described above and route progress stalls near zero. Their recorders
accept a generic `--anti-crawl` flag that gives the ego a rolling start by
nudging the **applied** throttle in sustained bursts while it crawls; the
**recorded** control stage still holds the model's raw steer/throttle/brake, so
the clean-vs-stress control comparison stays a pure model measurement. With
`--anti-crawl --creep-speed 2.5 --creep-frames 4 --creep-throttle 0.6 --creep-duration 40`,
all three complete ~85–90% of the route at a moving speed. Flags:
`--creep-speed` (crawl threshold m/s), `--creep-frames` (crawl frames before a
burst), `--creep-throttle` (burst throttle), `--creep-duration` (burst length).

Record and analyze AIM (with anti-crawl):

```powershell
python experiments/aim_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/AIM/aim/best_model.pth --stress none --anti-crawl --creep-speed 2.5 --creep-frames 4 --creep-throttle 0.6 --creep-duration 40 --output data/carla/aim_town10_clean_seed42.jsonl --spawn-index 0
python experiments/aim_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/AIM/aim/best_model.pth --stress gaussian_noise --stress-severity 3 --output data/carla/aim_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
sd2 analyze --clean data/carla/aim_town10_clean_seed42.jsonl --stress data/carla/aim_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/aim_town10_gaussian_noise_s3 --report
```

Record and analyze CILRS:

```powershell
python experiments/cilrs_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/CILRS/cilrs/best_model.pth --stress none --output data/carla/cilrs_town10_clean_seed42.jsonl --spawn-index 0
python experiments/cilrs_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/CILRS/cilrs/best_model.pth --stress gaussian_noise --stress-severity 3 --output data/carla/cilrs_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
sd2 analyze --clean data/carla/cilrs_town10_clean_seed42.jsonl --stress data/carla/cilrs_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/cilrs_town10_gaussian_noise_s3 --report
```

Record and analyze NEAT:

```powershell
python experiments/neat_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/NEAT/neat --stress none --output data/carla/neat_town10_clean_seed42.jsonl --spawn-index 0
python experiments/neat_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/NEAT/neat --stress gaussian_noise --stress-severity 3 --output data/carla/neat_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
sd2 analyze --clean data/carla/neat_town10_clean_seed42.jsonl --stress data/carla/neat_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/neat_town10_gaussian_noise_s3 --report
```

Record and analyze TCP:

```powershell
python experiments/tcp_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/TCP/checkpoints/tcp_b2d.ckpt --planner-type only_traj --stress none --output data/carla/tcp_town10_clean_seed42.jsonl --spawn-index 0
python experiments/tcp_record.py --host localhost --port 2000 --town Town10HD_Opt --frames 300 --warmup 20 --seed 42 --delta 0.05 --checkpoint models/TCP/checkpoints/tcp_b2d.ckpt --planner-type only_traj --stress gaussian_noise --stress-severity 3 --output data/carla/tcp_town10_gaussian_noise_s3_seed42.jsonl --spawn-index 0
sd2 analyze --clean data/carla/tcp_town10_clean_seed42.jsonl --stress data/carla/tcp_town10_gaussian_noise_s3_seed42.jsonl --config configs/mvp.yaml --output outputs/tcp_town10_gaussian_noise_s3 --report
```

TCP stage mapping:

- `vision`: mean-pooled TCP ResNet/perception feature as `feature`, plus
  single-front-camera `image_mean` and `image_std` fallback.
- `semantic`: absent/unobserved; TCP has no explicit semantic head.
- `planning`: `pred_wp` future waypoints (`pred_len=4`), control-PID
  `desired_speed` as `target_speed`, route command, and local target point.
- `control`: final steer/throttle/brake after TCP planner selection, throttle
  clamp, and brake gating, with raw trajectory-branch and control-branch
  steer/throttle/brake preserved in `details`.
- `outcome`: CARLA collision and lane-invasion events, route progress, and
  optional TTC placeholder.

Aggregate all E2E fingerprints:

```powershell
sd2 fingerprint --analysis-dir outputs --output outputs/e2e_fingerprint_summary.md
sd2 aggregate --analysis-dir outputs --output outputs/e2e_aggregate.md
```

### Calibrated thresholds on real data

Real CARLA drives have natural run-to-run variation (engine non-determinism),
so repeated clean runs give a meaningful clean-clean baseline. Record several
clean runs and calibrate per-stage thresholds, then analyze the stress pair
with them:

```powershell
sd2 calibrate --clean data/carla/calib/clean_rep1.jsonl --clean data/carla/calib/clean_rep2.jsonl --clean data/carla/calib/clean_rep3.jsonl --config configs/mvp.yaml --output data/carla/calib/calibrated_thresholds.json
sd2 analyze --clean data/carla/town10_clean_seed42.jsonl --stress data/carla/town10_control_noise_s3_seed42.jsonl --config configs/mvp.yaml --thresholds data/carla/calib/calibrated_thresholds.json --output outputs/carla_control_noise_s3_calibrated --report
```

On the bundled `control_noise` example this matters: the static `0.4/0.7`
thresholds are far above the real clean-clean deviation (Planning/Control
critical calibrate to about `0.06-0.07`), and a single-frame Planning spike
would otherwise be mistaken for the primary failure. With calibrated thresholds
plus the `onset_persistence_frames` requirement (a collapse onset must persist
for several consecutive frames, filtering outlier spikes), the diagnosis
correctly identifies **Control** — the stage that was actually perturbed — as
the primary failure stage.

## Expected Outputs

The demo writes:

- `outputs/sample_analysis/paired_frames.json`
- `outputs/sample_analysis/pairing_summary.json`
- `outputs/sample_analysis/deviation_table.json`
- `outputs/sample_analysis/deviation_table.csv`
- `outputs/sample_analysis/propagation.json`
- `outputs/sample_analysis/diagnosis.json`
- `outputs/sample_analysis/fingerprint.json`
- `outputs/sample_analysis/report.md`
- `outputs/sample_analysis/plots/deviation_timeline.png`
- `outputs/sample_analysis/plots/robustness_fingerprint.png`
- `outputs/sample_analysis/plots/propagation_scores.png`

Calibration writes `calibrated_thresholds.json`, containing per-stage clean-clean mean/std, warning/critical thresholds computed as `mean + k * std`, and fallback flags for stages whose clean-clean variance is near zero.

A copy of the demo output (report and plots) is kept under [docs/example/](docs/example/) for reference.

## Stressors

Stressors perturb clean image inputs to produce offline stress-run inputs. Severity is an integer from `1` to `5`; `0` and out-of-range values are rejected. Each stressor maps that severity to concrete parameters internally and records those parameters in `stress_manifest.json`.

Visual stressors operate on `HxWx3` RGB `uint8` images and write the same filenames to the output directory:

- `gaussian_noise`
- `motion_blur`
- `fog`
- `brightness_shift`
- `contrast_shift`
- `jpeg_compression`
- `low_light`

Temporal stressors operate on the sorted image list as a frame sequence:

- `frame_drop`
- `frame_delay`
- `camera_blackout`
- `low_fps`

For temporal materialization, dropped frames are omitted, camera blackouts are written as black images, and delayed frames hold earlier source images at the current output position. Run a stress pass with:

```powershell
python -m sd2.cli stress --input <image-dir> --config <stress-yaml> --output <output-dir> --seed 42
```

Existing stress configs live in `configs/stress/`, for example `gaussian_noise.yaml`, `motion_blur.yaml`, and `frame_drop.yaml`.

## Current Status

MVP Phase 1 through the offline stressor layer are complete:

- src-layout Python package scaffold
- Pydantic v2 run and frame schema
- deterministic JSONL sample data
- deterministic sample image generator for stressor demos
- JSONL run loader with line-numbered validation errors
- clean/stress frame pairing with skipped-frame summary and saved run metadata
- stage-wise metric registry and MVP metrics for vision, semantic, planning, and control stages (plus an optional reasoning stage for language-based agents)
- visual and temporal stressor registry with `sd2 stress` CLI materialization
- min-max clipping and threshold status classification (`healthy`, `warning`, `critical`)
- optional clean-clean threshold calibration with `sd2 calibrate` and `sd2 analyze --thresholds`
- propagation analysis with adjacent-stage robust evidence bundles: legacy ratio, clipped ratio, log-ratio, absolute increase, collapse order, and downstream persistence
- temporal-correlational failure-stage labeling using `first_critical_with_downstream_increase`, with documented fallbacks
- per-run robustness fingerprint where each observed stage score is `1 - mean(normalized deviation)`
- Markdown report generation with stage timeline, fingerprint, and propagation plots
- `sd2 analyze --report`, `sd2 report`, and `sd2 fingerprint` CLI flows
- `experiments/run_mvp.py` one-command demo
- labeled synthetic fault-injection benchmark with `sd2 benchmark`
- hard/ambiguous synthetic benchmark profile with per-ambiguity reporting
- optional-stage reasoning metric ablations and paraphrase-robustness probe
  (language-based agents only; unused by the E2E experiments)
- `experiments/run_fault_benchmark.py` one-command validation demo
- CARLA InterFuser, TransFuser, AIM, CILRS, NEAT, and TCP E2E recorders plus pure
  SD2 adapters for stage-wise diagnosis
- observed-stage and common-stage robustness means (`sd2 fingerprint`) and
  multi-seed statistical robustness (`sd2 aggregate`)

The synthetic benchmark validates the SD2 diagnosis machinery on controlled
offline logs; it does not replace real-model robustness experiments.

## Metric Config

Metrics are selected per stage in `configs/mvp.yaml` under `metrics`:

- `embedding_cosine` for `vision`: cosine distance over `embedding` or `feature`.
- `object_jaccard` for `semantic`: object-set Jaccard distance with missing/extra object, critical-object mismatch, and traffic-light mismatch details.
- `text_embedding_and_intent` for `reasoning`: weighted lexical token-set distance, intent mismatch, and critical-object mention mismatch.
- `waypoint_ade` for `planning`: ADE over common waypoint prefix, with FDE and target-speed difference details.
- `weighted_action_mae` for `control`: weighted absolute steer/throttle/brake error.
