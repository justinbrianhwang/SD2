# SD2

SD2 (System Deviation Diagnosis) is an offline analysis framework for studying how stress conditions affect functional stages in vision-language autonomous driving systems. The MVP reads stored clean and stress run logs, pairs matching frames, computes stage-wise deviation and downstream temporal evidence, labels the primary failure stage, and generates a Markdown report with plots.

Instead of asking *"how well does this model drive?"*, SD2 asks *"where in the pipeline does robustness collapse, and how does the error propagate?"*

Diagnosis outputs are **temporal-correlational**. They identify the earliest stage whose deviation crosses configured thresholds and is temporally followed by downstream deviation and/or driving failure evidence. They do not prove a mechanistic root explanation.

## Example Output

Running the bundled demo on the sample logs (Gaussian noise, severity 3) produces a per-stage **robustness fingerprint** — higher is more robust:

![Robustness fingerprint](docs/example/robustness_fingerprint.png)

The **deviation timeline** shows where robustness collapses first: vision stays stable while reasoning crosses the critical threshold at t=1.5s, followed by planning and control drift:

![Stage-wise deviation timeline](docs/example/deviation_timeline.png)

From this, the diagnosis module generates a natural-language summary:

> Under Gaussian Noise severity 3, the openemma model completed 92.0% of the route and experienced a collision and a lane invasion. The Reasoning stage showed the earliest critical deviation at t=1.500s (frame 15), preceding downstream Planning/Control deviation and the final driving failure. The primary_failure_stage label is Reasoning.

`diagnosis.json` includes `"diagnosis_type": "temporal_correlational"` to make this framing explicit.

See the full generated report at [docs/example/example_report.md](docs/example/example_report.md).

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

### Reasoning Metric: Ablations and Known Limitations

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
- stage-wise metric registry and MVP metrics for vision, semantic, reasoning, planning, and control stages
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
- reasoning metric ablations and paraphrase-robustness probe
- `experiments/run_fault_benchmark.py` one-command validation demo

CARLA integration and model-specific closed-loop adapters remain out of scope for the offline MVP. The synthetic benchmark validates the SD2 diagnosis machinery on controlled offline logs; it does not replace real-model robustness experiments.

## Metric Config

Metrics are selected per stage in `configs/mvp.yaml` under `metrics`:

- `embedding_cosine` for `vision`: cosine distance over `embedding` or `feature`.
- `object_jaccard` for `semantic`: object-set Jaccard distance with missing/extra object, critical-object mismatch, and traffic-light mismatch details.
- `text_embedding_and_intent` for `reasoning`: weighted lexical token-set distance, intent mismatch, and critical-object mention mismatch.
- `waypoint_ade` for `planning`: ADE over common waypoint prefix, with FDE and target-speed difference details.
- `weighted_action_mae` for `control`: weighted absolute steer/throttle/brake error.
