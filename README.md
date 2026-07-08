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
- `experiments/run_fault_benchmark.py` one-command validation demo

CARLA integration and model-specific closed-loop adapters remain out of scope for the offline MVP. The synthetic benchmark validates the SD2 diagnosis machinery on controlled offline logs; it does not replace real-model robustness experiments.

## Metric Config

Metrics are selected per stage in `configs/mvp.yaml` under `metrics`:

- `embedding_cosine` for `vision`: cosine distance over `embedding` or `feature`.
- `object_jaccard` for `semantic`: object-set Jaccard distance with missing/extra object, critical-object mismatch, and traffic-light mismatch details.
- `text_embedding_and_intent` for `reasoning`: weighted lexical token-set distance, intent mismatch, and critical-object mention mismatch.
- `waypoint_ade` for `planning`: ADE over common waypoint prefix, with FDE and target-speed difference details.
- `weighted_action_mae` for `control`: weighted absolute steer/throttle/brake error.
