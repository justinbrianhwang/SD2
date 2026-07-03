# SD2

SD2 (System Deviation Diagnosis) is an offline analysis framework for studying how stress conditions affect functional stages in vision-language autonomous driving systems. The MVP starts with stored clean and stress run logs, validates them against a shared schema, pairs matching frames, and writes stage-wise deviation, propagation, diagnosis, and fingerprint artifacts.

## Quickstart

Create or activate the expected conda environment:

```powershell
conda activate sd2
```

Install the package when working outside the source checkout:

```powershell
pip install -e .
```

Run the sample analysis:

```powershell
sd2 analyze --clean data/sample/clean_run.jsonl --stress data/sample/stress_run.jsonl --config configs/mvp.yaml --output outputs/sample_analysis
```

From an uninstalled checkout, this also works:

```powershell
python -m sd2.cli analyze --clean data/sample/clean_run.jsonl --stress data/sample/stress_run.jsonl --config configs/mvp.yaml --output outputs/sample_analysis
python -m sd2 --version
```

Run tests:

```powershell
conda run -n sd2 python -m pytest -q
```

## Current Status

Phase 1 through Phase 6 are complete for the MVP:

- src-layout Python package scaffold
- Pydantic v2 run and frame schema
- deterministic JSONL sample data
- JSONL run loader with line-numbered validation errors
- clean/stress frame pairing with skipped-frame summary
- stage-wise metric registry and MVP metrics for vision, semantic, reasoning, planning, and control stages
- min-max clipping and threshold status classification (`healthy`, `warning`, `critical`)
- propagation analysis with adjacent-stage scores, warning/critical collapse onsets, and downstream-increase evidence
- failure diagnosis using `first_critical_with_downstream_increase`, with fallbacks to earliest warning with downstream increase, highest mean-deviation stage, then `no_failure_detected`
- per-run robustness fingerprint where each observed stage score is `1 - mean(normalized deviation)`
- `sd2 analyze` CLI that writes `paired_frames.json`, `pairing_summary.json`, `deviation_table.json`, `deviation_table.csv`, `propagation.json`, `diagnosis.json`, and `fingerprint.json`

Outcome-stage metrics, CARLA integration, model-specific adapters, plots, and reports are reserved for later phases.

## Analysis Outputs

`propagation.json` contains per-frame and aggregate adjacent-stage propagation scores, per-stage collapse onsets for warning and critical thresholds, and before/after downstream-increase evidence.

`diagnosis.json` identifies the primary failure stage, records the policy and fallback used, compares clean versus stress outcomes, reports whether a driving failure occurred, and includes human-readable evidence for the decision.

`fingerprint.json` reports stage robustness scores from 0 to 1, with missing stages written as `null` rather than zero.

## Metric Config

Metrics are selected per stage in `configs/mvp.yaml` under `metrics`:

- `embedding_cosine` for `vision`: cosine distance over `embedding` or `feature`.
- `object_jaccard` for `semantic`: object-set Jaccard distance with missing/extra object, critical-object mismatch, and traffic-light mismatch details.
- `text_embedding_and_intent` for `reasoning`: weighted lexical token-set distance, intent mismatch, and critical-object mention mismatch. Supports `weights.text_embedding`, `weights.intent_mismatch`, and `weights.critical_object_mismatch`; weights are normalized to sum to 1.
- `waypoint_ade` for `planning`: ADE over common waypoint prefix, with FDE and target-speed difference details. Supports `ade_scale` for ADE normalization.
- `weighted_action_mae` for `control`: weighted absolute steer/throttle/brake error. Supports `weights.steer`, `weights.throttle`, and `weights.brake`; steering differences are normalized by the `[-1, 1]` range before weighting.
