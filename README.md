# SD2

SD2 (System Deviation Diagnosis) is an offline analysis framework for studying how stress conditions affect functional stages in vision-language autonomous driving systems. The MVP starts with stored clean and stress run logs, validates them against a shared schema, pairs matching frames, and writes stage-wise deviation artifacts.

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

Phase 1 through Phase 4 are complete for the MVP:

- src-layout Python package scaffold
- Pydantic v2 run and frame schema
- deterministic JSONL sample data
- JSONL run loader with line-numbered validation errors
- clean/stress frame pairing with skipped-frame summary
- stage-wise metric registry and MVP metrics for vision, semantic, reasoning, planning, and control stages
- min-max clipping and threshold status classification (`healthy`, `warning`, `critical`)
- `sd2 analyze` CLI that writes `paired_frames.json`, `pairing_summary.json`, `deviation_table.json`, and `deviation_table.csv`

Outcome-stage metrics, CARLA integration, model-specific adapters, propagation analysis, diagnosis, plots, and reports are reserved for later phases.

## Metric Config

Metrics are selected per stage in `configs/mvp.yaml` under `metrics`:

- `embedding_cosine` for `vision`: cosine distance over `embedding` or `feature`.
- `object_jaccard` for `semantic`: object-set Jaccard distance with missing/extra object, critical-object mismatch, and traffic-light mismatch details.
- `text_embedding_and_intent` for `reasoning`: weighted lexical token-set distance, intent mismatch, and critical-object mention mismatch. Supports `weights.text_embedding`, `weights.intent_mismatch`, and `weights.critical_object_mismatch`; weights are normalized to sum to 1.
- `waypoint_ade` for `planning`: ADE over common waypoint prefix, with FDE and target-speed difference details. Supports `ade_scale` for ADE normalization.
- `weighted_action_mae` for `control`: weighted absolute steer/throttle/brake error. Supports `weights.steer`, `weights.throttle`, and `weights.brake`; steering differences are normalized by the `[-1, 1]` range before weighting.
