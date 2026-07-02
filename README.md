# SD2

SD2 (System Deviation Diagnosis) is an offline analysis framework for studying how stress conditions affect functional stages in vision-language autonomous driving systems. The MVP starts with stored clean and stress run logs, validates them against a shared schema, pairs matching frames, and writes pairing artifacts for later deviation analysis.

## Quickstart

Create or activate the expected conda environment:

```powershell
conda activate sd2
```

Install the package when working outside the source checkout:

```powershell
pip install -e .
```

Run the sample pairing analysis:

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

Phase 1 and Phase 2 are complete for the MVP:

- src-layout Python package scaffold
- Pydantic v2 run and frame schema
- deterministic JSONL sample data
- JSONL run loader with line-numbered validation errors
- clean/stress frame pairing with skipped-frame summary
- `sd2 analyze` CLI that writes `paired_frames.json` and `pairing_summary.json`

CARLA integration, model-specific adapters, stage-wise metrics, propagation analysis, diagnosis, plots, and reports are reserved for later phases.
