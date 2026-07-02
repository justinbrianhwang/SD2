# SD2 Data Format

The MVP uses one JSONL file per run.

The first non-empty line must be tagged run metadata:

```json
{"type": "run_metadata", "run_id": "openemma_town05_route01_clean_seed42", "model_id": "openemma", "scenario_id": "town05_route01", "condition": "clean", "stress_type": null, "severity": 0, "seed": 42, "timestamp_start": "2026-01-01T00:00:00"}
```

Every subsequent non-empty line must be tagged as a frame:

```json
{"type": "frame", "run_id": "openemma_town05_route01_clean_seed42", "frame_idx": 0, "timestamp": 0.0, "states": {"control": {"steer": 0.0, "throttle": 0.2, "brake": 0.0}}}
```

Only stages present in the log need to be included. Valid stage keys are `vision`, `semantic`, `reasoning`, `planning`, `control`, and `outcome`.
