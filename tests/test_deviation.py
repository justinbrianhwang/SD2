from collections import Counter

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.analysis.deviation import classify_status, compute_deviation_table
from sd2.core.config import load_config
from sd2.core.run import pair_runs


def test_deviation_table_sample_data_shape_and_statuses() -> None:
    config = load_config("configs/mvp.yaml")
    clean = load_run_jsonl("data/sample/clean_run.jsonl")
    stress = load_run_jsonl("data/sample/stress_run.jsonl")
    paired_run = pair_runs(clean, stress)

    table = compute_deviation_table(paired_run, config)
    records = table.to_records()

    assert len(records) == 30 * 5
    assert Counter(record["stage"] for record in records) == {
        "vision": 30,
        "semantic": 30,
        "reasoning": 30,
        "planning": 30,
        "control": 30,
    }
    assert all(record["missing"] is False for record in records)
    assert all(
        record["status"] == classify_status(
            record["normalized_score"],
            config.thresholds,
        )
        for record in records
    )

    dataframe = table.to_dataframe()
    assert len(dataframe) == 30 * 5
    assert dataframe.loc[0, "details"].startswith("{")
