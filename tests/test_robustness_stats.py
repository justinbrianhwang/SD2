import json
from math import sqrt
from pathlib import Path

import pytest

from sd2.analysis.deviation import DeviationRecord, DeviationTable
from sd2.analysis.robustness_stats import (
    aggregate_run_statistics,
    discover_analysis_dirs,
    format_statistical_report,
)
from sd2.core.config import SD2Config
from sd2.core.stage import Stage


def test_aggregate_run_statistics_computes_stage_stats_and_stability(
    tmp_path: Path,
) -> None:
    config = SD2Config.model_validate(
        {"stages": ["vision", "reasoning", "outcome"]}
    )
    run_a = _write_analysis_run(
        tmp_path / "run_a",
        {Stage.VISION: 0.9, Stage.REASONING: 0.4},
        "reasoning",
    )
    run_b = _write_analysis_run(
        tmp_path / "run_b",
        {Stage.VISION: 0.8, Stage.REASONING: 0.6},
        "reasoning",
    )
    run_c = _write_analysis_run(
        tmp_path / "run_c",
        {Stage.VISION: 0.7, Stage.REASONING: 0.8},
        "vision",
    )

    report = aggregate_run_statistics([run_a, run_b, run_c], config)

    assert report.run_count == 3

    vision = report.stage_stats[Stage.VISION]
    assert vision.n == 3
    assert vision.values == pytest.approx([0.9, 0.8, 0.7])
    assert vision.mean == pytest.approx(0.8)
    assert vision.std == pytest.approx(0.1)
    assert vision.ci95_low == pytest.approx(0.8 - (1.96 * 0.1 / sqrt(3)))
    assert vision.ci95_high == pytest.approx(0.8 + (1.96 * 0.1 / sqrt(3)))
    assert vision.min == pytest.approx(0.7)
    assert vision.max == pytest.approx(0.9)

    reasoning = report.stage_stats[Stage.REASONING]
    assert reasoning.mean == pytest.approx(0.6)
    assert reasoning.std == pytest.approx(0.2)

    overall = report.mean_robustness_stat
    assert overall.n == 3
    assert overall.values == pytest.approx([0.65, 0.7, 0.75])
    assert overall.mean == pytest.approx(0.7)
    assert overall.std == pytest.approx(0.05)
    assert overall.ci95_low == pytest.approx(0.7 - (1.96 * 0.05 / sqrt(3)))
    assert overall.ci95_high == pytest.approx(0.7 + (1.96 * 0.05 / sqrt(3)))

    stability = report.diagnosis_stability
    assert stability.primary_stage_counts == {"reasoning": 2, "vision": 1}
    assert stability.modal_stage == "reasoning"
    assert stability.stability == pytest.approx(2 / 3)
    assert stability.n_runs == 3

    markdown = format_statistical_report(report)
    assert "| Stage | n | mean robustness | std | 95% CI |" in markdown
    assert "| Vision | 3 | 0.800 | 0.100 | [0.687, 0.913] |" in markdown
    assert "Overall mean robustness: 0.700 +/- 0.050" in markdown
    assert "primary stage = reasoning in 2/3 runs (stability 0.667)" in markdown
    assert "counts: reasoning=2, vision=1" in markdown

    payload = report.to_dict()
    assert payload["stage_stats"]["vision"]["mean"] == pytest.approx(0.8)
    assert payload["mean_robustness_stat"]["stage"] == "mean_robustness"

    output_path = tmp_path / "stats.json"
    report.write_json(output_path)
    round_trip = json.loads(output_path.read_text(encoding="utf-8"))
    assert round_trip["run_count"] == 3
    assert round_trip["diagnosis_stability"]["stability"] == pytest.approx(2 / 3)


def test_discover_analysis_dirs_finds_root_or_immediate_children(
    tmp_path: Path,
) -> None:
    run_a = _write_analysis_run(
        tmp_path / "run_a",
        {Stage.VISION: 0.9},
        "vision",
    )
    run_b = _write_analysis_run(
        tmp_path / "run_b",
        {Stage.VISION: 0.8},
        "vision",
    )
    nested = tmp_path / "nested" / "run_c"
    _write_analysis_run(nested, {Stage.VISION: 0.7}, "vision")

    assert discover_analysis_dirs(run_a) == [run_a]
    assert discover_analysis_dirs(tmp_path) == [run_a, run_b]


def _write_analysis_run(
    path: Path,
    robustness_scores: dict[Stage, float],
    primary_stage: str | None,
) -> Path:
    path.mkdir(parents=True)
    records = []
    for index, (stage, robustness_score) in enumerate(robustness_scores.items()):
        deviation_score = 1.0 - robustness_score
        records.append(
            DeviationRecord(
                pair_key=f"pair:{index}",
                frame_idx=0,
                timestamp=0.0,
                stage=stage,
                metric="synthetic",
                raw_score=deviation_score,
                normalized_score=deviation_score,
                status="healthy",
                missing=False,
                details={},
            )
        )
    DeviationTable(records).write_json(path / "deviation_table.json")
    (path / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": primary_stage}) + "\n",
        encoding="utf-8",
    )
    return path
