from math import isfinite, log

import pytest

from sd2.analysis.diagnosis import compute_failure_diagnosis
from sd2.analysis.deviation import DeviationRecord, DeviationTable, classify_status
from sd2.analysis.fingerprint import (
    aggregate_robustness_fingerprints,
    compute_robustness_fingerprint,
)
from sd2.analysis.propagation import _first_onset, _StagePoint, compute_propagation_analysis
from sd2.core.config import SD2Config
from sd2.core.run import RunLog, pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def _points(scores: list[float]) -> list[_StagePoint]:
    return [_StagePoint(frame_idx=i, timestamp=i * 0.1, score=s) for i, s in enumerate(scores)]


def test_onset_persistence_rejects_single_frame_spike() -> None:
    # A one-frame spike above threshold must not count as an onset when
    # persistence requires 3 consecutive frames.
    scores = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert _first_onset(_points(scores), threshold=0.5, persistence=3) is None
    # With persistence=1 (legacy behavior) the spike is an onset at frame 2.
    onset = _first_onset(_points(scores), threshold=0.5, persistence=1)
    assert onset is not None and onset.frame_idx == 2


def test_onset_persistence_accepts_sustained_crossing() -> None:
    # Sustained crossing from frame 3 onward is a real onset at frame 3.
    scores = [0.0, 0.1, 0.2, 0.6, 0.7, 0.8]
    onset = _first_onset(_points(scores), threshold=0.5, persistence=3)
    assert onset is not None and onset.frame_idx == 3


def test_reasoning_first_collapse_propagates_to_planning_and_control() -> None:
    config = _config()
    table = _table(
        {
            Stage.VISION: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            Stage.SEMANTIC: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            Stage.REASONING: [0.10, 0.20, 0.75, 0.80, 0.85, 0.90],
            Stage.PLANNING: [0.10, 0.10, 0.15, 0.45, 0.60, 0.75],
            Stage.CONTROL: [0.05, 0.06, 0.07, 0.20, 0.50, 0.75],
        }
    )
    propagation = compute_propagation_analysis(table, config)
    diagnosis = compute_failure_diagnosis(
        table,
        propagation,
        _paired_run(frame_count=6, collision_at=5, stress_final_progress=0.75),
        config,
    )

    assert diagnosis.primary_failure_stage == Stage.REASONING
    assert diagnosis.diagnosis_type == "temporal_correlational"
    assert diagnosis.driving_failure is True
    assert diagnosis.deviation_precedes_driving_failure is True
    assert any(
        "Reasoning showed the earliest critical deviation" in item
        for item in diagnosis.evidence
    )
    assert any("preceded the first driving failure" in item for item in diagnosis.evidence)
    assert "caused" not in " ".join(diagnosis.evidence).lower()


def test_all_healthy_run_has_no_failure_detected() -> None:
    config = _config()
    table = _table(
        {
            Stage.VISION: [0.05, 0.05, 0.04, 0.05],
            Stage.SEMANTIC: [0.03, 0.03, 0.03, 0.03],
            Stage.REASONING: [0.05, 0.06, 0.07, 0.08],
            Stage.PLANNING: [0.04, 0.05, 0.05, 0.05],
            Stage.CONTROL: [0.02, 0.02, 0.03, 0.03],
        }
    )
    propagation = compute_propagation_analysis(table, config)
    diagnosis = compute_failure_diagnosis(
        table,
        propagation,
        _paired_run(frame_count=4),
        config,
    )

    assert diagnosis.primary_failure_stage is None
    assert diagnosis.status == "no_failure_detected"
    assert diagnosis.driving_failure is False


def test_vision_first_collapse_is_primary() -> None:
    config = _config()
    table = _table(
        {
            Stage.VISION: [0.10, 0.75, 0.80, 0.85],
            Stage.SEMANTIC: [0.05, 0.10, 0.45, 0.60],
            Stage.REASONING: [0.05, 0.10, 0.55, 0.75],
            Stage.PLANNING: [0.05, 0.08, 0.50, 0.70],
            Stage.CONTROL: [0.04, 0.05, 0.30, 0.60],
        }
    )
    propagation = compute_propagation_analysis(table, config)
    diagnosis = compute_failure_diagnosis(
        table,
        propagation,
        _paired_run(frame_count=4, lane_invasion_at=3),
        config,
    )

    assert diagnosis.primary_failure_stage == Stage.VISION


def test_collapse_onset_and_propagation_score_numeric_correctness() -> None:
    config = _config(
        stages=[Stage.VISION.value, Stage.SEMANTIC.value],
        diagnosis={"epsilon": 0.1, "noise_floor": 0.05},
    )
    table = _table(
        {
            Stage.VISION: [0.10, 0.20, 0.00],
            Stage.SEMANTIC: [0.20, 0.60, 0.30],
        }
    )

    propagation = compute_propagation_analysis(table, config)
    score = propagation.propagation_scores[0]
    semantic_onset = propagation.collapse_by_stage()[Stage.SEMANTIC]

    assert [item.propagation_score for item in score.frame_scores] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert score.aggregate_score == pytest.approx(1.5)
    assert score.ratio_clipped == pytest.approx(1.5)
    assert score.log_ratio == pytest.approx((log(0.3 / 0.2) + log(0.7 / 0.3)) / 2)
    assert semantic_onset.warning is not None
    assert semantic_onset.warning.frame_idx == 1
    assert semantic_onset.warning.timestamp == pytest.approx(0.1)
    assert semantic_onset.critical is None


def test_tiny_denominator_propagation_bundle_is_bounded_and_temporal() -> None:
    config = _config(
        stages=[Stage.VISION.value, Stage.SEMANTIC.value],
        thresholds={"warning": 0.005, "critical": 0.90},
        diagnosis={
            "epsilon": 1.0e-6,
            "noise_floor": 0.005,
            "propagation_ratio_cap": 10.0,
            "downstream_window": 2,
        },
    )
    table = _table(
        {
            Stage.VISION: [0.01, 0.01, 0.01],
            Stage.SEMANTIC: [0.95, 0.95, 0.95],
        },
        thresholds={"warning": 0.005, "critical": 0.90},
    )

    propagation = compute_propagation_analysis(table, config)
    score = propagation.propagation_scores[0]
    diagnosis = compute_failure_diagnosis(
        table,
        propagation,
        _paired_run(frame_count=3),
        config,
    )

    assert score.aggregate_score is not None
    assert score.aggregate_score > 90.0
    assert score.ratio_clipped is not None
    assert score.ratio_clipped <= 10.0
    assert score.log_ratio is not None
    assert isfinite(score.log_ratio)
    assert score.absolute_increase == pytest.approx(0.94)
    assert isfinite(score.absolute_increase)
    assert score.downstream_persistence == pytest.approx(1.0)
    assert diagnosis.primary_failure_stage != Stage.VISION
    assert all("ratio" not in item.lower() for item in diagnosis.evidence)


def test_missing_stages_do_not_block_diagnosis_or_fingerprint() -> None:
    config = _config()
    table = _table(
        {
            Stage.REASONING: [0.10, 0.72, 0.80, 0.85],
            Stage.PLANNING: [0.10, 0.15, 0.55, 0.75],
        }
    )

    propagation = compute_propagation_analysis(table, config)
    diagnosis = compute_failure_diagnosis(
        table,
        propagation,
        _paired_run(frame_count=4, collision_at=3),
        config,
    )
    fingerprint = compute_robustness_fingerprint(table, config)

    assert diagnosis.primary_failure_stage == Stage.REASONING
    assert propagation.collapse_by_stage()[Stage.VISION].critical is None
    assert fingerprint.stage_scores[Stage.VISION] is None
    assert fingerprint.stage_scores[Stage.SEMANTIC] is None
    assert fingerprint.stage_scores[Stage.REASONING] == pytest.approx(
        1.0 - ((0.10 + 0.72 + 0.80 + 0.85) / 4)
    )


def test_fingerprint_aggregation_accepts_tables_and_json_files(tmp_path) -> None:
    config = _config(stages=[Stage.REASONING.value, Stage.PLANNING.value])
    table_a = _table(
        {
            Stage.REASONING: [0.20, 0.40],
            Stage.PLANNING: [0.10, 0.20],
        }
    )
    table_b = _table(
        {
            Stage.REASONING: [0.60, 0.80],
            Stage.PLANNING: [0.20, 0.30],
        }
    )
    table_path = tmp_path / "deviation_table.json"
    table_b.write_json(table_path)

    aggregate = aggregate_robustness_fingerprints([table_a, table_path], config)

    assert aggregate.run_count == 2
    assert aggregate.stage_scores[Stage.REASONING] == pytest.approx(0.50)
    assert aggregate.stage_scores[Stage.PLANNING] == pytest.approx(0.80)


def _config(
    stages: list[str] | None = None,
    diagnosis: dict[str, float | int | str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> SD2Config:
    diagnosis_config: dict[str, float | int | str] = {
        "primary_failure_policy": "first_critical_with_downstream_increase",
        "propagation_lag": 0,
        "downstream_window": 2,
        "downstream_min_delta": 0.0,
        "epsilon": 1.0e-6,
        "noise_floor": 0.05,
        "route_progress_drop_threshold": 0.05,
    }
    if diagnosis is not None:
        diagnosis_config.update(diagnosis)
    return SD2Config.model_validate(
        {
            "stages": stages
            or [
                Stage.VISION.value,
                Stage.SEMANTIC.value,
                Stage.REASONING.value,
                Stage.PLANNING.value,
                Stage.CONTROL.value,
                Stage.OUTCOME.value,
            ],
            "thresholds": thresholds or {"warning": 0.4, "critical": 0.7},
            "diagnosis": diagnosis_config,
        }
    )


def _table(
    series_by_stage: dict[Stage, list[float]],
    thresholds: dict[str, float] | None = None,
) -> DeviationTable:
    status_thresholds = thresholds or {"warning": 0.4, "critical": 0.7}
    records: list[DeviationRecord] = []
    frame_count = max(len(series) for series in series_by_stage.values())
    for frame_idx in range(frame_count):
        for stage, series in series_by_stage.items():
            if frame_idx >= len(series):
                continue
            score = float(series[frame_idx])
            records.append(
                DeviationRecord(
                    pair_key=f"synthetic:{frame_idx}",
                    frame_idx=frame_idx,
                    timestamp=frame_idx * 0.1,
                    stage=stage,
                    metric="synthetic",
                    raw_score=score,
                    normalized_score=score,
                    status=classify_status(
                        score,
                        status_thresholds,
                    ),
                    missing=False,
                    details={},
                )
            )
    return DeviationTable(records=records)


def _paired_run(
    frame_count: int,
    collision_at: int | None = None,
    lane_invasion_at: int | None = None,
    clean_final_progress: float = 1.0,
    stress_final_progress: float = 1.0,
) -> object:
    clean_metadata = RunMetadata(
        run_id="clean",
        model_id="model",
        scenario_id="scenario",
        condition="clean",
        severity=0,
        seed=1,
    )
    stress_metadata = RunMetadata(
        run_id="stress",
        model_id="model",
        scenario_id="scenario",
        condition="stress",
        severity=1,
        seed=1,
    )
    clean_frames = [
        _frame(
            run_id="clean",
            frame_idx=frame_idx,
            collision=False,
            lane_invasion=False,
            route_progress=_progress(frame_idx, frame_count, clean_final_progress),
        )
        for frame_idx in range(frame_count)
    ]
    stress_frames = [
        _frame(
            run_id="stress",
            frame_idx=frame_idx,
            collision=collision_at is not None and frame_idx >= collision_at,
            lane_invasion=lane_invasion_at is not None
            and frame_idx >= lane_invasion_at,
            route_progress=_progress(frame_idx, frame_count, stress_final_progress),
        )
        for frame_idx in range(frame_count)
    ]
    return pair_runs(
        RunLog(metadata=clean_metadata, frames=clean_frames),
        RunLog(metadata=stress_metadata, frames=stress_frames),
    )


def _frame(
    run_id: str,
    frame_idx: int,
    collision: bool,
    lane_invasion: bool,
    route_progress: float,
) -> FrameLog:
    return FrameLog.model_validate(
        {
            "run_id": run_id,
            "frame_idx": frame_idx,
            "timestamp": frame_idx * 0.1,
            "states": {
                "outcome": {
                    "collision": collision,
                    "lane_invasion": lane_invasion,
                    "route_progress": route_progress,
                    "driving_score": 0.5 if collision or lane_invasion else 1.0,
                }
            },
        }
    )


def _progress(frame_idx: int, frame_count: int, final_progress: float) -> float:
    if frame_count <= 1:
        return final_progress
    return final_progress * frame_idx / (frame_count - 1)
