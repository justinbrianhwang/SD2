import pytest

from sd2.analysis.deviation import DeviationTable
from sd2.analysis.diagnosis import compute_failure_diagnosis
from sd2.analysis.intervention import _count_events, _outcome_summary
from sd2.analysis.propagation import compute_propagation_analysis
from sd2.core.config import SD2Config
from sd2.core.run import RunLog, pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([False, False, True, True, True, False, False], 1),
        ([False, False, False], 0),
        ([True, True, True], 1),
        ([True, True, False, False, False, False, True, True], 2),
        ([True, True, False, True, True], 1),
    ],
)
def test_count_events_merges_only_short_gaps(
    flags: list[bool],
    expected: int,
) -> None:
    assert _count_events(flags) == expected


def test_outcome_summary_counts_neat_pinning_as_one_collision_event() -> None:
    collision_flags = [False] * 300
    for frame_idx in [*range(151, 189), *range(190, 221)]:
        collision_flags[frame_idx] = True

    summary = _outcome_summary(_run_from_flags(collision_flags=collision_flags))

    assert summary["collision_count"] == 1
    assert summary["collision_frames"] == 69
    assert summary["collision_any"] is True
    assert summary["collision_count"] != summary["collision_frames"]


def test_outcome_summary_counts_brief_separate_contacts_as_separate_events() -> None:
    collision_flags = [False] * 40
    for frame_idx in (5, 20, 35):
        collision_flags[frame_idx] = True

    lane_invasion_flags = [False] * 40
    for frame_idx in [*range(8, 11), *range(12, 15)]:
        lane_invasion_flags[frame_idx] = True

    summary = _outcome_summary(
        _run_from_flags(
            collision_flags=collision_flags,
            lane_invasion_flags=lane_invasion_flags,
        )
    )

    assert summary["collision_count"] == 3
    assert summary["collision_frames"] == 3
    assert summary["lane_invasion_count"] == 1
    assert summary["lane_invasion_frames"] == 6
    assert summary["lane_invasion_any"] is True


def test_diagnosis_reports_one_collision_failure_event_for_pinning() -> None:
    collision_flags = [False] * 16
    for frame_idx in range(5, 13):
        collision_flags[frame_idx] = True
    paired_run = pair_runs(
        _run_from_flags(
            collision_flags=[False] * len(collision_flags),
            condition="clean",
        ),
        _run_from_flags(collision_flags=collision_flags, condition="stress"),
    )
    config = _config()
    table = DeviationTable(records=[])
    propagation = compute_propagation_analysis(table, config)

    diagnosis = compute_failure_diagnosis(table, propagation, paired_run, config)

    collision_evidence = [
        item
        for item in diagnosis.driving_failure_evidence
        if item.startswith("Collision occurred")
    ]
    assert len(collision_evidence) == 1
    assert diagnosis.driving_failure_time is not None
    assert diagnosis.driving_failure_time["type"] == "collision"
    assert diagnosis.driving_failure_time["frame_idx"] == 5
    assert diagnosis.driving_failure_time["timestamp"] == pytest.approx(0.5)


def _run_from_flags(
    *,
    collision_flags: list[bool],
    lane_invasion_flags: list[bool] | None = None,
    condition: str = "stress",
) -> RunLog:
    lane_flags = lane_invasion_flags or [False] * len(collision_flags)
    metadata = RunMetadata(
        run_id=condition,
        model_id="model",
        scenario_id="scenario",
        condition=condition,
        severity=0 if condition == "clean" else 1,
        seed=1,
    )
    return RunLog(
        metadata=metadata,
        frames=[
            _frame(
                run_id=metadata.run_id,
                frame_idx=frame_idx,
                collision=collision,
                lane_invasion=lane_invasion,
                frame_count=len(collision_flags),
            )
            for frame_idx, (collision, lane_invasion) in enumerate(
                zip(collision_flags, lane_flags)
            )
        ],
    )


def _frame(
    *,
    run_id: str,
    frame_idx: int,
    collision: bool,
    lane_invasion: bool,
    frame_count: int,
) -> FrameLog:
    route_progress = 1.0 if frame_count <= 1 else frame_idx / (frame_count - 1)
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


def _config() -> SD2Config:
    return SD2Config.model_validate(
        {
            "stages": [
                Stage.VISION.value,
                Stage.SEMANTIC.value,
                Stage.REASONING.value,
                Stage.PLANNING.value,
                Stage.CONTROL.value,
                Stage.OUTCOME.value,
            ],
            "thresholds": {"warning": 0.4, "critical": 0.7},
            "diagnosis": {
                "primary_failure_policy": "first_critical_with_downstream_increase",
                "propagation_lag": 0,
                "downstream_window": 2,
                "downstream_min_delta": 0.0,
                "epsilon": 1.0e-6,
                "noise_floor": 0.05,
                "route_progress_drop_threshold": 0.05,
            },
        }
    )
