from pydantic import ValidationError
import pytest

from sd2.core.schema import FrameLog, RunMetadata, SemanticState
from sd2.core.stage import Stage


def test_frame_log_validates_stage_specific_states() -> None:
    frame = FrameLog.model_validate(
        {
            "run_id": "run_a",
            "frame_idx": 7,
            "timestamp": 0.7,
            "states": {
                "semantic": {
                    "objects": ["vehicle", "lane"],
                    "critical_object": "vehicle",
                },
                "control": {"steer": 0.1, "throttle": 0.2, "brake": 0.0},
            },
        }
    )

    assert set(frame.states) == {Stage.SEMANTIC, Stage.CONTROL}
    assert isinstance(frame.states[Stage.SEMANTIC], SemanticState)
    assert frame.states[Stage.SEMANTIC].critical_object == "vehicle"


def test_frame_log_allows_missing_stages() -> None:
    frame = FrameLog(run_id="run_a", frame_idx=1, timestamp=0.1, states={})

    assert frame.states == {}


def test_run_metadata_requires_core_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RunMetadata.model_validate(
            {
                "model_id": "openemma",
                "scenario_id": "town05_route01",
                "condition": "clean",
                "severity": 0,
                "seed": 42,
            }
        )

    assert "run_id" in str(exc_info.value)


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        FrameLog.model_validate(
            {
                "run_id": "run_a",
                "frame_idx": 1,
                "timestamp": 0.1,
                "states": {"unknown": {}},
            }
        )

    assert "unknown stage" in str(exc_info.value)


def test_stage_ordering_helpers() -> None:
    assert Stage.ordered() == [
        Stage.VISION,
        Stage.SEMANTIC,
        Stage.REASONING,
        Stage.PLANNING,
        Stage.CONTROL,
        Stage.OUTCOME,
    ]
    assert Stage.REASONING.upstream() == [Stage.VISION, Stage.SEMANTIC]
    assert Stage.REASONING.downstream() == [Stage.PLANNING, Stage.CONTROL, Stage.OUTCOME]
