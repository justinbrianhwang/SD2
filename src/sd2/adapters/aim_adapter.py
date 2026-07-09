"""Pure AIM-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``, ``torch``, or any AIM
package. AIM is camera-only and has no semantic head, so the semantic stage is
omitted rather than emitted as an empty state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sd2.adapters import transfuser_adapter as _tf
from sd2.adapters.carla_adapter import write_sd2_jsonl
from sd2.core.schema import FrameLog, RunMetadata


AIM_MODEL_ID = "aim"


def aim_record_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert an extracted AIM frame record into an SD2 frame record."""

    vision_record = _tf._as_mapping(record.get("vision"))
    planning_record = _tf._as_mapping(record.get("planning"))
    control_record = _tf._as_mapping(record.get("control"))
    outcome_record = _tf._as_mapping(record.get("outcome"))

    vision = _tf._vision_state(vision_record)
    planning = _planning_state(planning_record, record.get("ego"))
    control = _tf._control_state(control_record)
    outcome = _tf._outcome_state(outcome_record)

    payload = {
        "run_id": str(run_id),
        "frame_idx": int(record.get("frame_idx", 0)),
        "timestamp": float(record.get("timestamp", 0.0)),
        "states": {
            "vision": vision,
            "planning": planning,
            "control": control,
            "outcome": outcome,
        },
    }
    frame = FrameLog.model_validate(payload)
    return {"type": "frame", **frame.model_dump(mode="json", exclude_none=True)}


def build_aim_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None = None,
    severity: int = 0,
    seed: int = 42,
    *,
    model_id: str = AIM_MODEL_ID,
    town: str | None = None,
    timestamp_start: str | None = None,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for an AIM CARLA run."""

    normalized_stress_type = None if stress_type in (None, "", "none") else str(stress_type)
    metadata = RunMetadata.model_validate(
        {
            "run_id": str(run_id),
            "model_id": str(model_id),
            "scenario_id": _tf._scenario_with_town(scenario_id, town),
            "condition": str(condition),
            "stress_type": normalized_stress_type,
            "severity": int(severity),
            "seed": int(seed),
            "timestamp_start": timestamp_start
            or datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    return {
        "type": "run_metadata",
        **metadata.model_dump(mode="json", exclude_none=True),
    }


def _planning_state(record: Any, ego_record: Any) -> dict[str, Any]:
    planning = _tf._planning_state(record, ego_record)
    target_speed = _tf._optional_float(record.get("desired_speed"))
    if target_speed is not None:
        planning["target_speed"] = target_speed
    if record.get("planning_source") is not None:
        planning["planning_source"] = str(record.get("planning_source"))
    return planning


__all__ = [
    "AIM_MODEL_ID",
    "aim_record_to_sd2",
    "build_aim_run_metadata",
    "write_sd2_jsonl",
]
