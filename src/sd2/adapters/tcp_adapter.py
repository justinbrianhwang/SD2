"""Pure TCP-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``, ``torch``, ``torchvision``,
or any TCP package. TCP exposes image-backbone features, predicted trajectory
waypoints, and both trajectory-PID and control-branch actions. It has no
explicit semantic head, so the semantic stage is omitted rather than emitted as
an empty state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sd2.adapters import transfuser_adapter as _tf
from sd2.adapters.carla_adapter import write_sd2_jsonl
from sd2.core.schema import FrameLog, RunMetadata


TCP_MODEL_ID = "tcp"


def tcp_record_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert an extracted TCP frame record into an SD2 frame record."""

    vision_record = _tf._as_mapping(record.get("vision"))
    planning_record = _tf._as_mapping(record.get("planning"))
    control_record = _tf._as_mapping(record.get("control"))
    outcome_record = _tf._as_mapping(record.get("outcome"))

    vision = _vision_state(vision_record)
    planning = _planning_state(planning_record, record.get("ego"))
    control = _control_state(control_record)
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


def build_tcp_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None = None,
    severity: int = 0,
    seed: int = 42,
    *,
    model_id: str = TCP_MODEL_ID,
    town: str | None = None,
    timestamp_start: str | None = None,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for a TCP CARLA run."""

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


def _vision_state(record: Any) -> dict[str, Any]:
    vision = _tf._vision_state(record)
    if record.get("input_simplification") is not None:
        vision["input_simplification"] = str(record.get("input_simplification"))
    return vision


def _planning_state(record: Any, ego_record: Any) -> dict[str, Any]:
    planning = _tf._planning_state(record, ego_record)
    desired_speed = _tf._optional_float(record.get("desired_speed"))
    if desired_speed is not None:
        planning["target_speed"] = desired_speed
    if record.get("planning_source") is not None:
        planning["planning_source"] = str(record.get("planning_source"))
    return planning


def _control_state(record: Any) -> dict[str, Any]:
    control = _tf._control_state(record)

    if record.get("planner_type") is not None:
        control["planner_type"] = str(record.get("planner_type"))

    details = _tf._as_mapping(record.get("details"))
    normalized_details: dict[str, Any] = _tf._jsonable(dict(details)) if details else {}
    for key in (
        "traj_branch",
        "ctrl_branch",
        "selected_branch",
        "pre_clamp",
        "post_clamp",
        "pid_metadata",
        "ctrl_metadata",
        "traj_metadata",
    ):
        if record.get(key) is not None:
            normalized_details[key] = _tf._jsonable(record.get(key))

    if normalized_details:
        control["details"] = normalized_details
    return control


__all__ = [
    "TCP_MODEL_ID",
    "build_tcp_run_metadata",
    "tcp_record_to_sd2",
    "write_sd2_jsonl",
]
