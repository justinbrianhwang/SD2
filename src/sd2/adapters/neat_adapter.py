"""Pure NEAT-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``, ``torch``, or any NEAT
package. NEAT exposes a BEV semantic occupancy summary, which is recorded in the
semantic stage using the same ``bev_seg_summary`` shape as TransFuser.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sd2.adapters import transfuser_adapter as _tf
from sd2.adapters.carla_adapter import Sd2JsonlWriter, write_sd2_jsonl
from sd2.adapters.intervention_adapter import intervention_state
from sd2.core.schema import FrameLog, RunMetadata


NEAT_MODEL_ID = "neat"


def neat_record_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert an extracted NEAT frame record into an SD2 frame record."""

    vision_record = _tf._as_mapping(record.get("vision"))
    semantic_record = _tf._as_mapping(record.get("semantic"))
    planning_record = dict(_tf._as_mapping(record.get("planning")))
    control_record = _tf._as_mapping(record.get("control"))
    outcome_record = _tf._as_mapping(record.get("outcome"))

    if "waypoints" not in planning_record and "pred_waypoint_mean" in planning_record:
        planning_record["waypoints"] = planning_record.get("pred_waypoint_mean")

    vision = _tf._vision_state(vision_record)
    semantic = _semantic_state(semantic_record)
    planning = _planning_state(planning_record, record.get("ego"))
    control = _tf._control_state(control_record)
    outcome = _tf._outcome_state(outcome_record)
    states = {
        "vision": vision,
        "semantic": semantic,
        "planning": planning,
        "control": control,
        "outcome": outcome,
    }
    intervention = intervention_state(record.get("intervention"))
    if intervention is not None:
        states["intervention"] = intervention

    payload = {
        "run_id": str(run_id),
        "frame_idx": int(record.get("frame_idx", 0)),
        "timestamp": float(record.get("timestamp", 0.0)),
        "states": states,
    }
    frame = FrameLog.model_validate(payload)
    return {"type": "frame", **frame.model_dump(mode="json", exclude_none=True)}


def build_neat_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None = None,
    severity: int = 0,
    seed: int = 42,
    *,
    model_id: str = NEAT_MODEL_ID,
    town: str | None = None,
    timestamp_start: str | None = None,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for a NEAT CARLA run."""

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


def _semantic_state(record: Any) -> dict[str, Any]:
    semantic = _tf._semantic_state(record)
    red_light_occ = _tf._optional_float(record.get("red_light_occ"))
    if red_light_occ is not None:
        semantic["red_light_occ"] = red_light_occ
    if record.get("semantic_source") is not None:
        semantic["semantic_source"] = str(record.get("semantic_source"))
    return semantic


def _planning_state(record: Any, ego_record: Any) -> dict[str, Any]:
    planning = _tf._planning_state(record, ego_record)
    desired_speed = _tf._optional_float(record.get("desired_speed"))
    if desired_speed is not None:
        planning["target_speed"] = desired_speed
    if record.get("planning_source") is not None:
        planning["planning_source"] = str(record.get("planning_source"))
    return planning


__all__ = [
    "NEAT_MODEL_ID",
    "Sd2JsonlWriter",
    "build_neat_run_metadata",
    "neat_record_to_sd2",
    "write_sd2_jsonl",
]
