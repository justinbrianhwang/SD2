"""Pure helpers for SD2 counterfactual intervention payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def intervention_state(record: Any) -> dict[str, Any] | None:
    """Normalize a recorder intervention block for ``FrameLog`` states."""

    if not isinstance(record, Mapping):
        return None

    state: dict[str, Any] = {}
    for key in ("stage", "direction", "applied_source"):
        if record.get(key) is not None:
            state[key] = str(record[key])

    stress_control = _control_state(record.get("control_from_stress_forward"))
    clean_control = _control_state(record.get("control_from_clean_forward"))
    if stress_control is not None:
        state["control_from_stress_forward"] = stress_control
    if clean_control is not None:
        state["control_from_clean_forward"] = clean_control
    for key in ("control_hybrid_planning_clean", "control_hybrid_semantic_clean"):
        hybrid_control = _control_state(record.get(key))
        if hybrid_control is not None:
            state[key] = hybrid_control

    clean_waypoints = _coerce_waypoints(record.get("planning_waypoints_clean_forward"))
    if clean_waypoints is not None:
        state["planning_waypoints_clean_forward"] = clean_waypoints

    semantic_clean = _as_mapping(record.get("semantic_clean_forward"))
    if semantic_clean:
        state["semantic_clean_forward"] = _jsonable(dict(semantic_clean))

    for key in ("config", "notes", "model_id"):
        if record.get(key) is not None:
            state[key] = _jsonable(record[key])

    return state or None


def _control_state(value: Any) -> dict[str, float] | None:
    record = _as_mapping(value)
    if not record:
        return None
    return {
        "steer": _optional_float(record.get("steer"), default=0.0),
        "throttle": _optional_float(record.get("throttle"), default=0.0),
        "brake": _optional_float(record.get("brake"), default=0.0),
    }


def _coerce_waypoints(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    waypoints: list[list[float]] = []
    for point in _tolist(value):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        waypoints.append([float(point[0]), float(point[1])])
    return waypoints


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    value = _tolist(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = ["intervention_state"]
