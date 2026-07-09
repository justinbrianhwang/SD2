"""Pure InterFuser-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``, ``torch``, ``timm``, or
any InterFuser package. The live CARLA recorder extracts plain Python and
NumPy-like values, then passes dictionaries here.

Vision representation: SD2's default vision metric is ``embedding_cosine`` and
reads either ``embedding`` or ``feature``. InterFuser's live recorder therefore
stores a compact ``feature`` vector, normally a mean-pooled model feature. When
no model feature is available, this adapter can fall back to
``[image_mean, image_std]`` so test and smoke logs remain measurable while still
preserving the raw image statistics as extra fields.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from sd2.adapters.carla_adapter import write_sd2_jsonl
from sd2.core.schema import FrameLog, RunMetadata


INTERFUSER_MODEL_ID = "interfuser"


def interfuser_record_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert an extracted InterFuser frame record into an SD2 frame record."""

    vision_record = _as_mapping(record.get("vision"))
    semantic_record = _as_mapping(record.get("semantic"))
    planning_record = _as_mapping(record.get("planning"))
    control_record = _as_mapping(record.get("control"))
    outcome_record = _as_mapping(record.get("outcome"))

    vision = _vision_state(vision_record)
    semantic = _semantic_state(semantic_record)
    planning = _planning_state(planning_record, record.get("ego"))
    control = _control_state(control_record)
    outcome = _outcome_state(outcome_record)

    payload = {
        "run_id": str(run_id),
        "frame_idx": int(record.get("frame_idx", 0)),
        "timestamp": float(record.get("timestamp", 0.0)),
        "states": {
            "vision": vision,
            "semantic": semantic,
            "planning": planning,
            "control": control,
            "outcome": outcome,
        },
    }
    frame = FrameLog.model_validate(payload)
    return {"type": "frame", **frame.model_dump(mode="json", exclude_none=True)}


def build_interfuser_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None = None,
    severity: int = 0,
    seed: int = 42,
    *,
    model_id: str = INTERFUSER_MODEL_ID,
    town: str | None = None,
    timestamp_start: str | None = None,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for an InterFuser CARLA run."""

    normalized_stress_type = None if stress_type in (None, "", "none") else str(stress_type)
    metadata = RunMetadata.model_validate(
        {
            "run_id": str(run_id),
            "model_id": str(model_id),
            "scenario_id": _scenario_with_town(scenario_id, town),
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


def _vision_state(record: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    image_mean = _optional_float(record.get("image_mean"))
    image_std = _optional_float(record.get("image_std"))
    if image_mean is not None:
        state["image_mean"] = image_mean
    if image_std is not None:
        state["image_std"] = image_std

    feature = _coerce_float_vector(record.get("feature"))
    if feature is not None:
        state["feature"] = feature
        state["feature_source"] = str(record.get("feature_source") or "model_feature")
    elif image_mean is not None and image_std is not None:
        state["feature"] = [image_mean, image_std]
        state["feature_source"] = "image_stats"

    for key in ("image_path", "brightness", "noise_level"):
        if record.get(key) is not None:
            state[key] = _jsonable(record[key])
    return state


def _semantic_state(record: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    summary = _as_mapping(record.get("object_density_summary"))
    if summary:
        state["object_density_summary"] = _jsonable(dict(summary))

    num_objects = _optional_int(record.get("num_objects"))
    if num_objects is not None:
        state["num_objects"] = num_objects

    junction = _optional_float(record.get("junction"))
    if junction is not None:
        state["junction"] = junction

    traffic_light_score = _optional_float(record.get("traffic_light_state"))
    if traffic_light_score is not None:
        state["traffic_light_state"] = _traffic_light_label(traffic_light_score)
        state["traffic_light_state_score"] = traffic_light_score
    elif record.get("traffic_light_state") is not None:
        state["traffic_light_state"] = str(record.get("traffic_light_state"))

    stop_sign = _optional_float(record.get("stop_sign"))
    if stop_sign is not None:
        state["stop_sign"] = stop_sign

    objects = _coerce_string_list(record.get("objects"))
    if objects is None:
        objects = _objects_from_summary(summary)
    if objects is not None:
        state["objects"] = objects
        if objects:
            state["critical_object"] = _critical_object(objects, summary)

    if record.get("semantic_description") is not None:
        state["semantic_description"] = str(record.get("semantic_description"))
    return state


def _planning_state(record: Mapping[str, Any], ego_record: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    waypoints = _coerce_waypoints(record.get("waypoints"))
    if waypoints is not None:
        state["waypoints"] = waypoints

    target_speed = _optional_float(record.get("target_speed"))
    if target_speed is not None:
        state["target_speed"] = target_speed

    target_point = _coerce_float_vector(record.get("target_point"))
    if target_point is not None:
        state["target_point"] = target_point

    if record.get("command") is not None:
        state["command"] = _jsonable(record.get("command"))

    ego = _coerce_ego(ego_record)
    if ego:
        state["ego"] = ego
    return state


def _control_state(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "steer": _optional_float(record.get("steer"), default=0.0),
        "throttle": _optional_float(record.get("throttle"), default=0.0),
        "brake": _optional_float(record.get("brake"), default=0.0),
    }


def _outcome_state(record: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "collision": bool(record.get("collision", False)),
        "lane_invasion": bool(record.get("lane_invasion", False)),
    }

    route_progress = _optional_float(record.get("route_progress"))
    if route_progress is not None:
        state["route_progress"] = _clamp(route_progress, 0.0, 1.0)

    min_ttc = _optional_float(record.get("min_ttc"))
    if min_ttc is not None:
        state["min_ttc"] = min_ttc
    return state


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_waypoints(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    waypoints: list[list[float]] = []
    for point in _tolist(value):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        waypoints.append([float(point[0]), float(point[1])])
    return waypoints


def _coerce_ego(value: Any) -> dict[str, float]:
    ego_record = _as_mapping(value)
    ego: dict[str, float] = {}
    for key in ("x", "y", "z", "yaw", "speed"):
        item = _optional_float(ego_record.get(key))
        if item is not None:
            ego[key] = item
    return ego


def _coerce_float_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    values = _flatten(_tolist(value))
    if not values:
        return []
    return [float(item) for item in values]


def _coerce_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    items = _tolist(value)
    if isinstance(items, (str, bytes)):
        items = [items]
    result = [str(item).strip() for item in items if str(item).strip()]
    return result


def _objects_from_summary(summary: Mapping[str, Any]) -> list[str] | None:
    if not summary:
        return None
    objects: list[str] = []
    key_map = {
        "car": "vehicle",
        "cars": "vehicle",
        "vehicle": "vehicle",
        "vehicles": "vehicle",
        "bike": "bike",
        "bikes": "bike",
        "pedestrian": "pedestrian",
        "pedestrians": "pedestrian",
    }
    for raw_key, raw_value in summary.items():
        key = str(raw_key).strip().lower()
        object_name = key_map.get(key)
        if object_name is None:
            continue
        value = _optional_float(raw_value, default=0.0)
        if value and value > 0.0 and object_name not in objects:
            objects.append(object_name)
    return sorted(objects)


def _critical_object(objects: list[str], summary: Mapping[str, Any]) -> str | None:
    if not objects:
        return None
    if not summary:
        return objects[0]
    scores: dict[str, float] = {}
    for raw_key, raw_value in summary.items():
        value = _optional_float(raw_value, default=0.0) or 0.0
        key = str(raw_key).strip().lower()
        if key in ("car", "cars", "vehicle", "vehicles"):
            scores["vehicle"] = max(scores.get("vehicle", 0.0), value)
        elif key in ("bike", "bikes"):
            scores["bike"] = max(scores.get("bike", 0.0), value)
        elif key in ("pedestrian", "pedestrians"):
            scores["pedestrian"] = max(scores.get("pedestrian", 0.0), value)
    return max(objects, key=lambda item: scores.get(item, 0.0))


def _traffic_light_label(score: float) -> str:
    return "red_or_yellow" if score >= 0.5 else "not_red_or_yellow"


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scenario_with_town(scenario_id: str, town: str | None) -> str:
    scenario = str(scenario_id)
    if not town:
        return scenario
    town_text = str(town)
    if town_text.lower() in scenario.lower():
        return scenario
    return f"{town_text}_{scenario}"


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


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [value]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


__all__ = [
    "INTERFUSER_MODEL_ID",
    "build_interfuser_run_metadata",
    "interfuser_record_to_sd2",
    "write_sd2_jsonl",
]
