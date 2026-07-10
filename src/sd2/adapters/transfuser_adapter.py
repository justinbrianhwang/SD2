"""Pure TransFuser-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``, ``torch``, ``timm``, or
any TransFuser package. The live CARLA recorder extracts plain Python and
NumPy-like values, then passes dictionaries here.

Vision representation: SD2's default vision metric is ``embedding_cosine`` and
reads either ``embedding`` or ``feature``. The TransFuser recorder stores a
compact ``feature`` vector from the fused image/LiDAR backbone embedding. When
no model feature is available, this adapter can fall back to
``[image_mean, image_std]`` so test and smoke logs remain measurable while still
preserving the raw image statistics as extra fields.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from sd2.adapters.carla_adapter import Sd2JsonlWriter, write_sd2_jsonl
from sd2.adapters.intervention_adapter import intervention_state
from sd2.core.schema import FrameLog, RunMetadata


TRANSFUSER_MODEL_ID = "transfuser"


def transfuser_record_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert an extracted TransFuser frame record into an SD2 frame record."""

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


def build_transfuser_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None = None,
    severity: int = 0,
    seed: int = 42,
    *,
    model_id: str = TRANSFUSER_MODEL_ID,
    town: str | None = None,
    timestamp_start: str | None = None,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for a TransFuser CARLA run."""

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
        state["feature_source"] = str(
            record.get("feature_source") or "mean_pooled_fused_features"
        )
    elif image_mean is not None and image_std is not None:
        state["feature"] = [image_mean, image_std]
        state["feature_source"] = "image_stats"

    for key in ("image_path", "brightness", "noise_level"):
        if record.get(key) is not None:
            state[key] = _jsonable(record[key])
    return state


def _semantic_state(record: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}

    raw_detections = _first_present(
        record.get("detections"),
        record.get("rotated_bb"),
        record.get("rotated_bboxes"),
        record.get("bounding_boxes"),
    )
    detections = (
        _coerce_detections(raw_detections)
        if raw_detections is not _MISSING
        else None
    )

    generated_summary: dict[str, Any] = {}
    generated_counts: dict[str, int] = {}
    if detections is not None:
        generated_summary, generated_counts = _summarize_detections(detections)

    provided_summary = _as_mapping(record.get("object_density_summary"))
    summary = {**generated_summary, **_jsonable(dict(provided_summary))}
    if summary:
        state["object_density_summary"] = summary

    provided_counts = _as_mapping(record.get("per_class_counts"))
    per_class_counts = {**generated_counts, **_coerce_count_mapping(provided_counts)}
    if per_class_counts:
        state["per_class_counts"] = per_class_counts

    num_objects = _optional_int(record.get("num_objects"))
    if num_objects is None and detections is not None:
        num_objects = len(detections)
    if num_objects is not None:
        state["num_objects"] = num_objects

    objects = _coerce_string_list(record.get("objects"))
    if objects is None:
        objects = _objects_from_counts(per_class_counts)
    if objects is None and summary:
        objects = _objects_from_summary(summary)
    if objects is None and detections is not None:
        objects = []
    if objects is not None:
        state["objects"] = objects
        if objects:
            state["critical_object"] = _critical_object(objects, per_class_counts, summary)

    if detections is not None:
        state["detections"] = detections

    bev_seg_summary = record.get("bev_seg_summary")
    if bev_seg_summary is None:
        bev_seg_summary = record.get("bev_semantic_summary")
    if bev_seg_summary is not None:
        state["bev_seg_summary"] = _jsonable(bev_seg_summary)

    if record.get("semantic_description") is not None:
        state["semantic_description"] = str(record.get("semantic_description"))
    return state


def _planning_state(record: Mapping[str, Any], ego_record: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    raw_waypoints = _first_present(record.get("waypoints"), record.get("pred_wp"))
    waypoints = None if raw_waypoints is _MISSING else _coerce_waypoints(raw_waypoints)
    if waypoints is not None:
        state["waypoints"] = waypoints

    target_speed = _optional_float(record.get("target_speed"))
    if target_speed is None and waypoints:
        target_speed = _target_speed_from_waypoints(waypoints)
    if target_speed is not None:
        state["target_speed"] = target_speed

    target_point = _coerce_float_vector(record.get("target_point"))
    if target_point is not None:
        state["target_point"] = target_point

    if record.get("command") is not None:
        state["command"] = _jsonable(record.get("command"))
    if record.get("is_stuck") is not None:
        state["is_stuck"] = bool(record.get("is_stuck"))

    ego = _coerce_ego(ego_record)
    if ego:
        state["ego"] = ego
    return state


def _control_state(record: Mapping[str, Any]) -> dict[str, Any]:
    control = {
        "steer": _optional_float(record.get("steer"), default=0.0),
        "throttle": _optional_float(record.get("throttle"), default=0.0),
        "brake": _optional_float(record.get("brake"), default=0.0),
    }
    if record.get("anti_crawl_applied") is not None:
        control["anti_crawl_applied"] = bool(record.get("anti_crawl_applied"))
    applied = _optional_float(record.get("applied_throttle"))
    if applied is not None:
        control["applied_throttle"] = applied
    return control


def _outcome_state(record: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "collision": bool(record.get("collision", False)),
        "lane_invasion": bool(record.get("lane_invasion", False)),
    }

    route_progress = _optional_float(record.get("route_progress"))
    if route_progress is not None:
        state["route_progress"] = _clamp(route_progress, 0.0, 1.0)

    if record.get("off_route") is not None:
        state["off_route"] = bool(record.get("off_route"))

    min_ttc = _optional_float(record.get("min_ttc"))
    if min_ttc is not None:
        state["min_ttc"] = min_ttc
    return state


def _coerce_detections(value: Any) -> list[dict[str, Any]]:
    value = _tolist(value)
    if isinstance(value, Mapping):
        nested = _first_present(value.get("detections"), value.get("boxes"))
        if nested is not _MISSING:
            value = _tolist(nested)
        else:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return []

    detections: list[dict[str, Any]] = []
    for item in value:
        detection = _coerce_detection(item)
        if detection:
            detections.append(detection)
    return detections


def _coerce_detection(item: Any) -> dict[str, Any]:
    item = _tolist(item)
    bbox: Any = None
    confidence: float | None = None
    brake: float | None = None
    class_name: str | None = None

    if isinstance(item, Mapping):
        bbox = _first_present(
            item.get("bbox"),
            item.get("box"),
            item.get("corners"),
            item.get("points"),
        )
        confidence = _optional_float(item.get("confidence"))
        brake = _optional_float(item.get("brake"))
        class_name = _normalize_object_class(
            _first_present(item.get("class"), item.get("label"), item.get("type"))
        )
    elif isinstance(item, (list, tuple)):
        if len(item) >= 3 and _looks_like_bbox(item[0]):
            bbox = item[0]
            brake = _optional_float(item[1])
            confidence = _optional_float(item[2])
        elif len(item) == 2 and _looks_like_bbox(item[0]) and isinstance(item[1], Mapping):
            bbox = item[0]
            meta = item[1]
            confidence = _optional_float(meta.get("confidence"))
            brake = _optional_float(meta.get("brake"))
            class_name = _normalize_object_class(
                _first_present(meta.get("class"), meta.get("label"), meta.get("type"))
            )
        else:
            bbox = item
    else:
        bbox = item

    points = _coerce_bbox_points(bbox)
    if not points:
        return {}

    center = _bbox_center(points)
    area = _polygon_area(points[:4]) if len(points) >= 4 else None
    detection: dict[str, Any] = {
        "class": class_name or "vehicle",
        "bbox": points,
        "center": center,
        "distance": math.hypot(center[0], center[1]),
    }
    if area is not None:
        detection["area"] = area
    if confidence is not None:
        detection["confidence"] = confidence
    if brake is not None:
        detection["brake"] = brake
    return detection


def _coerce_bbox_points(value: Any) -> list[list[float]]:
    value = _tolist(value)
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        nested = _first_present(
            value.get("bbox"),
            value.get("box"),
            value.get("corners"),
            value.get("points"),
        )
        return [] if nested is _MISSING else _coerce_bbox_points(nested)
    if not isinstance(value, (list, tuple)):
        return []
    if _is_flat_number_sequence(value):
        return _bbox_points_from_flat(value)

    points: list[list[float]] = []
    for point in value:
        point = _tolist(point)
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _optional_float(point[0])
        y = _optional_float(point[1])
        if x is None or y is None:
            continue
        if len(point) >= 3:
            z = _optional_float(point[2])
            points.append([x, y, z if z is not None else 0.0])
        else:
            points.append([x, y])
    return points


def _bbox_points_from_flat(value: Any) -> list[list[float]]:
    values = [float(item) for item in _tolist(value)]
    if len(values) < 4:
        return []
    x, y, w, h = values[:4]
    half_w = abs(w) / 2.0
    half_h = abs(h) / 2.0
    return [
        [x - half_w, y - half_h],
        [x - half_w, y + half_h],
        [x + half_w, y + half_h],
        [x + half_w, y - half_h],
        [x, y],
    ]


def _summarize_detections(
    detections: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    counts = Counter(str(det.get("class") or "vehicle") for det in detections)
    cell_counts: Counter[tuple[int, int]] = Counter()
    distances: list[float] = []
    areas: list[float] = []
    confidences: list[float] = []

    for det in detections:
        center = det.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            x = _optional_float(center[0])
            y = _optional_float(center[1])
            if x is not None and y is not None:
                cell_counts[(math.floor(x / 4.0), math.floor(y / 4.0))] += 1
        distance = _optional_float(det.get("distance"))
        if distance is not None:
            distances.append(distance)
        area = _optional_float(det.get("area"))
        if area is not None:
            areas.append(area)
        confidence = _optional_float(det.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)

    num_objects = len(detections)
    occupied_cells = len(cell_counts)
    summary: dict[str, Any] = dict(sorted(counts.items()))
    summary.update(
        {
            "occupied_cells": occupied_cells,
            "mean_density": num_objects / max(1, occupied_cells),
            "max_density": max(cell_counts.values(), default=0),
        }
    )
    if distances:
        summary["min_distance"] = min(distances)
        summary["mean_distance"] = sum(distances) / len(distances)
    if areas:
        summary["mean_bbox_area"] = sum(areas) / len(areas)
    if confidences:
        summary["max_confidence"] = max(confidences)
        summary["mean_confidence"] = sum(confidences) / len(confidences)
    return summary, dict(sorted(counts.items()))


def _bbox_center(points: list[list[float]]) -> list[float]:
    usable = points[:4] if len(points) >= 4 else points
    return [
        sum(point[0] for point in usable) / len(usable),
        sum(point[1] for point in usable) / len(usable),
    ]


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for left, right in zip(points, [*points[1:], points[0]]):
        area += left[0] * right[1] - right[0] * left[1]
    return abs(area) / 2.0


def _target_speed_from_waypoints(waypoints: list[list[float]]) -> float | None:
    if len(waypoints) < 2:
        return None
    return 2.0 * math.hypot(
        waypoints[1][0] - waypoints[0][0],
        waypoints[1][1] - waypoints[0][1],
    )


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


def _coerce_count_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        object_name = _normalize_object_class(raw_key)
        count = _optional_int(raw_value)
        if object_name is not None and count is not None:
            counts[object_name] = count
    return dict(sorted(counts.items()))


def _objects_from_counts(counts: Mapping[str, Any]) -> list[str] | None:
    if not counts:
        return None
    return sorted(
        key
        for key, raw_value in counts.items()
        if (_optional_float(raw_value, default=0.0) or 0.0) > 0.0
    )


def _objects_from_summary(summary: Mapping[str, Any]) -> list[str] | None:
    if not summary:
        return None
    objects: list[str] = []
    for raw_key, raw_value in summary.items():
        object_name = _normalize_object_class(raw_key)
        if object_name is None:
            continue
        value = _optional_float(raw_value, default=0.0)
        if value and value > 0.0 and object_name not in objects:
            objects.append(object_name)
    return sorted(objects)


def _critical_object(
    objects: list[str],
    counts: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str | None:
    if not objects:
        return None
    scores: dict[str, float] = {}
    for source in (summary, counts):
        for raw_key, raw_value in source.items():
            object_name = _normalize_object_class(raw_key)
            if object_name is None:
                continue
            value = _optional_float(raw_value, default=0.0) or 0.0
            scores[object_name] = max(scores.get(object_name, 0.0), value)
    return max(objects, key=lambda item: scores.get(item, 0.0))


def _normalize_object_class(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, bool):
        return "vehicle"
    if isinstance(value, (int, float)):
        return "vehicle" if int(value) == 0 else str(int(value))
    text = str(value).strip().lower()
    aliases = {
        "car": "vehicle",
        "cars": "vehicle",
        "vehicle": "vehicle",
        "vehicles": "vehicle",
        "bbox": "vehicle",
        "box": "vehicle",
        "ped": "pedestrian",
        "person": "pedestrian",
        "people": "pedestrian",
        "pedestrian": "pedestrian",
        "pedestrians": "pedestrian",
        "bike": "bike",
        "bikes": "bike",
        "bicycle": "bike",
        "cyclist": "bike",
        "motorcycle": "bike",
    }
    return aliases.get(text, text or None)


def _looks_like_bbox(value: Any) -> bool:
    value = _tolist(value)
    if isinstance(value, Mapping):
        return any(key in value for key in ("bbox", "box", "corners", "points"))
    if not isinstance(value, (list, tuple)):
        return False
    if _is_flat_number_sequence(value):
        return len(value) >= 4
    return bool(value) and all(
        isinstance(point, (list, tuple)) and len(point) >= 2 for point in value[:4]
    )


def _is_flat_number_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    return all(_optional_float(item) is not None for item in value)


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is _MISSING or value is None:
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


class _Missing:
    pass


_MISSING = _Missing()


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return _MISSING


__all__ = [
    "TRANSFUSER_MODEL_ID",
    "Sd2JsonlWriter",
    "build_transfuser_run_metadata",
    "transfuser_record_to_sd2",
    "write_sd2_jsonl",
]
