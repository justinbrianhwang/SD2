"""Pure CARLA-to-SD2 JSONL conversion helpers.

This module intentionally does not import ``carla``. The live recorder extracts
plain Python measurements from CARLA and passes those dictionaries here.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TextIO

from sd2.core.schema import FrameLog, RunMetadata


CARLA_MODEL_ID = "carla_basic_agent"
LOGGER = logging.getLogger(__name__)


class Sd2JsonlWriter:
    """Stream schema-validated SD2 JSONL records through a partial file."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self._metadata_record = _validate_metadata_record(metadata)
        self._run_id = str(self._metadata_record["run_id"])
        self._handle: TextIO | None = None
        self._closed = False

        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.partial_path.open("w", encoding="utf-8")
        self._write_record(self._metadata_record)
        self._handle.flush()

    def write_frame(self, frame: dict[str, Any]) -> None:
        """Validate and append one frame record, then flush it."""

        if self._closed or self._handle is None:
            raise ValueError("cannot write to a closed SD2 JSONL writer")

        frame_record = _validate_frame_record(frame)
        if frame_record["run_id"] != self._run_id:
            raise ValueError(
                f"frame run_id {frame_record['run_id']!r} does not match "
                f"metadata run_id {self._run_id!r}"
            )

        self._write_record(frame_record)
        self._handle.flush()

    def close(self) -> None:
        """Close the partial file and atomically promote it to the final path."""

        self._close(finalize=True)

    def __enter__(self) -> "Sd2JsonlWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        del exc, traceback
        if exc_type is None:
            self.close()
        else:
            try:
                self._close(finalize=False)
            except Exception:
                LOGGER.exception(
                    "Failed to close incomplete SD2 JSONL partial file %s",
                    self.partial_path,
                )
            LOGGER.warning(
                "SD2 JSONL run failed; partial file left at %s",
                self.partial_path,
            )
        return False

    def _write_record(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise ValueError("cannot write to a closed SD2 JSONL writer")
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _close(self, *, finalize: bool) -> None:
        if self._closed:
            return
        handle = self._handle
        self._handle = None
        self._closed = True

        if handle is not None:
            handle.close()

        if finalize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.partial_path.replace(self.path)


def carla_frame_to_sd2(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Convert a plain-Python CARLA measurement dictionary into an SD2 frame.

    Only the planning, control, and outcome stages are populated. Ego pose and
    speed are preserved as an extra field on the planning state, which is valid
    because stage states allow adapter-specific fields.
    """

    planning: dict[str, Any] = {}
    waypoints = _coerce_waypoints(record.get("planned_waypoints"))
    if waypoints is not None:
        planning["waypoints"] = waypoints

    target_speed = _optional_float(record.get("target_speed"))
    if target_speed is not None:
        planning["target_speed"] = target_speed

    ego = _coerce_ego(record.get("ego"))
    if ego:
        planning["ego"] = ego

    control_record = _as_mapping(record.get("control"))
    control = {
        "steer": _optional_float(control_record.get("steer"), default=0.0),
        "throttle": _optional_float(control_record.get("throttle"), default=0.0),
        "brake": _optional_float(control_record.get("brake"), default=0.0),
    }
    if control_record.get("anti_crawl_applied") is not None:
        control["anti_crawl_applied"] = bool(control_record.get("anti_crawl_applied"))
    applied = _optional_float(control_record.get("applied_throttle"))
    if applied is not None:
        control["applied_throttle"] = applied

    outcome: dict[str, Any] = {
        "collision": bool(record.get("collision", False)),
        "lane_invasion": bool(record.get("lane_invasion", False)),
    }
    route_progress = _optional_float(record.get("route_progress"))
    if route_progress is not None:
        outcome["route_progress"] = _clamp(route_progress, 0.0, 1.0)
    if record.get("off_route") is not None:
        outcome["off_route"] = bool(record.get("off_route"))
    min_ttc = _optional_float(record.get("min_ttc"))
    if min_ttc is not None:
        outcome["min_ttc"] = min_ttc

    payload = {
        "run_id": run_id,
        "frame_idx": int(record.get("frame_idx", 0)),
        "timestamp": float(record.get("timestamp", 0.0)),
        "states": {
            "planning": planning,
            "control": control,
            "outcome": outcome,
        },
    }
    frame = FrameLog.model_validate(payload)
    return {"type": "frame", **frame.model_dump(mode="json", exclude_none=True)}


def build_carla_run_metadata(
    run_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None,
    severity: int,
    seed: int,
    town: str,
) -> dict[str, Any]:
    """Build schema-valid SD2 metadata for a CARLA BasicAgent run.

    ``RunMetadata`` has no free-form fields, so the CARLA town is folded into
    ``scenario_id`` when the caller has not already included it.
    """

    canonical_scenario_id = _scenario_with_town(scenario_id, town)
    normalized_stress_type = None if stress_type in (None, "", "none") else str(stress_type)
    metadata = RunMetadata.model_validate(
        {
            "run_id": run_id,
            "model_id": CARLA_MODEL_ID,
            "scenario_id": canonical_scenario_id,
            "condition": condition,
            "stress_type": normalized_stress_type,
            "severity": int(severity),
            "seed": int(seed),
            "timestamp_start": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    return {
        "type": "run_metadata",
        **metadata.model_dump(mode="json", exclude_none=True),
    }


def write_sd2_jsonl(
    path: str | Path,
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
) -> None:
    """Write a schema-validated SD2 JSONL run file."""

    with Sd2JsonlWriter(path, metadata) as writer:
        for frame in frames:
            writer.write_frame(frame)


def _validate_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_without_type(record, "run_metadata")
    metadata = RunMetadata.model_validate(payload)
    return {
        "type": "run_metadata",
        **metadata.model_dump(mode="json", exclude_none=True),
    }


def _validate_frame_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_without_type(record, "frame")
    frame = FrameLog.model_validate(payload)
    return {"type": "frame", **frame.model_dump(mode="json", exclude_none=True)}


def _payload_without_type(record: dict[str, Any], expected_type: str) -> dict[str, Any]:
    record_type = record.get("type")
    if record_type != expected_type:
        raise ValueError(f"expected record type {expected_type!r}, got {record_type!r}")
    return {key: value for key, value in record.items() if key != "type"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _coerce_waypoints(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    waypoints: list[list[float]] = []
    for point in value:
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


def _optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def _scenario_with_town(scenario_id: str, town: str) -> str:
    scenario = str(scenario_id)
    town_text = str(town)
    if not town_text:
        return scenario
    if town_text.lower() in scenario.lower():
        return scenario
    return f"{town_text}_{scenario}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


__all__ = [
    "CARLA_MODEL_ID",
    "Sd2JsonlWriter",
    "build_carla_run_metadata",
    "carla_frame_to_sd2",
    "write_sd2_jsonl",
]
