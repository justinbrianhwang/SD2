"""JSONL adapter for the SD2 MVP run-log format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sd2.core.run import RunLog
from sd2.core.schema import FrameLog, RunMetadata


class JSONLLoadError(ValueError):
    """Raised when an SD2 JSONL run cannot be loaded."""


def load_run_jsonl(path: str | Path) -> RunLog:
    """Load a single-file SD2 JSONL run.

    The first non-empty line must be a ``run_metadata`` record. All subsequent
    non-empty lines must be ``frame`` records. Validation errors include the
    offending line number.
    """

    jsonl_path = Path(path)
    if not jsonl_path.is_file():
        raise JSONLLoadError(f"run log is not a file: {jsonl_path}")

    metadata: RunMetadata | None = None
    frames: list[FrameLog] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = _parse_json_line(jsonl_path, line_no, stripped)
            record_type = record.get("type")
            payload = {key: value for key, value in record.items() if key != "type"}

            if metadata is None:
                if record_type != "run_metadata":
                    raise JSONLLoadError(
                        f"{jsonl_path}:{line_no}: first record must have "
                        f"type='run_metadata', got {record_type!r}"
                    )
                metadata = _validate_metadata(jsonl_path, line_no, payload)
                continue

            if record_type != "frame":
                raise JSONLLoadError(
                    f"{jsonl_path}:{line_no}: expected type='frame', got {record_type!r}"
                )
            frame = _validate_frame(jsonl_path, line_no, payload)
            if frame.run_id != metadata.run_id:
                raise JSONLLoadError(
                    f"{jsonl_path}:{line_no}: frame run_id {frame.run_id!r} "
                    f"does not match metadata run_id {metadata.run_id!r}"
                )
            frames.append(frame)

    if metadata is None:
        raise JSONLLoadError(f"{jsonl_path}: missing run_metadata record")
    return RunLog(metadata=metadata, frames=frames)


def _parse_json_line(path: Path, line_no: int, line: str) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise JSONLLoadError(
            f"{path}:{line_no}: invalid JSON: {exc.msg} at column {exc.colno}"
        ) from exc
    if not isinstance(record, dict):
        raise JSONLLoadError(f"{path}:{line_no}: JSONL record must be an object")
    return record


def _validate_metadata(path: Path, line_no: int, payload: dict[str, Any]) -> RunMetadata:
    try:
        return RunMetadata.model_validate(payload)
    except ValidationError as exc:
        raise JSONLLoadError(f"{path}:{line_no}: invalid run_metadata: {exc}") from exc


def _validate_frame(path: Path, line_no: int, payload: dict[str, Any]) -> FrameLog:
    try:
        return FrameLog.model_validate(payload)
    except ValidationError as exc:
        raise JSONLLoadError(f"{path}:{line_no}: invalid frame: {exc}") from exc
