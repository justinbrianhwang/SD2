"""Run containers and clean/stress frame pairing."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sd2.core.schema import FrameLog, PairedFrameLog, RunMetadata
from sd2.core.stage import Stage


DEFAULT_PAIRING_MODE = "frame_idx"
DEFAULT_TIMESTAMP_TOLERANCE = 0.06
DEFAULT_PROGRESS_TOLERANCE = 0.02
PAIRING_MODES = {"frame_idx", "timestamp", "route_progress"}


@dataclass(frozen=True)
class RunLog:
    """Loaded SD2 run log."""

    metadata: RunMetadata
    frames: list[FrameLog]


class PairingSummary(BaseModel):
    """Summary of frame pairing and skipped frames."""

    model_config = ConfigDict(extra="forbid")

    clean_run_id: str
    stress_run_id: str
    model_id: str
    scenario_id: str
    seed: int
    clean_frame_count: int
    stress_frame_count: int
    paired_count: int
    skipped_count: int
    missing_in_clean: list[int] = Field(default_factory=list)
    missing_in_stress: list[int] = Field(default_factory=list)
    clean_metadata: dict[str, Any] = Field(default_factory=dict)
    stress_metadata: dict[str, Any] = Field(default_factory=dict)
    mode: str = DEFAULT_PAIRING_MODE
    mean_anchor_mismatch: float | None = 0.0
    max_anchor_mismatch: float | None = 0.0


@dataclass(frozen=True)
class PairedRun:
    """Paired frames and pairing summary."""

    pairs: list[PairedFrameLog]
    summary: PairingSummary


def pair_runs(
    clean: RunLog,
    stress: RunLog,
    mode: str = DEFAULT_PAIRING_MODE,
    *,
    timestamp_tolerance: float = DEFAULT_TIMESTAMP_TOLERANCE,
    progress_tolerance: float = DEFAULT_PROGRESS_TOLERANCE,
) -> PairedRun:
    """Pair clean and stress runs by the selected anchor.

    The two runs must share model ID, scenario ID, and seed. Frames that are
    present in only one run are skipped and counted in the summary for
    ``frame_idx`` mode. In clean-centric anchor modes, skipped frames are clean
    frames with no stress match within tolerance.

    Summary anchor mismatch units are frame-index delta for ``frame_idx``
    (always 0.0), seconds for ``timestamp``, and route-progress fraction for
    ``route_progress``.
    """

    _verify_compatible_runs(clean.metadata, stress.metadata)
    normalized_mode = _normalize_pairing_mode(mode)

    if normalized_mode == "frame_idx":
        return _pair_by_frame_idx(clean, stress)
    if normalized_mode == "timestamp":
        return _pair_by_anchor(
            clean,
            stress,
            mode=normalized_mode,
            tolerance=_validate_tolerance(
                timestamp_tolerance,
                "timestamp_tolerance",
            ),
            anchor_getter=lambda frame: float(frame.timestamp),
        )

    _validate_route_progress_available(clean, "clean")
    _validate_route_progress_available(stress, "stress")
    return _pair_by_anchor(
        clean,
        stress,
        mode=normalized_mode,
        tolerance=_validate_tolerance(progress_tolerance, "progress_tolerance"),
        anchor_getter=_route_progress,
    )


def _pair_by_frame_idx(clean: RunLog, stress: RunLog) -> PairedRun:
    clean_by_idx = _frames_by_index(clean.frames, "clean")
    stress_by_idx = _frames_by_index(stress.frames, "stress")

    clean_indices = set(clean_by_idx)
    stress_indices = set(stress_by_idx)
    paired_indices = sorted(clean_indices & stress_indices)
    missing_in_clean = sorted(stress_indices - clean_indices)
    missing_in_stress = sorted(clean_indices - stress_indices)

    pairs: list[PairedFrameLog] = []
    for frame_idx in paired_indices:
        clean_frame = clean_by_idx[frame_idx]
        stress_frame = stress_by_idx[frame_idx]
        pairs.append(
            PairedFrameLog(
                pair_key=_pair_key(clean.metadata, frame_idx),
                frame_idx=frame_idx,
                timestamp=clean_frame.timestamp,
                clean=clean_frame,
                stress=stress_frame,
            )
        )

    summary = PairingSummary(
        clean_run_id=clean.metadata.run_id,
        stress_run_id=stress.metadata.run_id,
        model_id=clean.metadata.model_id,
        scenario_id=clean.metadata.scenario_id,
        seed=clean.metadata.seed,
        clean_frame_count=len(clean.frames),
        stress_frame_count=len(stress.frames),
        paired_count=len(pairs),
        skipped_count=len(missing_in_clean) + len(missing_in_stress),
        missing_in_clean=missing_in_clean,
        missing_in_stress=missing_in_stress,
        clean_metadata=clean.metadata.model_dump(mode="json"),
        stress_metadata=stress.metadata.model_dump(mode="json"),
        mode="frame_idx",
        mean_anchor_mismatch=0.0,
        max_anchor_mismatch=0.0,
    )
    return PairedRun(pairs=pairs, summary=summary)


def _pair_by_anchor(
    clean: RunLog,
    stress: RunLog,
    *,
    mode: str,
    tolerance: float,
    anchor_getter: Callable[[FrameLog], float],
) -> PairedRun:
    _frames_by_index(clean.frames, "clean")
    _frames_by_index(stress.frames, "stress")

    stress_candidates = sorted(
        ((anchor_getter(frame), frame) for frame in stress.frames),
        key=lambda item: (item[0], item[1].frame_idx),
    )
    stress_anchor_values = [anchor for anchor, _frame in stress_candidates]
    clean_candidates = sorted(
        ((anchor_getter(frame), frame) for frame in clean.frames),
        key=lambda item: (item[0], item[1].frame_idx),
    )
    clean_anchor_values = [anchor for anchor, _frame in clean_candidates]

    pairs: list[PairedFrameLog] = []
    skipped_clean_indices: list[int] = []
    anchor_mismatches: list[float] = []
    for clean_frame in sorted(clean.frames, key=lambda frame: frame.frame_idx):
        clean_anchor = anchor_getter(clean_frame)
        match = _closest_frame_by_anchor(
            clean_anchor,
            stress_candidates,
            stress_anchor_values,
            tolerance,
        )
        if match is None:
            skipped_clean_indices.append(clean_frame.frame_idx)
            continue

        stress_frame, mismatch = match
        anchor_mismatches.append(mismatch)
        pairs.append(
            PairedFrameLog(
                pair_key=_pair_key(clean.metadata, clean_frame.frame_idx),
                frame_idx=clean_frame.frame_idx,
                timestamp=clean_frame.timestamp,
                clean=clean_frame,
                stress=stress_frame,
            )
        )

    missing_in_clean = sorted(
        stress_frame.frame_idx
        for stress_anchor, stress_frame in stress_candidates
        if _closest_frame_by_anchor(
            stress_anchor,
            clean_candidates,
            clean_anchor_values,
            tolerance,
        )
        is None
    )
    mean_anchor_mismatch, max_anchor_mismatch = _mismatch_stats(anchor_mismatches)
    summary = PairingSummary(
        clean_run_id=clean.metadata.run_id,
        stress_run_id=stress.metadata.run_id,
        model_id=clean.metadata.model_id,
        scenario_id=clean.metadata.scenario_id,
        seed=clean.metadata.seed,
        clean_frame_count=len(clean.frames),
        stress_frame_count=len(stress.frames),
        paired_count=len(pairs),
        skipped_count=len(skipped_clean_indices),
        missing_in_clean=missing_in_clean,
        missing_in_stress=skipped_clean_indices,
        clean_metadata=clean.metadata.model_dump(mode="json"),
        stress_metadata=stress.metadata.model_dump(mode="json"),
        mode=mode,
        mean_anchor_mismatch=mean_anchor_mismatch,
        max_anchor_mismatch=max_anchor_mismatch,
    )
    return PairedRun(pairs=pairs, summary=summary)


def _closest_frame_by_anchor(
    clean_anchor: float,
    candidates: list[tuple[float, FrameLog]],
    anchor_values: list[float],
    tolerance: float,
) -> tuple[FrameLog, float] | None:
    insertion_index = bisect_left(anchor_values, clean_anchor)
    best: tuple[float, int, FrameLog] | None = None
    for candidate_index in (insertion_index - 1, insertion_index):
        if candidate_index < 0 or candidate_index >= len(candidates):
            continue
        anchor, frame = candidates[candidate_index]
        mismatch = abs(anchor - clean_anchor)
        if mismatch > tolerance:
            continue
        candidate = (mismatch, frame.frame_idx, frame)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    mismatch, _frame_idx, frame = best
    return frame, mismatch


def _mismatch_stats(mismatches: list[float]) -> tuple[float | None, float | None]:
    if not mismatches:
        return None, None
    return sum(mismatches) / len(mismatches), max(mismatches)


def _pair_key(metadata: RunMetadata, clean_frame_idx: int) -> str:
    return (
        f"{metadata.model_id}:"
        f"{metadata.scenario_id}:"
        f"{metadata.seed}:"
        f"{clean_frame_idx}"
    )


def _normalize_pairing_mode(mode: str) -> str:
    normalized = str(mode or DEFAULT_PAIRING_MODE).strip().lower()
    if normalized not in PAIRING_MODES:
        allowed = ", ".join(sorted(PAIRING_MODES))
        raise ValueError(f"unsupported pairing mode {mode!r}; expected one of: {allowed}")
    return normalized


def _validate_tolerance(raw_value: float, name: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pairing {name} must be a number") from exc
    if value < 0:
        raise ValueError(f"pairing {name} must be non-negative")
    return value


def _validate_route_progress_available(run: RunLog, label: str) -> None:
    missing = [
        frame.frame_idx
        for frame in run.frames
        if _frame_route_progress(frame) is None
    ]
    if not missing:
        return
    preview = ", ".join(str(frame_idx) for frame_idx in missing[:5])
    suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
    raise ValueError(
        "route_progress pairing requires outcome.route_progress on every clean "
        "and stress frame; "
        f"missing in {label} run {run.metadata.run_id!r} at frame_idx "
        f"{preview}{suffix}. Use frame_idx mode or provide route_progress in "
        "the run logs."
    )


def _route_progress(frame: FrameLog) -> float:
    progress = _frame_route_progress(frame)
    if progress is None:
        raise AssertionError("route_progress availability was not validated")
    return float(progress)


def _frame_route_progress(frame: FrameLog) -> float | None:
    outcome = frame.states.get(Stage.OUTCOME)
    if outcome is None:
        return None
    progress = getattr(outcome, "route_progress", None)
    return None if progress is None else float(progress)


def _verify_compatible_runs(clean: RunMetadata, stress: RunMetadata) -> None:
    mismatches: list[str] = []
    for field_name in ("model_id", "scenario_id", "seed"):
        clean_value = getattr(clean, field_name)
        stress_value = getattr(stress, field_name)
        if clean_value != stress_value:
            mismatches.append(
                f"{field_name} mismatch (clean={clean_value!r}, stress={stress_value!r})"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"cannot pair runs {clean.run_id!r} and {stress.run_id!r}: {details}")


def _frames_by_index(frames: list[FrameLog], label: str) -> dict[int, FrameLog]:
    by_index: dict[int, FrameLog] = {}
    duplicates: list[int] = []
    for frame in frames:
        if frame.frame_idx in by_index:
            duplicates.append(frame.frame_idx)
        by_index[frame.frame_idx] = frame
    if duplicates:
        duplicate_text = ", ".join(str(idx) for idx in sorted(set(duplicates)))
        raise ValueError(f"{label} run contains duplicate frame_idx values: {duplicate_text}")
    return by_index
