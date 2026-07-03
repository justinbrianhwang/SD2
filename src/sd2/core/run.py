"""Run containers and clean/stress frame pairing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sd2.core.schema import FrameLog, PairedFrameLog, RunMetadata


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


@dataclass(frozen=True)
class PairedRun:
    """Paired frames and pairing summary."""

    pairs: list[PairedFrameLog]
    summary: PairingSummary


def pair_runs(clean: RunLog, stress: RunLog) -> PairedRun:
    """Pair clean and stress runs by frame index.

    The two runs must share model ID, scenario ID, and seed. Frames that are
    present in only one run are skipped and counted in the summary.
    """

    _verify_compatible_runs(clean.metadata, stress.metadata)

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
        pair_key = (
            f"{clean.metadata.model_id}:"
            f"{clean.metadata.scenario_id}:"
            f"{clean.metadata.seed}:"
            f"{frame_idx}"
        )
        pairs.append(
            PairedFrameLog(
                pair_key=pair_key,
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
    )
    return PairedRun(pairs=pairs, summary=summary)


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
