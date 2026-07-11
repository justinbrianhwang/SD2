"""Outcome event counting helpers."""

from __future__ import annotations


# gap_tolerance=2 merges short CARLA sensor flicker inside one physical contact,
# including the one-frame gap observed in the NEAT pinning recording.
DEFAULT_EVENT_GAP_TOLERANCE = 2


def _count_events(
    flags: list[bool],
    *,
    gap_tolerance: int = DEFAULT_EVENT_GAP_TOLERANCE,
) -> int:
    """Count contiguous runs of True, merging runs separated by <= gap_tolerance False frames.
    One physical collision that flickers or briefly separates counts once.
    """

    return len(_event_start_indices(flags, gap_tolerance=gap_tolerance))


def _event_start_indices(
    flags: list[bool],
    *,
    gap_tolerance: int = DEFAULT_EVENT_GAP_TOLERANCE,
) -> list[int]:
    """Return the first index of each gap-merged True event."""

    if gap_tolerance < 0:
        raise ValueError("gap_tolerance must be non-negative")

    starts: list[int] = []
    in_event = False
    false_gap = 0
    for index, flag in enumerate(flags):
        if flag:
            if not in_event:
                starts.append(index)
                in_event = True
            false_gap = 0
            continue

        if in_event:
            false_gap += 1
            if false_gap > gap_tolerance:
                in_event = False
                false_gap = 0

    return starts
