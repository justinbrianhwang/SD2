"""Temporal stressors for frame-reference sequences.

Severity mappings:

- ``frame_drop``: seeded exact drop rate ``[0.10, 0.20, 0.30, 0.40, 0.50]``.
- ``frame_delay``: held-frame delay length ``[1, 2, 3, 4, 5]``.
- ``camera_blackout``: contiguous blank-window length ``[1, 2, 3, 4, 5]``.
- ``low_fps``: decimation factor ``[2, 3, 4, 5, 6]``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from sd2.stressors.base import (
    FrameRef,
    SequenceStressor,
    register_stressor,
    validate_severity,
)


@register_stressor("frame_drop")
class FrameDropStressor(SequenceStressor):
    """Drop a seeded exact fraction of frames; severity rates 0.10..0.50."""

    DROP_RATE_BY_SEVERITY = {
        1: 0.10,
        2: 0.20,
        3: 0.30,
        4: 0.40,
        5: 0.50,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"drop_rate": self.DROP_RATE_BY_SEVERITY[severity_value]}

    def apply_sequence(
        self,
        frames: list[FrameRef],
        severity: int,
        rng: np.random.Generator,
    ) -> list[FrameRef]:
        params = self.params_for_severity(severity)
        frame_count = len(frames)
        if frame_count == 0:
            self._set_report({"params": params, "dropped_indices": [], "dropped_positions": []})
            return []

        requested_drop_count = int(np.floor(frame_count * params["drop_rate"] + 0.5))
        drop_count = min(max(requested_drop_count, 0), max(frame_count - 1, 0))
        if drop_count == 0:
            drop_positions: list[int] = []
        else:
            drop_positions = sorted(
                int(position)
                for position in rng.choice(frame_count, size=drop_count, replace=False)
            )
        drop_set = set(drop_positions)
        result = [replace(frame) for idx, frame in enumerate(frames) if idx not in drop_set]
        dropped_indices = [frames[position].frame_idx for position in drop_positions]
        self._set_report(
            {
                "params": params,
                "dropped_count": len(drop_positions),
                "dropped_indices": dropped_indices,
                "dropped_positions": drop_positions,
            }
        )
        return result


@register_stressor("frame_delay")
class FrameDelayStressor(SequenceStressor):
    """Shift frames later by holding previous frames; severity delay 1..5."""

    DELAY_BY_SEVERITY = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"delay_frames": self.DELAY_BY_SEVERITY[severity_value]}

    def apply_sequence(
        self,
        frames: list[FrameRef],
        severity: int,
        rng: np.random.Generator,
    ) -> list[FrameRef]:
        del rng
        params = self.params_for_severity(severity)
        if not frames:
            self._set_report({"params": params, "mapping": []})
            return []

        delay = min(params["delay_frames"], len(frames) - 1)
        result: list[FrameRef] = []
        mapping: list[dict[str, int]] = []
        for output_position in range(len(frames)):
            source_position = max(0, output_position - delay)
            result.append(replace(frames[source_position]))
            mapping.append(
                {
                    "output_position": output_position,
                    "source_position": source_position,
                    "source_frame_idx": frames[source_position].frame_idx,
                }
            )

        self._set_report(
            {
                "params": params,
                "effective_delay_frames": delay,
                "mapping": mapping,
                "held_positions": list(range(min(delay, len(frames)))),
            }
        )
        return result


@register_stressor("camera_blackout")
class CameraBlackoutStressor(SequenceStressor):
    """Blank a contiguous window; severity window length 1..5."""

    WINDOW_BY_SEVERITY = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"window_length": self.WINDOW_BY_SEVERITY[severity_value]}

    def apply_sequence(
        self,
        frames: list[FrameRef],
        severity: int,
        rng: np.random.Generator,
    ) -> list[FrameRef]:
        params = self.params_for_severity(severity)
        if not frames:
            self._set_report(
                {
                    "params": params,
                    "window_start_position": None,
                    "blanked_indices": [],
                    "blanked_positions": [],
                }
            )
            return []

        window_length = min(params["window_length"], len(frames))
        max_start = len(frames) - window_length
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        blank_positions = list(range(start, start + window_length))
        blank_set = set(blank_positions)

        result = [
            replace(frame, image_path=None, blanked=True)
            if position in blank_set
            else replace(frame)
            for position, frame in enumerate(frames)
        ]
        blanked_indices = [frames[position].frame_idx for position in blank_positions]
        self._set_report(
            {
                "params": params,
                "window_start_position": start,
                "window_length": window_length,
                "blanked_indices": blanked_indices,
                "blanked_positions": blank_positions,
            }
        )
        return result


@register_stressor("low_fps")
class LowFpsStressor(SequenceStressor):
    """Subsample frames with severity decimation factor 2, 3, 4, 5, 6."""

    FACTOR_BY_SEVERITY = {
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
    }

    def params_for_severity(self, severity: int) -> dict[str, Any]:
        severity_value = validate_severity(severity)
        return {"decimation_factor": self.FACTOR_BY_SEVERITY[severity_value]}

    def apply_sequence(
        self,
        frames: list[FrameRef],
        severity: int,
        rng: np.random.Generator,
    ) -> list[FrameRef]:
        del rng
        params = self.params_for_severity(severity)
        factor = params["decimation_factor"]
        kept_positions = list(range(0, len(frames), factor))
        kept_set = set(kept_positions)
        result = [replace(frames[position]) for position in kept_positions]
        dropped_positions = [
            position for position in range(len(frames)) if position not in kept_set
        ]
        self._set_report(
            {
                "params": params,
                "kept_indices": [frames[position].frame_idx for position in kept_positions],
                "kept_positions": kept_positions,
                "dropped_indices": [
                    frames[position].frame_idx for position in dropped_positions
                ],
                "dropped_positions": dropped_positions,
            }
        )
        return result
