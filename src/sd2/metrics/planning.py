"""Planning-stage deviation metrics."""

from __future__ import annotations

from math import sqrt
from typing import Any

from sd2.core.stage import Stage
from sd2.metrics.base import MetricResult, StageMetric, clip01, register_metric


@register_metric("waypoint_ade")
class PlanningADEMetric(StageMetric):
    """Average displacement error between waypoint sequences."""

    def __init__(
        self,
        stage: Stage,
        name: str = "waypoint_ade",
        ade_scale: float = 5.0,
    ) -> None:
        super().__init__(stage=stage, name=name)
        self.ade_scale = float(ade_scale)
        if self.ade_scale <= 0.0:
            raise ValueError("ade_scale must be positive")

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        clean_waypoints = _extract_waypoints(clean_state)
        stress_waypoints = _extract_waypoints(stress_state)
        if clean_waypoints is None or stress_waypoints is None:
            return self._missing(
                "missing_waypoints",
                clean_has_waypoints=clean_waypoints is not None,
                stress_has_waypoints=stress_waypoints is not None,
            )

        compared_length = min(len(clean_waypoints), len(stress_waypoints))
        if compared_length == 0:
            return self._missing(
                "empty_waypoints",
                clean_length=len(clean_waypoints),
                stress_length=len(stress_waypoints),
            )

        try:
            distances = [
                _point_distance(clean_waypoints[idx], stress_waypoints[idx])
                for idx in range(compared_length)
            ]
        except (TypeError, ValueError) as exc:
            return self._missing("invalid_waypoint", error=str(exc))

        ade = sum(distances) / compared_length
        fde = distances[-1]
        target_speed_diff = _target_speed_diff(clean_state, stress_state)
        return MetricResult(
            raw_score=ade,
            normalized_score=clip01(ade / self.ade_scale),
            details={
                "ade": ade,
                "fde": fde,
                "target_speed_diff": target_speed_diff,
                "clean_length": len(clean_waypoints),
                "stress_length": len(stress_waypoints),
                "compared_length": compared_length,
                "length_mismatch": len(clean_waypoints) != len(stress_waypoints),
                "ade_scale": self.ade_scale,
            },
        )


def _extract_waypoints(state: Any) -> list[list[float]] | None:
    waypoints = getattr(state, "waypoints", None)
    if waypoints is None:
        waypoints = getattr(state, "trajectory", None)
    return waypoints


def _point_distance(clean_point: list[float], stress_point: list[float]) -> float:
    if len(clean_point) < 2 or len(stress_point) < 2:
        raise ValueError("waypoints must contain at least x and y coordinates")
    dx = float(clean_point[0]) - float(stress_point[0])
    dy = float(clean_point[1]) - float(stress_point[1])
    return sqrt((dx * dx) + (dy * dy))


def _target_speed_diff(clean_state: Any, stress_state: Any) -> float | None:
    clean_speed = getattr(clean_state, "target_speed", None)
    stress_speed = getattr(stress_state, "target_speed", None)
    if clean_speed is None or stress_speed is None:
        return None
    return abs(float(clean_speed) - float(stress_speed))
