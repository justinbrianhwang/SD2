"""Control-stage deviation metrics."""

from __future__ import annotations

from typing import Any, Mapping

from sd2.core.stage import Stage
from sd2.metrics.base import (
    MetricResult,
    StageMetric,
    clip01,
    normalize_weights,
    register_metric,
)


_DEFAULT_WEIGHTS = {
    "steer": 0.5,
    "throttle": 0.25,
    "brake": 0.25,
}
_COMPONENT_RANGES = {
    "steer": 2.0,
    "throttle": 1.0,
    "brake": 1.0,
}


@register_metric("weighted_action_mae")
class ControlWeightedMAEMetric(StageMetric):
    """Weighted absolute control-command error."""

    def __init__(
        self,
        stage: Stage,
        name: str = "weighted_action_mae",
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(stage=stage, name=name)
        self.weights = normalize_weights(weights, _DEFAULT_WEIGHTS)

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        raw_differences: dict[str, float] = {}
        normalized_components: dict[str, float] = {}
        for component, component_range in _COMPONENT_RANGES.items():
            clean_value = getattr(clean_state, component, None)
            stress_value = getattr(stress_state, component, None)
            if clean_value is None or stress_value is None:
                return self._missing(
                    "missing_control_component",
                    component=component,
                    clean_has_value=clean_value is not None,
                    stress_has_value=stress_value is not None,
                )
            raw_difference = abs(float(clean_value) - float(stress_value))
            raw_differences[component] = raw_difference
            normalized_components[component] = clip01(raw_difference / component_range)

        raw_score = sum(
            self.weights[component] * normalized_components[component]
            for component in _DEFAULT_WEIGHTS
        )
        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details={
                "components": normalized_components,
                "raw_differences": raw_differences,
                "weights": self.weights,
            },
        )
