"""Base interfaces and registry for stage-wise deviation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from sd2.core.stage import Stage


@dataclass(frozen=True)
class MetricResult:
    """Result produced by a stage metric for one clean/stress state pair."""

    raw_score: float
    normalized_score: float
    details: dict[str, Any] = field(default_factory=dict)
    missing: bool = False


class StageMetric(ABC):
    """Abstract base for stage-specific deviation metrics."""

    stage: Stage
    name: str

    def __init__(self, stage: Stage, name: str) -> None:
        self.stage = stage
        self.name = name

    @abstractmethod
    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        """Compute deviation between clean and stress stage states."""

    def _missing(self, reason: str, **details: Any) -> MetricResult:
        payload = {"reason": reason}
        payload.update(details)
        return MetricResult(
            raw_score=0.0,
            normalized_score=0.0,
            details=payload,
            missing=True,
        )


MetricFactory = type[StageMetric]
_METRIC_REGISTRY: dict[str, MetricFactory] = {}


def register_metric(metric_type: str):
    """Register a metric class under a config ``type`` string."""

    def decorator(metric_cls: MetricFactory) -> MetricFactory:
        if not issubclass(metric_cls, StageMetric):
            raise TypeError(f"{metric_cls.__name__} must inherit StageMetric")
        _METRIC_REGISTRY[metric_type] = metric_cls
        return metric_cls

    return decorator


def available_metric_types() -> list[str]:
    """Return registered metric type names."""

    return sorted(_METRIC_REGISTRY)


def build_metric(stage: Stage | str, config: Mapping[str, Any]) -> StageMetric:
    """Build a stage metric from a YAML metric config block."""

    if not isinstance(config, Mapping):
        raise ValueError(f"metric config for stage {stage!s} must be a mapping")
    metric_type = config.get("type")
    if not isinstance(metric_type, str) or not metric_type:
        raise ValueError(f"metric config for stage {stage!s} must include a type")

    metric_cls = _METRIC_REGISTRY.get(metric_type)
    if metric_cls is None:
        available = ", ".join(available_metric_types()) or "<none>"
        raise ValueError(
            f"unknown metric type {metric_type!r} for stage {stage!s}; "
            f"available types: {available}"
        )

    stage_value = stage if isinstance(stage, Stage) else Stage(str(stage))
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in {"type", "name"}
    }
    name = str(config.get("name") or metric_type)
    return metric_cls(stage=stage_value, name=name, **kwargs)


def clip01(value: float) -> float:
    """Clip a finite float to the inclusive 0..1 interval."""

    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def normalize_weights(
    weights: Mapping[str, Any] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    """Normalize metric weights to sum to one."""

    source = weights or {}
    normalized_input: dict[str, float] = {}
    for key, default_value in defaults.items():
        value = source.get(key, default_value)
        try:
            normalized_input[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weight {key!r} must be numeric") from exc
        if normalized_input[key] < 0.0:
            raise ValueError(f"weight {key!r} must be non-negative")

    total = sum(normalized_input.values())
    if total <= 0.0:
        raise ValueError("metric weights must sum to a positive value")
    return {key: value / total for key, value in normalized_input.items()}
