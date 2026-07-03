"""Semantic-stage deviation metrics."""

from __future__ import annotations

from typing import Any

from sd2.core.stage import Stage
from sd2.metrics.base import MetricResult, StageMetric, clip01, register_metric


@register_metric("object_jaccard")
class SemanticObjectJaccardMetric(StageMetric):
    """Jaccard distance between clean and stress object sets."""

    def __init__(self, stage: Stage, name: str = "object_jaccard") -> None:
        super().__init__(stage=stage, name=name)

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        clean_objects = _object_set(clean_state)
        stress_objects = _object_set(stress_state)
        if clean_objects is None or stress_objects is None:
            return self._missing(
                "missing_objects",
                clean_has_objects=clean_objects is not None,
                stress_has_objects=stress_objects is not None,
            )

        union = clean_objects | stress_objects
        intersection = clean_objects & stress_objects
        jaccard_similarity = 1.0 if not union else len(intersection) / len(union)
        raw_score = 1.0 - jaccard_similarity

        clean_critical = _normalized_text(getattr(clean_state, "critical_object", None))
        stress_critical = _normalized_text(getattr(stress_state, "critical_object", None))
        clean_light = _normalized_text(getattr(clean_state, "traffic_light_state", None))
        stress_light = _normalized_text(getattr(stress_state, "traffic_light_state", None))

        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details={
                "jaccard_similarity": jaccard_similarity,
                "missing_objects": sorted(clean_objects - stress_objects),
                "extra_objects": sorted(stress_objects - clean_objects),
                "critical_object_mismatch": clean_critical != stress_critical,
                "traffic_light_mismatch": clean_light != stress_light,
            },
        )


def _object_set(state: Any) -> set[str] | None:
    objects = getattr(state, "objects", None)
    if objects is None:
        return None
    return {str(obj).strip().lower() for obj in objects if str(obj).strip()}


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None
