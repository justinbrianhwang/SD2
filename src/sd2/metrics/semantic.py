"""Semantic-stage deviation metrics."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from sd2.core.stage import Stage
from sd2.metrics.base import (
    MetricResult,
    StageMetric,
    clip01,
    normalize_weights,
    register_metric,
)


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

        raw_score, details = _object_jaccard_from_sets(
            clean_state,
            stress_state,
            clean_objects,
            stress_objects,
        )

        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details=details,
        )


@register_metric("semantic_composite")
class SemanticCompositeMetric(StageMetric):
    """Semantic deviation from object-set and BEV-seg distribution changes."""

    def __init__(
        self,
        stage: Stage,
        name: str = "semantic_composite",
        object_weight: float = 0.5,
        seg_weight: float = 0.5,
    ) -> None:
        super().__init__(stage=stage, name=name)
        self.weights = normalize_weights(
            {
                "object_jaccard": object_weight,
                "seg_tv": seg_weight,
            },
            {
                "object_jaccard": 0.5,
                "seg_tv": 0.5,
            },
        )

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        components: list[str] = []
        component_scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        clean_objects = _object_set(clean_state)
        stress_objects = _object_set(stress_state)
        if (
            clean_objects is not None
            and stress_objects is not None
            and (clean_objects or stress_objects)
        ):
            object_score, object_details = _object_jaccard_from_sets(
                clean_state,
                stress_state,
                clean_objects,
                stress_objects,
            )
            components.append("object_jaccard")
            component_scores["object_jaccard"] = object_score
            details.update(object_details)
            details["object_jaccard_score"] = object_score

        seg_component = _seg_tv_component(clean_state, stress_state)
        if seg_component is not None:
            seg_score, seg_details = seg_component
            components.append("seg_tv")
            component_scores["seg_tv"] = seg_score
            details.update(seg_details)
            details["seg_tv_score"] = seg_score

        if not components:
            return self._missing(
                "missing_semantic_signal",
                clean_has_objects=clean_objects is not None,
                stress_has_objects=stress_objects is not None,
                clean_has_bev_seg_summary=_raw_bev_seg_summary(clean_state) is not None,
                stress_has_bev_seg_summary=_raw_bev_seg_summary(stress_state) is not None,
            )

        active_weights = (
            dict(self.weights)
            if len(components) == 2
            else {components[0]: 1.0}
        )
        raw_score = sum(
            active_weights[component] * component_scores[component]
            for component in components
        )

        details["components"] = components
        details["component_scores"] = component_scores
        details["weights"] = active_weights

        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details=details,
        )


def _object_jaccard_from_sets(
    clean_state: Any,
    stress_state: Any,
    clean_objects: set[str],
    stress_objects: set[str],
) -> tuple[float, dict[str, Any]]:
    union = clean_objects | stress_objects
    intersection = clean_objects & stress_objects
    jaccard_similarity = 1.0 if not union else len(intersection) / len(union)
    raw_score = 1.0 - jaccard_similarity

    clean_critical = _normalized_text(getattr(clean_state, "critical_object", None))
    stress_critical = _normalized_text(getattr(stress_state, "critical_object", None))
    clean_light = _normalized_text(getattr(clean_state, "traffic_light_state", None))
    stress_light = _normalized_text(getattr(stress_state, "traffic_light_state", None))

    return raw_score, {
        "jaccard_similarity": jaccard_similarity,
        "missing_objects": sorted(clean_objects - stress_objects),
        "extra_objects": sorted(stress_objects - clean_objects),
        "critical_object_mismatch": clean_critical != stress_critical,
        "traffic_light_mismatch": clean_light != stress_light,
    }


def _seg_tv_component(
    clean_state: Any,
    stress_state: Any,
) -> tuple[float, dict[str, Any]] | None:
    clean_distribution = _bev_seg_distribution(clean_state)
    stress_distribution = _bev_seg_distribution(stress_state)
    if clean_distribution is None or stress_distribution is None:
        return None

    clean_counts, clean_probs, clean_meta = clean_distribution
    stress_counts, stress_probs, stress_meta = stress_distribution
    tv_distance = 0.5 * sum(
        abs(clean_prob - stress_prob)
        for clean_prob, stress_prob in zip(clean_probs, stress_probs)
    )
    return tv_distance, {
        "seg_class_counts": {
            "clean": clean_counts,
            "stress": stress_counts,
        },
        "seg_class_distributions": {
            "clean": clean_probs,
            "stress": stress_probs,
        },
        "seg_nonzero_fraction": {
            "clean": clean_meta.get("nonzero_fraction"),
            "stress": stress_meta.get("nonzero_fraction"),
        },
        "seg_dominant_class": {
            "clean": clean_meta.get("dominant_class"),
            "stress": stress_meta.get("dominant_class"),
        },
    }


def _bev_seg_distribution(
    state: Any,
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    summary = _raw_bev_seg_summary(state)
    if not isinstance(summary, Mapping):
        return None

    counts: list[float] = []
    for key in ("class_0", "class_1", "class_2"):
        value = _finite_float(summary.get(key))
        if value is None or value < 0.0:
            return None
        counts.append(value)

    total = sum(counts)
    if total <= 0.0:
        return None

    return (
        counts,
        [count / total for count in counts],
        {
            "nonzero_fraction": _finite_float(summary.get("nonzero_fraction")),
            "dominant_class": _optional_int(summary.get("dominant_class")),
        },
    )


def _raw_bev_seg_summary(state: Any) -> Any:
    summary = getattr(state, "bev_seg_summary", None)
    if summary is None and hasattr(state, "model_extra"):
        model_extra = getattr(state, "model_extra", None)
        if isinstance(model_extra, Mapping):
            summary = model_extra.get("bev_seg_summary")
    return summary


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
