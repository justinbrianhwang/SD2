"""Vision-stage deviation metrics."""

from __future__ import annotations

from math import sqrt
from typing import Any

from sd2.core.stage import Stage
from sd2.metrics.base import MetricResult, StageMetric, clip01, register_metric


@register_metric("embedding_cosine")
class VisionEmbeddingCosineMetric(StageMetric):
    """Cosine distance between clean and stress vision embeddings."""

    def __init__(self, stage: Stage, name: str = "embedding_cosine") -> None:
        super().__init__(stage=stage, name=name)

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        clean_vector = _extract_embedding(clean_state)
        stress_vector = _extract_embedding(stress_state)
        if clean_vector is None or stress_vector is None:
            return self._missing(
                "missing_embedding",
                clean_has_embedding=clean_vector is not None,
                stress_has_embedding=stress_vector is not None,
            )
        if len(clean_vector) != len(stress_vector):
            return self._missing(
                "embedding_dimension_mismatch",
                clean_dim=len(clean_vector),
                stress_dim=len(stress_vector),
            )
        if not clean_vector:
            return self._missing("empty_embedding")

        clean_norm = sqrt(sum(value * value for value in clean_vector))
        stress_norm = sqrt(sum(value * value for value in stress_vector))
        if clean_norm == 0.0 or stress_norm == 0.0:
            return self._missing(
                "zero_vector_embedding",
                clean_norm=clean_norm,
                stress_norm=stress_norm,
            )

        dot_product = sum(
            clean_value * stress_value
            for clean_value, stress_value in zip(clean_vector, stress_vector)
        )
        cosine_similarity = dot_product / (clean_norm * stress_norm)
        raw_score = 1.0 - cosine_similarity
        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details={
                "cosine_similarity": cosine_similarity,
                "embedding_dim": len(clean_vector),
            },
        )


def _extract_embedding(state: Any) -> list[float] | None:
    raw_vector = getattr(state, "embedding", None)
    if raw_vector is None:
        raw_vector = getattr(state, "feature", None)
    if raw_vector is None:
        return None
    return [float(value) for value in raw_vector]
