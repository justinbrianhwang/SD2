"""Reasoning-stage deviation metrics."""

from __future__ import annotations

import re
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
    "text_embedding": 0.5,
    "intent_mismatch": 0.3,
    "critical_object_mismatch": 0.2,
}
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@register_metric("text_embedding_and_intent")
class ReasoningIntentMismatchMetric(StageMetric):
    """Weighted lexical, intent, and critical-object reasoning distance."""

    def __init__(
        self,
        stage: Stage,
        name: str = "text_embedding_and_intent",
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(stage=stage, name=name)
        self.weights = normalize_weights(weights, _DEFAULT_WEIGHTS)

    def compute(self, clean_state: Any, stress_state: Any) -> MetricResult:
        if clean_state is None or stress_state is None:
            return self._missing("missing_stage_state")

        clean_text = _reasoning_text(clean_state)
        stress_text = _reasoning_text(stress_state)
        text_distance = _token_jaccard_distance(clean_text, stress_text)

        clean_intent = getattr(clean_state, "intent", None)
        stress_intent = getattr(stress_state, "intent", None)
        intent_mismatch = 1.0 if clean_intent != stress_intent else 0.0

        clean_critical = getattr(clean_state, "critical_object_mentioned", None)
        stress_critical = getattr(stress_state, "critical_object_mentioned", None)
        critical_object_mismatch = 1.0 if clean_critical != stress_critical else 0.0

        components = {
            "text_embedding": text_distance,
            "intent_mismatch": intent_mismatch,
            "critical_object_mismatch": critical_object_mismatch,
        }
        raw_score = sum(
            self.weights[key] * components[key]
            for key in _DEFAULT_WEIGHTS
        )
        return MetricResult(
            raw_score=raw_score,
            normalized_score=clip01(raw_score),
            details={
                "components": components,
                "weights": self.weights,
                "clean_intent": clean_intent,
                "stress_intent": stress_intent,
            },
        )


@register_metric("reasoning_intent_only")
class ReasoningIntentOnlyMetric(ReasoningIntentMismatchMetric):
    """Intent-mismatch-only reasoning ablation."""

    def __init__(
        self,
        stage: Stage,
        name: str = "reasoning_intent_only",
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            stage=stage,
            name=name,
            weights={
                "text_embedding": 0.0,
                "intent_mismatch": 1.0,
                "critical_object_mismatch": 0.0,
            },
        )


@register_metric("reasoning_text_only")
class ReasoningTextOnlyMetric(ReasoningIntentMismatchMetric):
    """Token-set lexical-distance-only reasoning ablation."""

    def __init__(
        self,
        stage: Stage,
        name: str = "reasoning_text_only",
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            stage=stage,
            name=name,
            weights={
                "text_embedding": 1.0,
                "intent_mismatch": 0.0,
                "critical_object_mismatch": 0.0,
            },
        )


@register_metric("reasoning_critical_object_only")
class ReasoningCriticalObjectOnlyMetric(ReasoningIntentMismatchMetric):
    """Critical-object-mention-only reasoning ablation."""

    def __init__(
        self,
        stage: Stage,
        name: str = "reasoning_critical_object_only",
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            stage=stage,
            name=name,
            weights={
                "text_embedding": 0.0,
                "intent_mismatch": 0.0,
                "critical_object_mismatch": 1.0,
            },
        )


def _reasoning_text(state: Any) -> str:
    for attr_name in ("text", "decision_text", "explanation"):
        value = getattr(state, attr_name, None)
        if value:
            return str(value)
    return ""


def _token_jaccard_distance(clean_text: str, stress_text: str) -> float:
    clean_tokens = set(_TOKEN_RE.findall(clean_text.lower()))
    stress_tokens = set(_TOKEN_RE.findall(stress_text.lower()))
    union = clean_tokens | stress_tokens
    if not union:
        return 0.0
    return 1.0 - (len(clean_tokens & stress_tokens) / len(union))
