import pytest

from experiments.reasoning_paraphrase_probe import evaluate_probe
from sd2.core.schema import ReasoningState
from sd2.core.stage import Stage
from sd2.metrics import available_metric_types, build_metric


def test_reasoning_ablation_metrics_are_registered_and_selectable() -> None:
    registered = set(available_metric_types())

    for metric_type in (
        "reasoning_intent_only",
        "reasoning_text_only",
        "reasoning_critical_object_only",
    ):
        assert metric_type in registered
        metric = build_metric(Stage.REASONING, {"type": metric_type})
        assert metric.name == metric_type


def test_reasoning_intent_only_scores_exact_intent_mismatch() -> None:
    metric = build_metric(Stage.REASONING, {"type": "reasoning_intent_only"})
    clean = ReasoningState(
        text="The lane is clear.",
        intent="follow_lane",
        critical_object_mentioned=True,
    )

    same_intent = metric.compute(
        clean,
        ReasoningState(
            text="The roadway is open.",
            intent="follow_lane",
            critical_object_mentioned=False,
        ),
    )
    different_intent = metric.compute(
        clean,
        ReasoningState(
            text="The lane is clear.",
            intent="stop",
            critical_object_mentioned=True,
        ),
    )

    assert same_intent.normalized_score == pytest.approx(0.0)
    assert different_intent.normalized_score == pytest.approx(1.0)


def test_reasoning_text_only_ignores_intent() -> None:
    metric = build_metric(Stage.REASONING, {"type": "reasoning_text_only"})

    result = metric.compute(
        ReasoningState(
            text="Follow the lane.",
            intent="follow_lane",
            critical_object_mentioned=True,
        ),
        ReasoningState(
            text="Follow the lane.",
            intent="stop",
            critical_object_mentioned=False,
        ),
    )

    assert result.normalized_score == pytest.approx(0.0)


def test_full_reasoning_metric_default_behavior_regression() -> None:
    metric = build_metric(Stage.REASONING, {"type": "text_embedding_and_intent"})

    result = metric.compute(
        ReasoningState(
            text="Follow the lane.",
            intent="follow_lane",
            critical_object_mentioned=True,
        ),
        ReasoningState(
            text="Follow the lane.",
            intent="stop",
            critical_object_mentioned=True,
        ),
    )

    assert result.normalized_score == pytest.approx(0.3)
    assert result.details["weights"] == {
        "text_embedding": pytest.approx(0.5),
        "intent_mismatch": pytest.approx(0.3),
        "critical_object_mismatch": pytest.approx(0.2),
    }


def test_paraphrase_probe_intent_variant_separates_better_than_text_only() -> None:
    rows = {row.metric_type: row for row in evaluate_probe()}

    assert (
        rows["reasoning_intent_only"].separation
        > rows["reasoning_text_only"].separation
    )
