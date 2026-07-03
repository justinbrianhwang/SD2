import pytest

from sd2.core.schema import (
    ControlState,
    PlanningState,
    ReasoningState,
    SemanticState,
    VisionState,
)
from sd2.core.stage import Stage
from sd2.metrics import build_metric
from sd2.metrics.control import ControlWeightedMAEMetric
from sd2.metrics.planning import PlanningADEMetric
from sd2.metrics.reasoning import ReasoningIntentMismatchMetric
from sd2.metrics.semantic import SemanticObjectJaccardMetric
from sd2.metrics.vision import VisionEmbeddingCosineMetric


def test_vision_embedding_cosine_identical_embeddings_score_zero() -> None:
    metric = VisionEmbeddingCosineMetric(stage=Stage.VISION)

    result = metric.compute(
        VisionState(embedding=[1.0, 0.0, 0.0]),
        VisionState(embedding=[1.0, 0.0, 0.0]),
    )

    assert result.raw_score == pytest.approx(0.0)
    assert result.normalized_score == pytest.approx(0.0)
    assert result.missing is False


def test_vision_embedding_cosine_handles_zero_vector_and_dim_mismatch() -> None:
    metric = VisionEmbeddingCosineMetric(stage=Stage.VISION)

    zero_result = metric.compute(
        VisionState(embedding=[0.0, 0.0]),
        VisionState(embedding=[1.0, 0.0]),
    )
    mismatch_result = metric.compute(
        VisionState(embedding=[1.0, 0.0]),
        VisionState(embedding=[1.0, 0.0, 0.0]),
    )

    assert zero_result.missing is True
    assert zero_result.details["reason"] == "zero_vector_embedding"
    assert mismatch_result.missing is True
    assert mismatch_result.details["reason"] == "embedding_dimension_mismatch"


def test_semantic_object_jaccard_disjoint_sets_score_one() -> None:
    metric = SemanticObjectJaccardMetric(stage=Stage.SEMANTIC)

    result = metric.compute(
        SemanticState(
            objects=["lane", "vehicle"],
            critical_object="vehicle",
            traffic_light_state="green",
        ),
        SemanticState(
            objects=["pedestrian", "sign"],
            critical_object="pedestrian",
            traffic_light_state="red",
        ),
    )

    assert result.raw_score == pytest.approx(1.0)
    assert result.normalized_score == pytest.approx(1.0)
    assert result.details["missing_objects"] == ["lane", "vehicle"]
    assert result.details["extra_objects"] == ["pedestrian", "sign"]
    assert result.details["critical_object_mismatch"] is True
    assert result.details["traffic_light_mismatch"] is True


def test_reasoning_metric_normalizes_weights_and_scores_intent_flip() -> None:
    metric = ReasoningIntentMismatchMetric(
        stage=Stage.REASONING,
        weights={
            "text_embedding": 2.0,
            "intent_mismatch": 1.0,
            "critical_object_mismatch": 1.0,
        },
    )

    result = metric.compute(
        ReasoningState(
            text="Follow the lane",
            intent="follow_lane",
            critical_object_mentioned=True,
        ),
        ReasoningState(
            text="Follow the lane",
            intent="stop",
            critical_object_mentioned=True,
        ),
    )

    assert sum(metric.weights.values()) == pytest.approx(1.0)
    assert metric.weights["intent_mismatch"] == pytest.approx(0.25)
    assert result.raw_score == pytest.approx(0.25)
    assert result.details["components"]["intent_mismatch"] == pytest.approx(1.0)


def test_planning_ade_truncates_to_common_length_and_normalizes() -> None:
    metric = PlanningADEMetric(stage=Stage.PLANNING, ade_scale=4.0)

    result = metric.compute(
        PlanningState(
            waypoints=[[0.0, 0.0], [1.0, 0.0]],
            target_speed=6.0,
        ),
        PlanningState(
            waypoints=[[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            target_speed=4.5,
        ),
    )

    assert result.raw_score == pytest.approx(1.0)
    assert result.normalized_score == pytest.approx(0.25)
    assert result.details["fde"] == pytest.approx(2.0)
    assert result.details["target_speed_diff"] == pytest.approx(1.5)
    assert result.details["length_mismatch"] is True


def test_control_weighted_mae_normalizes_components_before_weighting() -> None:
    metric = ControlWeightedMAEMetric(
        stage=Stage.CONTROL,
        weights={"steer": 2.0, "throttle": 1.0, "brake": 1.0},
    )

    result = metric.compute(
        ControlState(steer=0.0, throttle=0.2, brake=0.0),
        ControlState(steer=1.0, throttle=0.4, brake=0.4),
    )

    assert result.raw_score == pytest.approx(0.4)
    assert result.normalized_score == pytest.approx(0.4)
    assert result.details["components"]["steer"] == pytest.approx(0.5)
    assert result.details["components"]["throttle"] == pytest.approx(0.2)
    assert result.details["components"]["brake"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("metric", "state"),
    [
        (
            VisionEmbeddingCosineMetric(stage=Stage.VISION),
            VisionState(embedding=[1.0, 0.0]),
        ),
        (
            SemanticObjectJaccardMetric(stage=Stage.SEMANTIC),
            SemanticState(objects=["vehicle"]),
        ),
        (
            ReasoningIntentMismatchMetric(stage=Stage.REASONING),
            ReasoningState(text="go", intent="follow_lane"),
        ),
        (
            PlanningADEMetric(stage=Stage.PLANNING),
            PlanningState(waypoints=[[0.0, 0.0]]),
        ),
        (
            ControlWeightedMAEMetric(stage=Stage.CONTROL),
            ControlState(steer=0.0, throttle=0.0, brake=0.0),
        ),
    ],
)
def test_metrics_flag_missing_stage_state(metric, state) -> None:
    result = metric.compute(None, state)

    assert result.missing is True
    assert result.details["reason"] == "missing_stage_state"


def test_metric_registry_reports_unknown_type_with_available_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_metric(Stage.VISION, {"type": "not_a_metric"})

    message = str(exc_info.value)
    assert "unknown metric type" in message
    assert "embedding_cosine" in message
    assert "weighted_action_mae" in message
