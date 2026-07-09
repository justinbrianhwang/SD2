import pytest

from sd2.core.schema import SemanticState
from sd2.core.stage import Stage
from sd2.metrics import build_metric
from sd2.metrics.semantic import (
    SemanticCompositeMetric,
    SemanticObjectJaccardMetric,
)


def test_semantic_composite_object_only_matches_object_jaccard() -> None:
    clean = SemanticState(objects=["lane", "vehicle"])
    stress = SemanticState(objects=["vehicle", "pedestrian"])
    composite = SemanticCompositeMetric(stage=Stage.SEMANTIC)
    jaccard = SemanticObjectJaccardMetric(stage=Stage.SEMANTIC)

    composite_result = composite.compute(clean, stress)
    jaccard_result = jaccard.compute(clean, stress)

    assert composite_result.missing is False
    assert composite_result.normalized_score == pytest.approx(
        jaccard_result.normalized_score
    )
    assert composite_result.details["components"] == ["object_jaccard"]
    assert composite_result.details["missing_objects"] == ["lane"]
    assert composite_result.details["extra_objects"] == ["pedestrian"]


def test_semantic_composite_seg_only_uses_total_variation_distance() -> None:
    metric = SemanticCompositeMetric(stage=Stage.SEMANTIC)

    result = metric.compute(
        SemanticState(
            bev_seg_summary={
                "class_0": 100,
                "class_1": 0,
                "class_2": 0,
                "nonzero_fraction": 0.1,
                "dominant_class": 0,
            }
        ),
        SemanticState(
            bev_seg_summary={
                "class_0": 50,
                "class_1": 50,
                "class_2": 0,
                "nonzero_fraction": 0.2,
                "dominant_class": 1,
            }
        ),
    )

    assert result.missing is False
    assert result.raw_score == pytest.approx(0.5)
    assert result.normalized_score == pytest.approx(0.5)
    assert result.details["components"] == ["seg_tv"]
    assert result.details["seg_class_distributions"]["clean"] == [1.0, 0.0, 0.0]
    assert result.details["seg_class_distributions"]["stress"] == [0.5, 0.5, 0.0]
    assert result.details["seg_nonzero_fraction"] == {"clean": 0.1, "stress": 0.2}
    assert result.details["seg_dominant_class"] == {"clean": 0, "stress": 1}


def test_semantic_composite_weights_object_and_seg_components() -> None:
    metric = SemanticCompositeMetric(
        stage=Stage.SEMANTIC,
        object_weight=1.0,
        seg_weight=3.0,
    )

    result = metric.compute(
        SemanticState(
            objects=["vehicle"],
            bev_seg_summary={
                "class_0": 100,
                "class_1": 0,
                "class_2": 0,
                "nonzero_fraction": 0.1,
                "dominant_class": 0,
            },
        ),
        SemanticState(
            objects=["pedestrian"],
            bev_seg_summary={
                "class_0": 50,
                "class_1": 50,
                "class_2": 0,
                "nonzero_fraction": 0.2,
                "dominant_class": 1,
            },
        ),
    )

    assert result.details["components"] == ["object_jaccard", "seg_tv"]
    assert result.details["component_scores"]["object_jaccard"] == pytest.approx(1.0)
    assert result.details["component_scores"]["seg_tv"] == pytest.approx(0.5)
    assert result.details["weights"] == {
        "object_jaccard": pytest.approx(0.25),
        "seg_tv": pytest.approx(0.75),
    }
    assert result.normalized_score == pytest.approx(0.625)


def test_semantic_composite_missing_when_no_semantic_signal() -> None:
    metric = SemanticCompositeMetric(stage=Stage.SEMANTIC)

    result = metric.compute(SemanticState(), SemanticState())

    assert result.missing is True
    assert result.details["reason"] == "missing_semantic_signal"


def test_semantic_composite_skips_empty_both_object_lists() -> None:
    metric = SemanticCompositeMetric(stage=Stage.SEMANTIC)

    result = metric.compute(
        SemanticState(
            objects=[],
            bev_seg_summary={
                "class_0": 100,
                "class_1": 0,
                "class_2": 0,
                "nonzero_fraction": 0.1,
                "dominant_class": 0,
            },
        ),
        SemanticState(
            objects=[],
            bev_seg_summary={
                "class_0": 50,
                "class_1": 50,
                "class_2": 0,
                "nonzero_fraction": 0.2,
                "dominant_class": 1,
            },
        ),
    )

    assert result.missing is False
    assert result.details["components"] == ["seg_tv"]
    assert "jaccard_similarity" not in result.details
    assert result.normalized_score == pytest.approx(0.5)


def test_semantic_composite_is_registered_for_build_metric() -> None:
    metric = build_metric(
        Stage.SEMANTIC,
        {
            "type": "semantic_composite",
            "object_weight": 1.0,
            "seg_weight": 1.0,
        },
    )

    assert isinstance(metric, SemanticCompositeMetric)
