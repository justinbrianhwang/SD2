"""Common-stage mean makes cross-model robustness comparable."""

from __future__ import annotations

import pytest

from sd2.analysis.fingerprint import (
    COMMON_STAGES,
    RobustnessFingerprint,
    mean_over_stages,
)
from sd2.core.stage import Stage


def test_common_stages_are_observed_by_every_adapter() -> None:
    assert COMMON_STAGES == (Stage.VISION, Stage.CONTROL)


def test_mean_over_stages_averages_only_the_requested_stages() -> None:
    stage_scores = {
        Stage.VISION: 0.9,
        Stage.SEMANTIC: 0.1,
        Stage.PLANNING: 0.2,
        Stage.CONTROL: 0.7,
    }
    assert mean_over_stages(stage_scores) == pytest.approx(0.8)
    assert mean_over_stages(
        stage_scores, (Stage.VISION, Stage.PLANNING, Stage.CONTROL)
    ) == pytest.approx(0.6)


def test_mean_over_stages_requires_every_requested_stage() -> None:
    # CILRS-like: no semantic head and no waypoints, but vision+control observed.
    cilrs_like = {Stage.VISION: 0.98, Stage.SEMANTIC: None, Stage.PLANNING: None, Stage.CONTROL: 0.96}
    assert mean_over_stages(cilrs_like) == pytest.approx(0.97)
    # Missing a common stage must not silently average a smaller subset.
    assert mean_over_stages({Stage.VISION: 0.9, Stage.CONTROL: None}) is None
    assert mean_over_stages({Stage.VISION: 0.9}) is None
    assert mean_over_stages({}) is None


def test_observed_mean_and_common_mean_differ_for_partially_observed_models() -> None:
    # A model exposing a fragile semantic stage is penalised by the observed mean
    # but is compared fairly against a camera-only model by the common mean.
    rich = RobustnessFingerprint(
        stage_scores={
            Stage.VISION: 0.9,
            Stage.SEMANTIC: 0.5,
            Stage.PLANNING: 0.8,
            Stage.CONTROL: 0.9,
        },
        mean_robustness=0.775,
        common_stage_mean=mean_over_stages(
            {Stage.VISION: 0.9, Stage.SEMANTIC: 0.5, Stage.PLANNING: 0.8, Stage.CONTROL: 0.9}
        ),
    )
    lean = RobustnessFingerprint(
        stage_scores={Stage.VISION: 0.9, Stage.CONTROL: 0.9},
        mean_robustness=0.9,
        common_stage_mean=mean_over_stages({Stage.VISION: 0.9, Stage.CONTROL: 0.9}),
    )

    # Observed means disagree only because the models expose different stages...
    assert rich.mean_robustness != lean.mean_robustness
    # ...while the common-stage mean puts them on the same footing.
    assert rich.common_stage_mean == pytest.approx(lean.common_stage_mean)


def test_to_dict_exposes_common_stage_mean_and_the_stage_list() -> None:
    fingerprint = RobustnessFingerprint(
        stage_scores={Stage.VISION: 0.8, Stage.CONTROL: 0.6},
        mean_robustness=0.7,
        common_stage_mean=0.7,
        run_count=3,
    )
    payload = fingerprint.to_dict()
    assert payload["common_stage_mean"] == pytest.approx(0.7)
    assert payload["common_stages"] == ["vision", "control"]
    assert payload["run_count"] == 3
