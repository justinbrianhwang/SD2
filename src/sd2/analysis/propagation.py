"""Propagation analysis over a stage-wise deviation table.

Adjacent propagation keeps the legacy MVP ratio:

    Propagation(i -> j, t) = D_j(t + lag) / (D_i(t) + epsilon)

``aggregate_score`` remains the backward-compatible unbounded mean of ratios
for frames whose upstream deviation is at or above ``noise_floor``. Newer
robust evidence is emitted beside it: a capped ratio mean, mean log-ratio,
post-onset absolute downstream increase, collapse ordering, and downstream
persistence. Diagnosis should treat the raw ratio as supporting evidence only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite, log
from pathlib import Path
from statistics import mean
from typing import Any

from sd2.analysis.deviation import DeviationTable
from sd2.analysis.thresholds import (
    ThresholdSet,
    threshold_set_from_config,
    threshold_set_from_mapping,
)
from sd2.core.config import SD2Config
from sd2.core.stage import Stage


@dataclass(frozen=True)
class CollapsePoint:
    """First frame where a stage crosses a configured threshold."""

    frame_idx: int
    timestamp: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "score": self.score,
        }


@dataclass(frozen=True)
class StageCollapseOnset:
    """Warning and critical onset for one stage."""

    stage: Stage
    warning: CollapsePoint | None
    critical: CollapsePoint | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "warning": None if self.warning is None else self.warning.to_dict(),
            "critical": None if self.critical is None else self.critical.to_dict(),
        }


@dataclass(frozen=True)
class PropagationFrameScore:
    """Propagation score for one upstream frame and lagged downstream frame."""

    source_frame_idx: int
    source_timestamp: float
    source_score: float
    downstream_frame_idx: int
    downstream_timestamp: float
    downstream_score: float
    propagation_score: float
    used_in_aggregate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_idx": self.source_frame_idx,
            "source_timestamp": self.source_timestamp,
            "source_score": self.source_score,
            "downstream_frame_idx": self.downstream_frame_idx,
            "downstream_timestamp": self.downstream_timestamp,
            "downstream_score": self.downstream_score,
            "propagation_score": self.propagation_score,
            "used_in_aggregate": self.used_in_aggregate,
        }


@dataclass(frozen=True)
class CollapseOrderEvidence:
    """Relative warning/critical onset order for an adjacent stage pair."""

    upstream_onset_frame: int | None
    downstream_onset_frame: int | None
    downstream_after_upstream: bool | None
    upstream_onset_timestamp: float | None = None
    downstream_onset_timestamp: float | None = None
    upstream_onset_status: str | None = None
    downstream_onset_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_onset_frame": self.upstream_onset_frame,
            "downstream_onset_frame": self.downstream_onset_frame,
            "downstream_after_upstream": self.downstream_after_upstream,
            "upstream_onset_timestamp": self.upstream_onset_timestamp,
            "downstream_onset_timestamp": self.downstream_onset_timestamp,
            "upstream_onset_status": self.upstream_onset_status,
            "downstream_onset_status": self.downstream_onset_status,
        }


@dataclass(frozen=True)
class AdjacentPropagationScore:
    """Propagation summary for adjacent pipeline stages.

    ``aggregate_score`` is the legacy unbounded ratio mean over frames where
    upstream deviation is at or above ``noise_floor``. ``ratio_clipped`` applies
    ``ratio_cap`` before averaging over the same frames. ``log_ratio`` averages
    ``log((D_j + eps) / (D_i + eps))`` over the same frames. ``absolute_increase``
    averages ``D_j - D_i`` over aligned frames at or after the upstream onset.
    ``downstream_persistence`` is the fraction of those post-onset frames where
    the downstream stage remains at or above its warning threshold.
    """

    upstream_stage: Stage
    downstream_stage: Stage
    lag: int
    epsilon: float
    noise_floor: float
    ratio_cap: float
    aggregate_score: float | None
    ratio_clipped: float | None
    log_ratio: float | None
    absolute_increase: float | None
    collapse_order: CollapseOrderEvidence
    downstream_persistence: float | None
    frame_scores: list[PropagationFrameScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_stage": self.upstream_stage.value,
            "downstream_stage": self.downstream_stage.value,
            "lag": self.lag,
            "epsilon": self.epsilon,
            "noise_floor": self.noise_floor,
            "ratio_cap": self.ratio_cap,
            "aggregate_score": self.aggregate_score,
            "ratio_clipped": self.ratio_clipped,
            "log_ratio": self.log_ratio,
            "absolute_increase": self.absolute_increase,
            "collapse_order": self.collapse_order.to_dict(),
            "downstream_persistence": self.downstream_persistence,
            "frame_scores": [score.to_dict() for score in self.frame_scores],
        }


@dataclass(frozen=True)
class DownstreamIncreaseEvidence:
    """Before/after evidence for downstream deviation increase after onset."""

    source_stage: Stage
    downstream_stage: Stage
    onset_status: str
    onset: CollapsePoint
    downstream_window: int
    before_mean: float | None
    after_mean: float | None
    delta: float | None
    increased: bool
    before_frame_indices: list[int]
    after_frame_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stage": self.source_stage.value,
            "downstream_stage": self.downstream_stage.value,
            "onset_status": self.onset_status,
            "onset": self.onset.to_dict(),
            "downstream_window": self.downstream_window,
            "before_mean": self.before_mean,
            "after_mean": self.after_mean,
            "delta": self.delta,
            "increased": self.increased,
            "before_frame_indices": self.before_frame_indices,
            "after_frame_indices": self.after_frame_indices,
        }


@dataclass(frozen=True)
class PropagationResult:
    """Complete propagation analysis output."""

    thresholds: dict[str, Any]
    lag: int
    epsilon: float
    noise_floor: float
    ratio_cap: float
    downstream_window: int
    downstream_min_delta: float
    propagation_scores: list[AdjacentPropagationScore]
    collapse_onsets: list[StageCollapseOnset]
    downstream_increases: list[DownstreamIncreaseEvidence]

    def collapse_by_stage(self) -> dict[Stage, StageCollapseOnset]:
        """Return collapse onsets keyed by stage."""

        return {onset.stage: onset for onset in self.collapse_onsets}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable propagation result."""

        return {
            "thresholds": self.thresholds,
            "lag": self.lag,
            "epsilon": self.epsilon,
            "noise_floor": self.noise_floor,
            "ratio_cap": self.ratio_cap,
            "downstream_window": self.downstream_window,
            "downstream_min_delta": self.downstream_min_delta,
            "propagation_scores": [
                propagation_score.to_dict()
                for propagation_score in self.propagation_scores
            ],
            "collapse_onsets": {
                onset.stage.value: {
                    "warning": None
                    if onset.warning is None
                    else onset.warning.to_dict(),
                    "critical": None
                    if onset.critical is None
                    else onset.critical.to_dict(),
                }
                for onset in self.collapse_onsets
            },
            "downstream_increases": [
                increase.to_dict() for increase in self.downstream_increases
            ],
        }

    def write_json(self, path: str | Path) -> None:
        """Write the propagation result as JSON."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class _StagePoint:
    frame_idx: int
    timestamp: float
    score: float


def compute_propagation_analysis(
    deviation_table: DeviationTable,
    config: SD2Config,
    thresholds: ThresholdSet | dict[str, Any] | None = None,
) -> PropagationResult:
    """Compute propagation scores, collapse onsets, and downstream increases."""

    diagnosis_config = config.diagnosis
    threshold_set = (
        threshold_set_from_config(config)
        if thresholds is None
        else _ensure_threshold_set(thresholds)
    )
    lag = int(diagnosis_config.get("propagation_lag", diagnosis_config.get("lag", 0)))
    epsilon = float(diagnosis_config.get("epsilon", 1.0e-6))
    noise_floor = float(diagnosis_config.get("noise_floor", 0.05))
    ratio_cap = float(diagnosis_config.get("propagation_ratio_cap", 10.0))
    downstream_window = int(diagnosis_config.get("downstream_window", 5))
    downstream_min_delta = float(diagnosis_config.get("downstream_min_delta", 0.0))

    if lag < 0:
        raise ValueError("diagnosis.propagation_lag must be non-negative")
    if epsilon <= 0:
        raise ValueError("diagnosis.epsilon must be positive")
    if ratio_cap <= 0:
        raise ValueError("diagnosis.propagation_ratio_cap must be positive")
    if downstream_window < 1:
        raise ValueError("diagnosis.downstream_window must be at least 1")

    stage_order = _configured_stage_order(config)
    series_by_stage = _stage_series(deviation_table)

    collapse_onsets = [
        _compute_collapse_onset(stage, series_by_stage.get(stage, []), threshold_set)
        for stage in stage_order
    ]
    onset_by_stage = {onset.stage: onset for onset in collapse_onsets}

    propagation_scores = [
        _compute_adjacent_propagation(
            upstream_stage=upstream_stage,
            downstream_stage=downstream_stage,
            upstream_points=series_by_stage.get(upstream_stage, []),
            downstream_points=series_by_stage.get(downstream_stage, []),
            upstream_onset=onset_by_stage.get(upstream_stage),
            downstream_onset=onset_by_stage.get(downstream_stage),
            downstream_warning_threshold=threshold_set.for_stage(
                downstream_stage
            ).warning,
            lag=lag,
            epsilon=epsilon,
            noise_floor=noise_floor,
            ratio_cap=ratio_cap,
        )
        for upstream_stage, downstream_stage in zip(stage_order, stage_order[1:])
    ]

    downstream_increases = _compute_downstream_increases(
        stage_order=stage_order,
        series_by_stage=series_by_stage,
        collapse_onsets=collapse_onsets,
        downstream_window=downstream_window,
        downstream_min_delta=downstream_min_delta,
    )

    return PropagationResult(
        thresholds=threshold_set.to_dict(),
        lag=lag,
        epsilon=epsilon,
        noise_floor=noise_floor,
        ratio_cap=ratio_cap,
        downstream_window=downstream_window,
        downstream_min_delta=downstream_min_delta,
        propagation_scores=propagation_scores,
        collapse_onsets=collapse_onsets,
        downstream_increases=downstream_increases,
    )


def _configured_stage_order(config: SD2Config) -> list[Stage]:
    stages: list[Stage] = []
    for raw_stage in config.stages:
        stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
        if stage != Stage.OUTCOME:
            stages.append(stage)
    return stages


def _ensure_threshold_set(thresholds: ThresholdSet | dict[str, Any]) -> ThresholdSet:
    if isinstance(thresholds, ThresholdSet):
        return thresholds
    return threshold_set_from_mapping(thresholds)


def _stage_series(deviation_table: DeviationTable) -> dict[Stage, list[_StagePoint]]:
    grouped: dict[tuple[Stage, int, float], list[float]] = defaultdict(list)
    for record in deviation_table.records:
        if record.missing or not isfinite(record.normalized_score):
            continue
        grouped[(record.stage, record.frame_idx, record.timestamp)].append(
            float(record.normalized_score)
        )

    series_by_stage: dict[Stage, list[_StagePoint]] = defaultdict(list)
    for (stage, frame_idx, timestamp), scores in grouped.items():
        series_by_stage[stage].append(
            _StagePoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                score=mean(scores),
            )
        )

    return {
        stage: sorted(points, key=lambda point: (point.frame_idx, point.timestamp))
        for stage, points in series_by_stage.items()
    }


def _compute_adjacent_propagation(
    upstream_stage: Stage,
    downstream_stage: Stage,
    upstream_points: list[_StagePoint],
    downstream_points: list[_StagePoint],
    upstream_onset: StageCollapseOnset | None,
    downstream_onset: StageCollapseOnset | None,
    downstream_warning_threshold: float,
    lag: int,
    epsilon: float,
    noise_floor: float,
    ratio_cap: float,
) -> AdjacentPropagationScore:
    downstream_by_frame = {point.frame_idx: point for point in downstream_points}
    frame_scores: list[PropagationFrameScore] = []
    aggregate_values: list[float] = []
    clipped_values: list[float] = []
    log_ratio_values: list[float] = []

    for upstream_point in upstream_points:
        downstream_point = downstream_by_frame.get(upstream_point.frame_idx + lag)
        if downstream_point is None:
            continue
        propagation_score = downstream_point.score / (upstream_point.score + epsilon)
        used_in_aggregate = (
            upstream_point.score >= noise_floor and isfinite(propagation_score)
        )
        if used_in_aggregate:
            aggregate_values.append(propagation_score)
            clipped_values.append(min(propagation_score, ratio_cap))
            log_ratio_values.append(
                log((downstream_point.score + epsilon) / (upstream_point.score + epsilon))
            )
        frame_scores.append(
            PropagationFrameScore(
                source_frame_idx=upstream_point.frame_idx,
                source_timestamp=upstream_point.timestamp,
                source_score=upstream_point.score,
                downstream_frame_idx=downstream_point.frame_idx,
                downstream_timestamp=downstream_point.timestamp,
                downstream_score=downstream_point.score,
                propagation_score=propagation_score,
                used_in_aggregate=used_in_aggregate,
            )
        )

    collapse_order = _collapse_order(upstream_onset, downstream_onset)
    post_onset_scores = _post_upstream_onset_scores(frame_scores, collapse_order)
    absolute_values = [
        score.downstream_score - score.source_score
        for score in post_onset_scores
        if isfinite(score.downstream_score) and isfinite(score.source_score)
    ]
    persistence_values = [
        score.downstream_score >= downstream_warning_threshold
        for score in post_onset_scores
        if isfinite(score.downstream_score)
    ]

    return AdjacentPropagationScore(
        upstream_stage=upstream_stage,
        downstream_stage=downstream_stage,
        lag=lag,
        epsilon=epsilon,
        noise_floor=noise_floor,
        ratio_cap=ratio_cap,
        aggregate_score=None if not aggregate_values else mean(aggregate_values),
        ratio_clipped=None if not clipped_values else mean(clipped_values),
        log_ratio=None if not log_ratio_values else mean(log_ratio_values),
        absolute_increase=None if not absolute_values else mean(absolute_values),
        collapse_order=collapse_order,
        downstream_persistence=None
        if not persistence_values
        else sum(1 for value in persistence_values if value) / len(persistence_values),
        frame_scores=frame_scores,
    )


def _collapse_order(
    upstream_onset: StageCollapseOnset | None,
    downstream_onset: StageCollapseOnset | None,
) -> CollapseOrderEvidence:
    upstream_status, upstream_point = _earliest_available_onset(upstream_onset)
    downstream_status, downstream_point = _earliest_available_onset(downstream_onset)

    downstream_after_upstream = None
    if upstream_point is not None and downstream_point is not None:
        downstream_after_upstream = downstream_point.frame_idx > upstream_point.frame_idx

    return CollapseOrderEvidence(
        upstream_onset_frame=None if upstream_point is None else upstream_point.frame_idx,
        downstream_onset_frame=None
        if downstream_point is None
        else downstream_point.frame_idx,
        downstream_after_upstream=downstream_after_upstream,
        upstream_onset_timestamp=None
        if upstream_point is None
        else upstream_point.timestamp,
        downstream_onset_timestamp=None
        if downstream_point is None
        else downstream_point.timestamp,
        upstream_onset_status=upstream_status,
        downstream_onset_status=downstream_status,
    )


def _earliest_available_onset(
    onset: StageCollapseOnset | None,
) -> tuple[str | None, CollapsePoint | None]:
    if onset is None:
        return None, None
    candidates = [
        (status, point)
        for status, point in (("warning", onset.warning), ("critical", onset.critical))
        if point is not None
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: (item[1].frame_idx, item[1].timestamp))


def _post_upstream_onset_scores(
    frame_scores: list[PropagationFrameScore],
    collapse_order: CollapseOrderEvidence,
) -> list[PropagationFrameScore]:
    if collapse_order.upstream_onset_frame is None:
        return []
    return [
        score
        for score in frame_scores
        if score.source_frame_idx >= collapse_order.upstream_onset_frame
    ]


def _compute_collapse_onset(
    stage: Stage,
    points: list[_StagePoint],
    thresholds: ThresholdSet,
) -> StageCollapseOnset:
    stage_thresholds = thresholds.for_stage(stage)
    return StageCollapseOnset(
        stage=stage,
        warning=_first_onset(points, stage_thresholds.warning),
        critical=_first_onset(points, stage_thresholds.critical),
    )


def _first_onset(
    points: list[_StagePoint],
    threshold: float,
) -> CollapsePoint | None:
    for point in points:
        if point.score >= threshold:
            return CollapsePoint(
                frame_idx=point.frame_idx,
                timestamp=point.timestamp,
                score=point.score,
            )
    return None


def _compute_downstream_increases(
    stage_order: list[Stage],
    series_by_stage: dict[Stage, list[_StagePoint]],
    collapse_onsets: list[StageCollapseOnset],
    downstream_window: int,
    downstream_min_delta: float,
) -> list[DownstreamIncreaseEvidence]:
    onset_by_stage = {onset.stage: onset for onset in collapse_onsets}
    increases: list[DownstreamIncreaseEvidence] = []

    for source_index, source_stage in enumerate(stage_order):
        collapse_onset = onset_by_stage[source_stage]
        for onset_status, onset in (
            ("warning", collapse_onset.warning),
            ("critical", collapse_onset.critical),
        ):
            if onset is None:
                continue
            for downstream_stage in stage_order[source_index + 1 :]:
                downstream_points = series_by_stage.get(downstream_stage, [])
                if not downstream_points:
                    continue
                increases.append(
                    _compute_downstream_increase(
                        source_stage=source_stage,
                        downstream_stage=downstream_stage,
                        onset_status=onset_status,
                        onset=onset,
                        downstream_points=downstream_points,
                        downstream_window=downstream_window,
                        downstream_min_delta=downstream_min_delta,
                    )
                )

    return increases


def _compute_downstream_increase(
    source_stage: Stage,
    downstream_stage: Stage,
    onset_status: str,
    onset: CollapsePoint,
    downstream_points: list[_StagePoint],
    downstream_window: int,
    downstream_min_delta: float,
) -> DownstreamIncreaseEvidence:
    before_points = [
        point
        for point in downstream_points
        if onset.frame_idx - downstream_window <= point.frame_idx < onset.frame_idx
    ]
    after_points = [
        point
        for point in downstream_points
        if onset.frame_idx < point.frame_idx <= onset.frame_idx + downstream_window
    ]

    before_mean = mean([point.score for point in before_points]) if before_points else None
    after_mean = mean([point.score for point in after_points]) if after_points else None
    delta = (
        None
        if before_mean is None or after_mean is None
        else after_mean - before_mean
    )
    increased = delta is not None and delta > downstream_min_delta

    return DownstreamIncreaseEvidence(
        source_stage=source_stage,
        downstream_stage=downstream_stage,
        onset_status=onset_status,
        onset=onset,
        downstream_window=downstream_window,
        before_mean=before_mean,
        after_mean=after_mean,
        delta=delta,
        increased=increased,
        before_frame_indices=[point.frame_idx for point in before_points],
        after_frame_indices=[point.frame_idx for point in after_points],
    )
