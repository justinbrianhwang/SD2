"""Temporal-correlational failure diagnosis from deviations and outcomes.

The MVP policy, ``first_critical_with_downstream_increase``, selects the
earliest stage whose critical onset is followed by an increase in any available
downstream stage. Fallbacks are applied in order:

1. earliest warning-stage onset with downstream increase;
2. highest mean-deviation stage when some stage crossed warning/critical but no
   downstream increase supports a candidate;
3. ``primary_failure_stage = None`` with ``status = "no_failure_detected"``
   when every observed stage remains below warning.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

from sd2.analysis.deviation import DeviationTable
from sd2.analysis.event_counting import _event_start_indices
from sd2.analysis.propagation import (
    CollapsePoint,
    DownstreamIncreaseEvidence,
    PropagationResult,
)
from sd2.core.config import SD2Config
from sd2.core.run import PairedRun
from sd2.core.schema import OutcomeState
from sd2.core.stage import Stage


SUPPORTED_POLICY = "first_critical_with_downstream_increase"
DIAGNOSIS_TYPE = "temporal_correlational"


@dataclass(frozen=True)
class DiagnosisResult:
    """Complete failure diagnosis output."""

    diagnosis_type: str
    primary_failure_stage: Stage | None
    status: str
    policy_used: str
    fallback_used: str | None
    collapse_times: dict[str, Any]
    propagation_scores: list[dict[str, Any]]
    final_outcome: dict[str, Any]
    driving_failure: bool
    driving_failure_time: dict[str, Any] | None
    driving_failure_evidence: list[str]
    deviation_precedes_driving_failure: bool | None
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnosis result."""

        return {
            "diagnosis_type": self.diagnosis_type,
            "primary_failure_stage": None
            if self.primary_failure_stage is None
            else self.primary_failure_stage.value,
            "status": self.status,
            "policy_used": self.policy_used,
            "fallback_used": self.fallback_used,
            "collapse_times": self.collapse_times,
            "propagation_scores": self.propagation_scores,
            "final_outcome": self.final_outcome,
            "driving_failure": self.driving_failure,
            "driving_failure_time": self.driving_failure_time,
            "driving_failure_evidence": self.driving_failure_evidence,
            "deviation_precedes_driving_failure": self.deviation_precedes_driving_failure,
            "evidence": self.evidence,
        }

    def write_json(self, path: str | Path) -> None:
        """Write the diagnosis result as JSON."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class _PolicySelection:
    stage: Stage | None
    onset_status: str | None
    onset: CollapsePoint | None
    status: str
    fallback_used: str | None


@dataclass(frozen=True)
class _DrivingFailureAnalysis:
    driving_failure: bool
    driving_failure_time: dict[str, Any] | None
    evidence: list[str]
    final_outcome: dict[str, Any]


def compute_failure_diagnosis(
    deviation_table: DeviationTable,
    propagation_result: PropagationResult,
    paired_run: PairedRun,
    config: SD2Config,
) -> DiagnosisResult:
    """Identify the primary failure stage and supporting evidence."""

    policy = str(
        config.diagnosis.get("primary_failure_policy", SUPPORTED_POLICY)
    )
    if policy != SUPPORTED_POLICY:
        raise ValueError(
            f"unsupported diagnosis.primary_failure_policy {policy!r}; "
            f"expected {SUPPORTED_POLICY!r}"
        )

    stage_means = _mean_deviation_by_stage(deviation_table)
    selection = _select_primary_stage(
        propagation_result=propagation_result,
        stage_means=stage_means,
    )
    driving_analysis = _analyze_driving_failure(paired_run, config)
    deviation_precedes = _deviation_precedes_driving_failure(
        onset=selection.onset,
        driving_failure_time=driving_analysis.driving_failure_time,
    )

    evidence = _build_evidence(
        selection=selection,
        propagation_result=propagation_result,
        stage_means=stage_means,
        driving_analysis=driving_analysis,
        deviation_precedes=deviation_precedes,
    )

    return DiagnosisResult(
        diagnosis_type=DIAGNOSIS_TYPE,
        primary_failure_stage=selection.stage,
        status=selection.status,
        policy_used=policy,
        fallback_used=selection.fallback_used,
        collapse_times=propagation_result.to_dict()["collapse_onsets"],
        propagation_scores=_compact_propagation_scores(propagation_result),
        final_outcome=driving_analysis.final_outcome,
        driving_failure=driving_analysis.driving_failure,
        driving_failure_time=driving_analysis.driving_failure_time,
        driving_failure_evidence=driving_analysis.evidence,
        deviation_precedes_driving_failure=deviation_precedes,
        evidence=evidence,
    )


def _select_primary_stage(
    propagation_result: PropagationResult,
    stage_means: dict[Stage, float],
) -> _PolicySelection:
    critical_candidate = _earliest_stage_with_downstream_increase(
        propagation_result,
        onset_status="critical",
    )
    if critical_candidate is not None:
        stage, onset = critical_candidate
        return _PolicySelection(
            stage=stage,
            onset_status="critical",
            onset=onset,
            status="failure_detected",
            fallback_used=None,
        )

    warning_candidate = _earliest_stage_with_downstream_increase(
        propagation_result,
        onset_status="warning",
    )
    if warning_candidate is not None:
        stage, onset = warning_candidate
        return _PolicySelection(
            stage=stage,
            onset_status="warning",
            onset=onset,
            status="failure_detected",
            fallback_used="earliest_warning_with_downstream_increase",
        )

    if _all_observed_stages_healthy(propagation_result):
        return _PolicySelection(
            stage=None,
            onset_status=None,
            onset=None,
            status="no_failure_detected",
            fallback_used=None,
        )

    if not stage_means:
        return _PolicySelection(
            stage=None,
            onset_status=None,
            onset=None,
            status="no_failure_detected",
            fallback_used=None,
        )

    stage = max(stage_means, key=stage_means.__getitem__)
    onset_status, onset = _best_available_onset(propagation_result, stage)
    return _PolicySelection(
        stage=stage,
        onset_status=onset_status,
        onset=onset,
        status="failure_detected",
        fallback_used="highest_mean_deviation_stage",
    )


def _earliest_stage_with_downstream_increase(
    propagation_result: PropagationResult,
    onset_status: str,
) -> tuple[Stage, CollapsePoint] | None:
    stages_with_increase = {
        evidence.source_stage
        for evidence in propagation_result.downstream_increases
        if evidence.onset_status == onset_status and evidence.increased
    }
    candidates: list[tuple[float, int, Stage, CollapsePoint]] = []
    for stage, collapse in propagation_result.collapse_by_stage().items():
        onset = getattr(collapse, onset_status)
        if onset is not None and stage in stages_with_increase:
            candidates.append((onset.timestamp, onset.frame_idx, stage, onset))

    if not candidates:
        return None

    _, _, stage, onset = min(candidates, key=lambda item: (item[0], item[1]))
    return stage, onset


def _all_observed_stages_healthy(propagation_result: PropagationResult) -> bool:
    for onset in propagation_result.collapse_onsets:
        if onset.warning is not None or onset.critical is not None:
            return False
    return True


def _best_available_onset(
    propagation_result: PropagationResult,
    stage: Stage,
) -> tuple[str | None, CollapsePoint | None]:
    collapse = propagation_result.collapse_by_stage().get(stage)
    if collapse is None:
        return None, None
    if collapse.critical is not None:
        return "critical", collapse.critical
    if collapse.warning is not None:
        return "warning", collapse.warning
    return None, None


def _mean_deviation_by_stage(deviation_table: DeviationTable) -> dict[Stage, float]:
    scores_by_stage: dict[Stage, list[float]] = defaultdict(list)
    for record in deviation_table.records:
        if record.missing or not isfinite(record.normalized_score):
            continue
        scores_by_stage[record.stage].append(float(record.normalized_score))
    return {
        stage: mean(scores)
        for stage, scores in scores_by_stage.items()
        if scores
    }


def _compact_propagation_scores(
    propagation_result: PropagationResult,
) -> list[dict[str, Any]]:
    return [
        {
            "upstream_stage": score.upstream_stage.value,
            "downstream_stage": score.downstream_stage.value,
            "aggregate_score": score.aggregate_score,
            "ratio_clipped": score.ratio_clipped,
            "log_ratio": score.log_ratio,
            "absolute_increase": score.absolute_increase,
            "collapse_order": score.collapse_order.to_dict(),
            "downstream_persistence": score.downstream_persistence,
        }
        for score in propagation_result.propagation_scores
    ]


def _analyze_driving_failure(
    paired_run: PairedRun,
    config: SD2Config,
) -> _DrivingFailureAnalysis:
    route_drop_threshold = float(
        config.diagnosis.get("route_progress_drop_threshold", 0.05)
    )
    evidence: list[str] = []
    failure_events: list[dict[str, Any]] = []

    if not paired_run.pairs:
        return _DrivingFailureAnalysis(
            driving_failure=False,
            driving_failure_time=None,
            evidence=["No paired frames were available for outcome comparison."],
            final_outcome={
                "clean": {},
                "stress": {},
                "route_progress_drop": None,
                "driving_score_drop": None,
            },
        )

    stress_outcomes = [
        _outcome_state(paired_frame.stress.states.get(Stage.OUTCOME))
        for paired_frame in paired_run.pairs
    ]
    collision_flags = [
        outcome.get("collision") is True for outcome in stress_outcomes
    ]
    for event_start in _event_start_indices(collision_flags):
        paired_frame = paired_run.pairs[event_start]
        failure_events.append(
            {
                "type": "collision",
                "frame_idx": paired_frame.frame_idx,
                "timestamp": paired_frame.timestamp,
                "message": (
                    "Collision occurred in stress run at "
                    f"t={_format_time(paired_frame.timestamp)} "
                    f"(frame {paired_frame.frame_idx})."
                ),
            }
        )

    lane_invasion_flags = [
        outcome.get("lane_invasion") is True for outcome in stress_outcomes
    ]
    for event_start in _event_start_indices(lane_invasion_flags):
        paired_frame = paired_run.pairs[event_start]
        failure_events.append(
            {
                "type": "lane_invasion",
                "frame_idx": paired_frame.frame_idx,
                "timestamp": paired_frame.timestamp,
                "message": (
                    "Lane invasion occurred in stress run at "
                    f"t={_format_time(paired_frame.timestamp)} "
                    f"(frame {paired_frame.frame_idx})."
                ),
            }
        )

    final_pair = paired_run.pairs[-1]
    clean_final = _outcome_state(final_pair.clean.states.get(Stage.OUTCOME))
    stress_final = _outcome_state(final_pair.stress.states.get(Stage.OUTCOME))
    route_progress_drop = _numeric_drop(
        clean_final.get("route_progress"),
        stress_final.get("route_progress"),
    )
    driving_score_drop = _numeric_drop(
        clean_final.get("driving_score"),
        stress_final.get("driving_score"),
    )

    if route_progress_drop is not None and route_progress_drop > route_drop_threshold:
        message = (
            "Stress final route progress was "
            f"{stress_final.get('route_progress'):.3f} vs clean "
            f"{clean_final.get('route_progress'):.3f} "
            f"(drop {route_progress_drop:.3f} > {route_drop_threshold:.3f})."
        )
        failure_events.append(
            {
                "type": "route_progress_drop",
                "frame_idx": final_pair.frame_idx,
                "timestamp": final_pair.timestamp,
                "message": message,
            }
        )

    failure_events.sort(key=lambda event: (event["timestamp"], event["frame_idx"]))
    for event in failure_events:
        evidence.append(str(event["message"]))
    if not evidence:
        evidence.append("No collision, lane invasion, or route-progress drop was detected.")

    driving_failure_time = None
    if failure_events:
        first_event = failure_events[0]
        driving_failure_time = {
            "type": first_event["type"],
            "frame_idx": first_event["frame_idx"],
            "timestamp": first_event["timestamp"],
        }

    final_outcome = {
        "clean": clean_final,
        "stress": stress_final,
        "route_progress_drop": route_progress_drop,
        "driving_score_drop": driving_score_drop,
        "route_progress_drop_threshold": route_drop_threshold,
    }

    return _DrivingFailureAnalysis(
        driving_failure=bool(failure_events),
        driving_failure_time=driving_failure_time,
        evidence=evidence,
        final_outcome=final_outcome,
    )


def _outcome_state(raw_state: Any) -> dict[str, Any]:
    if raw_state is None:
        return {}
    if isinstance(raw_state, OutcomeState):
        return raw_state.model_dump(mode="json", exclude_none=True)
    if hasattr(raw_state, "model_dump"):
        return raw_state.model_dump(mode="json", exclude_none=True)
    if isinstance(raw_state, dict):
        return {key: value for key, value in raw_state.items() if value is not None}
    return {}


def _numeric_drop(clean_value: Any, stress_value: Any) -> float | None:
    if clean_value is None or stress_value is None:
        return None
    return float(clean_value) - float(stress_value)


def _deviation_precedes_driving_failure(
    onset: CollapsePoint | None,
    driving_failure_time: dict[str, Any] | None,
) -> bool | None:
    if onset is None or driving_failure_time is None:
        return None
    return onset.timestamp < float(driving_failure_time["timestamp"])


def _build_evidence(
    selection: _PolicySelection,
    propagation_result: PropagationResult,
    stage_means: dict[Stage, float],
    driving_analysis: _DrivingFailureAnalysis,
    deviation_precedes: bool | None,
) -> list[str]:
    evidence: list[str] = []

    if selection.stage is None:
        evidence.append(
            "No stage exceeded warning or critical thresholds; all observed "
            "deviations remained healthy."
        )
    elif selection.onset is not None and selection.onset_status is not None:
        status_text = (
            "earliest critical deviation"
            if selection.onset_status == "critical"
            else "earliest warning deviation"
        )
        evidence.append(
            f"{_display_stage(selection.stage)} showed the {status_text} "
            "with temporal downstream support at "
            f"t={_format_time(selection.onset.timestamp)} "
            f"(frame {selection.onset.frame_idx})."
        )
    else:
        mean_score = stage_means.get(selection.stage)
        evidence.append(
            f"{_display_stage(selection.stage)} was selected by highest mean "
            f"normalized deviation ({_format_optional_score(mean_score)})."
        )

    if selection.stage is not None and selection.onset_status is not None:
        downstream_evidence = [
            item
            for item in propagation_result.downstream_increases
            if item.source_stage == selection.stage
            and item.onset_status == selection.onset_status
            and item.increased
        ]
        for item in downstream_evidence:
            evidence.append(_downstream_evidence_text(item))
        evidence.extend(_adjacent_bundle_evidence_texts(propagation_result, selection.stage))
        if not downstream_evidence and selection.fallback_used is not None:
            evidence.append(
                "No downstream stage had a confirmed temporal before/after "
                "increase for the primary onset."
            )

    if selection.fallback_used == "highest_mean_deviation_stage":
        mean_score = stage_means.get(selection.stage) if selection.stage else None
        evidence.append(
            "No critical or warning stage had downstream temporal support; "
            "selected the highest mean-deviation stage "
            f"({_display_stage(selection.stage)} at "
            f"{_format_optional_score(mean_score)})."
        )

    evidence.extend(driving_analysis.evidence)

    if (
        selection.stage is not None
        and selection.onset is not None
        and driving_analysis.driving_failure_time is not None
    ):
        delta = (
            float(driving_analysis.driving_failure_time["timestamp"])
            - selection.onset.timestamp
        )
        if deviation_precedes:
            evidence.append(
                f"{_display_stage(selection.stage)} {selection.onset_status} "
                f"deviation preceded the first driving failure by {delta:.3f}s."
            )
        else:
            evidence.append(
                f"{_display_stage(selection.stage)} {selection.onset_status} "
                "deviation did not precede the first driving failure."
            )
    elif driving_analysis.driving_failure_time is None:
        evidence.append("No driving failure timestamp was available for temporal ordering.")

    return evidence


def _downstream_evidence_text(item: DownstreamIncreaseEvidence) -> str:
    return (
        f"{_display_stage(item.downstream_stage)} deviation was higher after "
        f"{_display_stage(item.source_stage)} {item.onset_status} onset "
        f"(before mean {_format_optional_score(item.before_mean)}, "
        f"after mean {_format_optional_score(item.after_mean)})."
    )


def _adjacent_bundle_evidence_texts(
    propagation_result: PropagationResult,
    source_stage: Stage,
) -> list[str]:
    texts: list[str] = []
    for score in propagation_result.propagation_scores:
        if score.upstream_stage != source_stage:
            continue
        order = score.collapse_order
        if order.downstream_after_upstream is not True:
            continue
        texts.append(
            f"{_display_stage(score.downstream_stage)} onset followed "
            f"{_display_stage(score.upstream_stage)} onset; mean absolute "
            f"downstream-minus-upstream deviation was "
            f"{_format_optional_score(score.absolute_increase)} and downstream "
            f"persistence was {_format_optional_score(score.downstream_persistence)}."
        )
    return texts


def _display_stage(stage: Stage | None) -> str:
    if stage is None:
        return "Unknown stage"
    return stage.value.capitalize()


def _format_time(timestamp: float) -> str:
    return f"{timestamp:.3f}s"


def _format_optional_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"
