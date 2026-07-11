"""Counterfactual intervention analysis.

This module evaluates same-pose dual-forward intervention recordings. It is
careful not to overclaim controller mediation for single-input controllers:
for AIM, TCP, and TransFuser, control is a deterministic function of planning
and velocity, so restoring planning restores per-tick control by arithmetic.
For those models, the non-trivial evidence is the closed-loop outcome of the
intervened run. Controller-level decomposition is emitted only for InterFuser
and NEAT, whose controllers consume both planning and semantic stage outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.analysis.diagnosis import compute_failure_diagnosis
from sd2.analysis.deviation import compute_deviation_table
from sd2.analysis.event_counting import _count_events
from sd2.analysis.propagation import compute_propagation_analysis
from sd2.analysis.thresholds import resolve_threshold_set
from sd2.core.config import load_config
from sd2.core.run import RunLog, pair_runs
from sd2.core.stage import Stage


MULTI_INPUT_CONTROLLER_MODELS = {"interfuser", "neat"}
SINGLE_INPUT_REASON = (
    "controller consumes planning only; decomposition is analytic, not empirical"
)
CONTROL_EPSILON = 1e-6
MIN_OUTCOME_EFFECT = 0.05
OUTCOME_REPLICATE_K = 1.96
HYBRID_CONTROL_FIELD_BY_STAGE = {
    "planning": "control_hybrid_planning_clean",
    "semantic": "control_hybrid_semantic_clean",
}


@dataclass(frozen=True)
class InterventionAnalysisOutput:
    output_dir: Path
    json_path: Path
    markdown_path: Path


def run_intervention_analysis(
    *,
    baseline_clean: str | Path,
    stress: str | Path,
    intervened: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    clean_replicates: list[str | Path] | None = None,
) -> InterventionAnalysisOutput:
    clean_run = load_run_jsonl(baseline_clean)
    stress_run = load_run_jsonl(stress)
    intervened_run = load_run_jsonl(intervened)
    clean_replicate_runs = [
        load_run_jsonl(path) for path in (clean_replicates or [])
    ]
    config = load_config(config_path)

    _validate_run_compatibility(clean_run, stress_run, intervened_run)
    _validate_clean_replicates(clean_run, clean_replicate_runs)
    intervention = _first_intervention_state(intervened_run)
    outcomes = {
        "clean": _outcome_summary(clean_run),
        "stress": _outcome_summary(stress_run),
        "intervened": _outcome_summary(intervened_run),
    }
    outcome_threshold = _outcome_effect_threshold(clean_replicate_runs)
    recovery = _outcome_recovery(
        outcomes["clean"]["route_completion"],
        outcomes["stress"]["route_completion"],
        outcomes["intervened"]["route_completion"],
        threshold=outcome_threshold["threshold"],
        threshold_source=outcome_threshold["threshold_source"],
    )
    control_decomposition, control_decomposition_reason = _control_decomposition(
        intervened_run,
        model_id=intervened_run.metadata.model_id,
        intervention=intervention,
    )
    diagnosis = _load_or_compute_diagnosis(
        clean_run,
        stress_run,
        config,
        stress_path=Path(stress),
        output_dir=Path(output_dir),
    )
    agreement = _diagnosis_agreement(diagnosis, intervention, recovery)

    result = {
        "model_id": intervened_run.metadata.model_id,
        "scenario_id": intervened_run.metadata.scenario_id,
        "seed": intervened_run.metadata.seed,
        "intervention": intervention,
        "outcomes": outcomes,
        "outcome_recovery": recovery,
        "control_decomposition": control_decomposition,
        "control_decomposition_reason": control_decomposition_reason,
        "diagnosis": diagnosis,
        "agreement_with_sd2": agreement,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "intervention.json"
    markdown_path = output_path / "intervention.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_format_markdown(result), encoding="utf-8")
    return InterventionAnalysisOutput(
        output_dir=output_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def run_single_run_share_analysis(
    *,
    none_run: str | Path,
    output_dir: str | Path,
) -> InterventionAnalysisOutput:
    run = load_run_jsonl(none_run)
    intervention = _first_intervention_state(run)
    stage = str((intervention or {}).get("stage") or "none")
    if stage != "none":
        raise ValueError(
            "single-run shares require a run recorded with --intervene-stage none"
        )

    single_run_shares, reason = _single_run_shares(run)
    result = {
        "model_id": run.metadata.model_id,
        "scenario_id": run.metadata.scenario_id,
        "seed": run.metadata.seed,
        "intervention": intervention,
        "outcomes": {"none_run": _outcome_summary(run)},
        "outcome_recovery": None,
        "control_decomposition": {"single_run_shares": single_run_shares}
        if single_run_shares is not None
        else None,
        "control_decomposition_reason": reason,
        "diagnosis": None,
        "agreement_with_sd2": None,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "intervention.json"
    markdown_path = output_path / "intervention.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_format_single_run_markdown(result), encoding="utf-8")
    return InterventionAnalysisOutput(
        output_dir=output_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def _validate_run_compatibility(clean: RunLog, stress: RunLog, intervened: RunLog) -> None:
    mismatches: list[str] = []
    for field in ("model_id", "scenario_id", "seed"):
        values = {
            "clean": getattr(clean.metadata, field),
            "stress": getattr(stress.metadata, field),
            "intervened": getattr(intervened.metadata, field),
        }
        if len(set(values.values())) != 1:
            mismatches.append(
                f"{field} mismatch "
                f"(clean={values['clean']!r}, stress={values['stress']!r}, "
                f"intervened={values['intervened']!r})"
            )
    if mismatches:
        raise ValueError("cannot compare intervention runs: " + "; ".join(mismatches))


def _validate_clean_replicates(reference: RunLog, replicates: list[RunLog]) -> None:
    mismatches: list[str] = []
    for index, replicate in enumerate(replicates):
        if replicate.metadata.model_id != reference.metadata.model_id:
            mismatches.append(
                f"clean replicate {index} model_id mismatch "
                f"({replicate.metadata.model_id!r} != {reference.metadata.model_id!r})"
            )
        if replicate.metadata.scenario_id != reference.metadata.scenario_id:
            mismatches.append(
                f"clean replicate {index} scenario_id mismatch "
                f"({replicate.metadata.scenario_id!r} != {reference.metadata.scenario_id!r})"
            )
        if replicate.metadata.condition != "clean":
            mismatches.append(
                f"clean replicate {index} condition is {replicate.metadata.condition!r}, expected 'clean'"
            )
    if mismatches:
        raise ValueError("invalid clean replicates: " + "; ".join(mismatches))


def _outcome_summary(run: RunLog) -> dict[str, Any]:
    final_outcome = _state_dict(run.frames[-1], Stage.OUTCOME) if run.frames else {}
    route_completion = _optional_float(final_outcome.get("route_progress"))
    collision_flags = [
        _state_dict(frame, Stage.OUTCOME).get("collision") is True
        for frame in run.frames
    ]
    lane_invasion_flags = [
        _state_dict(frame, Stage.OUTCOME).get("lane_invasion") is True
        for frame in run.frames
    ]
    collision_frames = sum(1 for flag in collision_flags if flag)
    lane_invasion_frames = sum(1 for flag in lane_invasion_flags if flag)
    return {
        "run_id": run.metadata.run_id,
        "condition": run.metadata.condition,
        "route_completion": route_completion,
        "collision_count": _count_events(collision_flags),
        "collision_frames": collision_frames,
        "collision_any": collision_frames > 0,
        "lane_invasion_count": _count_events(lane_invasion_flags),
        "lane_invasion_frames": lane_invasion_frames,
        "lane_invasion_any": lane_invasion_frames > 0,
    }


def _outcome_effect_threshold(replicates: list[RunLog]) -> dict[str, Any]:
    completions = [
        value
        for value in (_outcome_summary(run)["route_completion"] for run in replicates)
        if value is not None and math.isfinite(value)
    ]
    if len(completions) >= 2:
        return {
            "threshold": OUTCOME_REPLICATE_K * stdev(completions),
            "threshold_source": "clean_replicates",
            "clean_replicate_count": len(completions),
        }
    return {
        "threshold": MIN_OUTCOME_EFFECT,
        "threshold_source": "default",
        "clean_replicate_count": len(completions),
    }


def _outcome_recovery(
    clean_completion: Any,
    stress_completion: Any,
    intervened_completion: Any,
    *,
    threshold: float = MIN_OUTCOME_EFFECT,
    threshold_source: str = "default",
) -> dict[str, Any]:
    clean_value = _optional_float(clean_completion)
    stress_value = _optional_float(stress_completion)
    intervened_value = _optional_float(intervened_completion)
    denominator = None
    raw = None
    clipped = None
    null_reason = None
    if clean_value is None or stress_value is None or intervened_value is None:
        null_reason = "missing_outcome"
        if clean_value is not None and stress_value is not None:
            denominator = clean_value - stress_value
    else:
        denominator = clean_value - stress_value
        if denominator <= 0.0:
            null_reason = "stress_did_not_degrade_outcome"
        elif denominator < threshold:
            null_reason = "effect_below_noise_floor"
        else:
            raw = (intervened_value - stress_value) / denominator
            clipped = min(2.0, max(-1.0, raw))
    return {
        "raw": raw,
        "clipped": clipped,
        "denominator": denominator,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "null_reason": null_reason,
    }


def _control_decomposition(
    run: RunLog,
    *,
    model_id: str,
    intervention: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if model_id not in MULTI_INPUT_CONTROLLER_MODELS:
        return None, SINGLE_INPUT_REASON

    stage = str((intervention or {}).get("stage") or "none")
    if stage == "none":
        return None, "no intervened stage was selected"

    total_deltas: list[dict[str, float]] = []
    stage_effects: list[dict[str, float]] = []
    for frame in run.frames:
        item = _state_dict(frame, Stage.INTERVENTION)
        stress_control = _as_mapping(item.get("control_from_stress_forward"))
        clean_control = _as_mapping(item.get("control_from_clean_forward"))
        applied_control = _state_dict(frame, Stage.CONTROL)
        if not stress_control or not clean_control or not applied_control:
            continue
        total_deltas.append(_control_abs_delta(clean_control, stress_control))
        baseline = clean_control if item.get("direction") == "inject" else stress_control
        stage_effects.append(_control_abs_delta(applied_control, baseline))

    if not total_deltas:
        return None, "intervention controls were not present in the run log"

    total = _mean_control_delta(total_deltas)
    effect = _mean_control_delta(stage_effects)
    total_l1 = total["l1"]
    effect_l1 = effect["l1"]
    share = None if total_l1 <= CONTROL_EPSILON else effect_l1 / total_l1
    return {
        "intervened_stage": stage,
        "mean_abs_clean_vs_stress_control": total,
        "mean_abs_stage_intervention_effect": effect,
        "share_of_total_control_change": share,
        "frame_count": len(total_deltas),
    }, None


def _single_run_shares(run: RunLog) -> tuple[dict[str, Any] | None, str | None]:
    total_deltas: list[dict[str, float]] = []
    stage_effects: dict[str, list[dict[str, float]]] = {
        stage: [] for stage in HYBRID_CONTROL_FIELD_BY_STAGE
    }

    for frame in run.frames:
        item = _state_dict(frame, Stage.INTERVENTION)
        stress_control = _as_mapping(item.get("control_from_stress_forward"))
        clean_control = _as_mapping(item.get("control_from_clean_forward"))
        hybrid_controls = {
            stage: _as_mapping(item.get(field))
            for stage, field in HYBRID_CONTROL_FIELD_BY_STAGE.items()
        }
        if stress_control and clean_control and any(hybrid_controls.values()):
            total_deltas.append(_control_abs_delta(clean_control, stress_control))

        if not stress_control:
            continue
        for stage, hybrid_control in hybrid_controls.items():
            if hybrid_control:
                stage_effects[stage].append(
                    _control_abs_delta(hybrid_control, stress_control)
                )

    if not total_deltas:
        return None, "clean/stress forward controls were not present in the run log"

    total = _mean_control_delta(total_deltas)
    total_l1 = total["l1"]
    stages: dict[str, Any] = {}
    for stage, records in stage_effects.items():
        if not records:
            continue
        effect = _mean_control_delta(records)
        stages[stage] = {
            "mean_abs_hybrid_effect": effect,
            "share_of_total_control_change": None
            if total_l1 <= CONTROL_EPSILON
            else effect["l1"] / total_l1,
            "frame_count": len(records),
        }

    if not stages:
        return None, "hybrid candidate controls were not present in the run log"

    return {
        "common_denominator": total,
        "denominator_frame_count": len(total_deltas),
        "stages": stages,
        "non_additivity_note": (
            "shares need not sum to 1 because the controller is non-linear"
        ),
    }, None


def _load_or_compute_diagnosis(
    clean_run: RunLog,
    stress_run: RunLog,
    config: Any,
    *,
    stress_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    diagnosis_path = _find_diagnosis_json(stress_path, output_dir)
    if diagnosis_path is not None:
        payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        return {
            "source": str(diagnosis_path),
            "primary_failure_stage": payload.get("primary_failure_stage"),
            "status": payload.get("status"),
        }

    thresholds = resolve_threshold_set(config, None)
    paired_run = pair_runs(clean_run, stress_run, **_pairing_options(config.pairing))
    deviation_table = compute_deviation_table(paired_run, config, thresholds)
    propagation = compute_propagation_analysis(deviation_table, config, thresholds)
    diagnosis = compute_failure_diagnosis(deviation_table, propagation, paired_run, config)
    payload = diagnosis.to_dict()
    return {
        "source": "computed_from_inputs",
        "primary_failure_stage": payload.get("primary_failure_stage"),
        "status": payload.get("status"),
    }


def _find_diagnosis_json(stress_path: Path, output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "diagnosis.json",
        stress_path.parent / "diagnosis.json",
        stress_path.parent.parent / "diagnosis.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _diagnosis_agreement(
    diagnosis: Mapping[str, Any],
    intervention: dict[str, Any] | None,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    primary_stage = diagnosis.get("primary_failure_stage")
    stage = None if intervention is None else intervention.get("stage")
    direction = None if intervention is None else intervention.get("direction")
    recovered_stage = stage if direction == "restore" and recovery.get("raw") is not None else None
    agrees = None
    if primary_stage is not None and recovered_stage is not None:
        agrees = str(primary_stage) == str(recovered_stage)
    return {
        "primary_failure_stage": primary_stage,
        "best_recovered_stage": recovered_stage,
        "agrees": agrees,
    }


def _first_intervention_state(run: RunLog) -> dict[str, Any] | None:
    for frame in run.frames:
        state = _state_dict(frame, Stage.INTERVENTION)
        if state:
            return {
                "stage": state.get("stage"),
                "direction": state.get("direction"),
                "applied_source": state.get("applied_source"),
                "config": state.get("config"),
            }
    return None


def _state_dict(frame: Any, stage: Stage) -> dict[str, Any]:
    state = frame.states.get(stage)
    if state is None:
        return {}
    if hasattr(state, "model_dump"):
        return state.model_dump(mode="json", exclude_none=True)
    if isinstance(state, Mapping):
        return dict(state)
    return {}


def _control_abs_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    deltas = {
        key: abs((_optional_float(left.get(key)) or 0.0) - (_optional_float(right.get(key)) or 0.0))
        for key in ("steer", "throttle", "brake")
    }
    deltas["l1"] = deltas["steer"] + deltas["throttle"] + deltas["brake"]
    return deltas


def _mean_control_delta(records: list[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: mean(float(record.get(key, 0.0)) for record in records)
        for key in ("steer", "throttle", "brake", "l1")
    }


def _pairing_options(pairing: dict[str, object]) -> dict[str, object]:
    from sd2.core.run import (
        DEFAULT_PAIRING_MODE,
        DEFAULT_PROGRESS_TOLERANCE,
        DEFAULT_TIMESTAMP_TOLERANCE,
    )

    return {
        "mode": pairing.get("mode", DEFAULT_PAIRING_MODE),
        "timestamp_tolerance": pairing.get(
            "timestamp_tolerance",
            DEFAULT_TIMESTAMP_TOLERANCE,
        ),
        "progress_tolerance": pairing.get(
            "progress_tolerance",
            DEFAULT_PROGRESS_TOLERANCE,
        ),
    }


def _format_markdown(result: Mapping[str, Any]) -> str:
    outcomes = _as_mapping(result.get("outcomes"))
    recovery = _as_mapping(result.get("outcome_recovery"))
    agreement = _as_mapping(result.get("agreement_with_sd2"))
    lines = [
        "# Counterfactual Intervention",
        "",
        f"- Model: `{result.get('model_id')}`",
        f"- Scenario: `{result.get('scenario_id')}`",
        f"- Intervention: `{_intervention_label(result.get('intervention'))}`",
        "",
        "## Outcomes",
        "",
        "| Run | route completion | collisions | lane invasions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in ("clean", "stress", "intervened"):
        item = _as_mapping(outcomes.get(label))
        lines.append(
            f"| {label} | {_fmt(item.get('route_completion'))} | "
            f"{item.get('collision_count', 0)} | {item.get('lane_invasion_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Recovery",
            "",
            f"- Raw outcome recovery: {_fmt(recovery.get('raw'))}",
            f"- Clipped outcome recovery: {_fmt(recovery.get('clipped'))}",
            f"- Effect threshold: {_fmt(recovery.get('threshold'))} ({recovery.get('threshold_source')})",
        ]
    )
    if recovery.get("null_reason"):
        lines.append(f"- Null reason: {recovery.get('null_reason')}")
    lines.extend(
        [
            "",
            "## SD2 Agreement",
            "",
            f"- Primary failure stage: `{agreement.get('primary_failure_stage')}`",
            f"- Best recovered stage: `{agreement.get('best_recovered_stage')}`",
            f"- Agrees: `{agreement.get('agrees')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _format_single_run_markdown(result: Mapping[str, Any]) -> str:
    decomposition = _as_mapping(result.get("control_decomposition"))
    shares = _as_mapping(decomposition.get("single_run_shares"))
    lines = [
        "# Counterfactual Intervention",
        "",
        f"- Model: `{result.get('model_id')}`",
        f"- Scenario: `{result.get('scenario_id')}`",
        f"- Intervention: `{_intervention_label(result.get('intervention'))}`",
        "",
        "## Single-Run Shares",
        "",
        "Shares use one clean-vs-stress control denominator from the same non-intervened trajectory.",
        "They need not sum to 1 because the controller is non-linear.",
        "",
    ]
    if not shares:
        lines.append(
            f"- Unavailable: {result.get('control_decomposition_reason') or 'unknown'}"
        )
        return "\n".join(lines)

    common = _as_mapping(shares.get("common_denominator"))
    lines.append(f"- Common denominator L1: {_fmt(common.get('l1'))}")
    stage_records = _as_mapping(shares.get("stages"))
    for stage, record in stage_records.items():
        item = _as_mapping(record)
        lines.append(
            f"- {stage}: share {_fmt(item.get('share_of_total_control_change'))}"
        )
    return "\n".join(lines)


def _intervention_label(value: Any) -> str:
    item = _as_mapping(value)
    if not item:
        return "none"
    return f"{item.get('direction')}:{item.get('stage')}"


def _fmt(value: Any) -> str:
    numeric = _optional_float(value)
    if numeric is None or not math.isfinite(numeric):
        return "null"
    return f"{numeric:.3f}"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MIN_OUTCOME_EFFECT",
    "SINGLE_INPUT_REASON",
    "run_intervention_analysis",
    "run_single_run_share_analysis",
]
