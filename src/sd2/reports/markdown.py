"""Markdown report generation for SD2 analysis artifacts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

from sd2.core.stage import Stage
from sd2.reports.plots import (
    plot_deviation_timeline,
    plot_fingerprint,
    plot_propagation,
)


REQUIRED_ARTIFACTS = (
    "paired_frames.json",
    "pairing_summary.json",
    "deviation_table.json",
    "propagation.json",
    "diagnosis.json",
    "fingerprint.json",
)


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Loaded JSON artifacts for one analysis directory."""

    analysis_dir: Path
    paired_frames: list[dict[str, Any]]
    pairing_summary: dict[str, Any]
    deviation_rows: list[dict[str, Any]]
    propagation: dict[str, Any]
    diagnosis: dict[str, Any]
    fingerprint: dict[str, Any]


@dataclass(frozen=True)
class FingerprintAggregate:
    """Averaged fingerprint for one model/stress grouping."""

    model_id: str
    condition: str
    stress_type: str
    severity: str
    run_count: int
    stage_scores: dict[str, float | None]
    mean_robustness: float | None
    source_dirs: list[Path]


def generate_report(
    analysis_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate plots and a self-contained Markdown report for an analysis run."""

    artifacts = load_analysis_artifacts(analysis_dir)
    output = Path(output_path) if output_path is not None else artifacts.analysis_dir / "report.md"
    plots_dir = artifacts.analysis_dir / "plots"
    thresholds = artifacts.propagation.get("thresholds", {})

    plot_paths = {
        "deviation_timeline": plot_deviation_timeline(
            artifacts.deviation_rows,
            plots_dir / "deviation_timeline.png",
            thresholds,
        ),
        "fingerprint": plot_fingerprint(
            artifacts.fingerprint,
            plots_dir / "robustness_fingerprint.png",
        ),
        "propagation": plot_propagation(
            artifacts.propagation,
            plots_dir / "propagation_scores.png",
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report_markdown(artifacts, output, plot_paths),
        encoding="utf-8",
    )
    return output


def load_analysis_artifacts(analysis_dir: str | Path) -> AnalysisArtifacts:
    """Load required analysis JSON artifacts or raise a clear missing-file error."""

    root = Path(analysis_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"analysis directory {root} is missing required artifacts: {joined}. "
            "Run `sd2 analyze` first."
        )

    return AnalysisArtifacts(
        analysis_dir=root,
        paired_frames=_load_json(root / "paired_frames.json"),
        pairing_summary=_load_json(root / "pairing_summary.json"),
        deviation_rows=_load_json(root / "deviation_table.json"),
        propagation=_load_json(root / "propagation.json"),
        diagnosis=_load_json(root / "diagnosis.json"),
        fingerprint=_load_json(root / "fingerprint.json"),
    )


def render_report_markdown(
    artifacts: AnalysisArtifacts,
    output_path: str | Path,
    plot_paths: dict[str, Path | None],
) -> str:
    """Render a report Markdown string from already-loaded artifacts."""

    output = Path(output_path)
    meta = _run_metadata(artifacts.pairing_summary)
    lines: list[str] = [
        "# SD2 Failure Diagnosis Report",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | {_md(meta['model_id'])} |",
        f"| Scenario | {_md(meta['scenario_id'])} |",
        f"| Condition | {_md(meta['condition'])} |",
        f"| Stress Type | {_md(meta['stress_type'])} |",
        f"| Severity | {_md(meta['severity'])} |",
        f"| Seed | {_md(meta['seed'])} |",
        "",
        "## Summary Diagnosis",
        "",
        build_summary_diagnosis(artifacts),
        "",
        "## Final Outcome Comparison",
        "",
        _final_outcome_table(artifacts.diagnosis),
        "",
        "## Stage-wise Mean Deviation",
        "",
        _stage_mean_table(artifacts.deviation_rows, artifacts.propagation),
        "",
        "## Collapse Onset Times",
        "",
        _collapse_table(artifacts.propagation),
        "",
        "## Propagation Summary",
        "",
        _propagation_table(artifacts.propagation),
        "",
        "## Robustness Fingerprint",
        "",
    ]

    fingerprint_plot = plot_paths.get("fingerprint")
    if fingerprint_plot is not None:
        lines.extend([
            f"![Robustness fingerprint]({_relative_link(output, fingerprint_plot)})",
            "",
        ])
    lines.extend([
        _fingerprint_table(artifacts.fingerprint),
        "",
        "```text",
        _fingerprint_text_bars(artifacts.fingerprint, meta["model_id"]),
        "```",
        "",
        "## Embedded Plots",
        "",
    ])

    for label, key in (
        ("Stage-wise deviation timeline", "deviation_timeline"),
        ("Propagation scores", "propagation"),
    ):
        path = plot_paths.get(key)
        if path is None:
            lines.append(f"- {label}: skipped because no data was available.")
        else:
            lines.extend([
                f"![{label}]({_relative_link(output, path)})",
                "",
            ])

    return "\n".join(lines).rstrip() + "\n"


def build_summary_diagnosis(artifacts: AnalysisArtifacts) -> str:
    """Build the deterministic natural-language diagnosis paragraph."""

    meta = _run_metadata(artifacts.pairing_summary)
    diagnosis = artifacts.diagnosis
    final_outcome = diagnosis.get("final_outcome", {})
    stress_outcome = _dict(final_outcome.get("stress"))
    collapse_times = _dict(
        diagnosis.get("collapse_times") or artifacts.propagation.get("collapse_onsets")
    )
    stage_means = _stage_mean_scores(artifacts.deviation_rows)

    stress_label = _stress_label(meta)
    route_text = _format_progress(stress_outcome.get("route_progress"))
    outcome_sentence = (
        f"Under {stress_label}, the {meta['model_id']} model completed "
        f"{route_text} of the route and {_failure_phrase(stress_outcome)}."
    )

    first_critical = _first_onset(collapse_times, "critical")
    if first_critical is None:
        first_sentence = "No stage crossed the critical deviation threshold."
        propagation_sentence = _propagation_from_primary(
            diagnosis,
            collapse_times,
            artifacts.propagation,
        )
    else:
        stage, point = first_critical
        first_sentence = (
            "The first critical deviation occurred in the "
            f"{_display_stage(stage)} stage at t={_format_time(point.get('timestamp'))} "
            f"(frame {_format_value(point.get('frame_idx'))})."
        )
        propagation_sentence = _propagation_order_sentence(
            stage,
            point,
            collapse_times,
            artifacts.propagation,
            "critical",
        )

    primary = diagnosis.get("primary_failure_stage")
    if primary:
        primary_sentence = (
            "The primary failure stage is diagnosed as "
            f"{_display_stage(str(primary))}."
        )
    else:
        primary_sentence = "No primary failure stage was diagnosed."

    interpretation = _interpretation_sentence(
        primary=str(primary) if primary else None,
        collapse_times=collapse_times,
        stage_means=stage_means,
    )

    return " ".join(
        [
            outcome_sentence,
            first_sentence,
            propagation_sentence,
            primary_sentence,
            interpretation,
        ]
    )


def generate_fingerprint_summary(
    analysis_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Aggregate fingerprint.json files and write a comparison Markdown table."""

    aggregates = aggregate_fingerprint_files(analysis_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_fingerprint_summary_markdown(aggregates), encoding="utf-8")
    return output


def aggregate_fingerprint_files(analysis_dir: str | Path) -> list[FingerprintAggregate]:
    """Scan for fingerprint.json files and aggregate by model/stress condition."""

    root = Path(analysis_dir)
    paths = sorted(root.rglob("fingerprint.json")) if root.is_dir() else []
    if not paths:
        raise FileNotFoundError(
            f"no fingerprint.json files found under {root}. Run `sd2 analyze` first."
        )

    grouped: dict[tuple[str, str, str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        fingerprint = _load_json(path)
        meta = _metadata_for_fingerprint(path.parent, fingerprint)
        key = (
            meta["model_id"],
            meta["condition"],
            meta["stress_type"],
            meta["severity"],
        )
        grouped[key].append((path.parent, fingerprint))

    aggregates: list[FingerprintAggregate] = []
    for (model_id, condition, stress_type, severity), items in sorted(grouped.items()):
        stage_scores = _average_stage_scores([fingerprint for _, fingerprint in items])
        observed = [score for score in stage_scores.values() if score is not None]
        mean_score = None if not observed else mean(observed)
        aggregates.append(
            FingerprintAggregate(
                model_id=model_id,
                condition=condition,
                stress_type=stress_type,
                severity=severity,
                run_count=len(items),
                stage_scores=stage_scores,
                mean_robustness=mean_score,
                source_dirs=[path for path, _ in items],
            )
        )
    return aggregates


def _fingerprint_summary_markdown(aggregates: list[FingerprintAggregate]) -> str:
    stage_names = _stage_names_from_fingerprints(
        [{"stage_scores": aggregate.stage_scores} for aggregate in aggregates]
    )
    headers = [
        "Model",
        "Condition",
        "Stress Type",
        "Severity",
        "Runs",
        *[_display_stage(stage) for stage in stage_names],
        "Mean",
    ]
    lines = [
        "# SD2 Fingerprint Summary",
        "",
        _table_header(headers),
        _table_separator(len(headers)),
    ]
    for aggregate in aggregates:
        row = [
            aggregate.model_id,
            aggregate.condition,
            aggregate.stress_type,
            aggregate.severity,
            str(aggregate.run_count),
            *[_format_optional_float(aggregate.stage_scores.get(stage)) for stage in stage_names],
            _format_optional_float(aggregate.mean_robustness),
        ]
        lines.append(_table_row(row))
    return "\n".join(lines) + "\n"


def _final_outcome_table(diagnosis: dict[str, Any]) -> str:
    final_outcome = _dict(diagnosis.get("final_outcome"))
    clean = _dict(final_outcome.get("clean"))
    stress = _dict(final_outcome.get("stress"))
    route_drop = final_outcome.get("route_progress_drop")
    rows = [
        ("Collision", _format_bool(clean.get("collision")), _format_bool(stress.get("collision")), "n/a"),
        (
            "Lane invasion",
            _format_bool(clean.get("lane_invasion")),
            _format_bool(stress.get("lane_invasion")),
            "n/a",
        ),
        (
            "Route progress",
            _format_progress(clean.get("route_progress")),
            _format_progress(stress.get("route_progress")),
            _format_optional_progress_delta(route_drop),
        ),
    ]
    return _simple_table(["Metric", "Clean", "Stress", "Delta"], rows)


def _stage_mean_table(
    deviation_rows: list[dict[str, Any]],
    propagation: dict[str, Any],
) -> str:
    thresholds = _dict(propagation.get("thresholds"))
    stage_means = _stage_mean_scores(deviation_rows)
    stage_max = _stage_max_scores(deviation_rows)
    stage_counts = _stage_counts(deviation_rows)
    rows = []
    for stage in _ordered_stage_names(stage_means):
        mean_score = stage_means.get(stage)
        rows.append(
            (
                _display_stage(stage),
                _format_optional_float(mean_score),
                _format_optional_float(stage_max.get(stage)),
                _status_from_score(mean_score, thresholds),
                str(stage_counts.get(stage, 0)),
            )
        )
    return _simple_table(["Stage", "Mean", "Max", "Status", "Samples"], rows)


def _collapse_table(propagation: dict[str, Any]) -> str:
    collapse_times = _dict(propagation.get("collapse_onsets"))
    rows = [
        (
            _display_stage(stage),
            _point_text(_dict(points).get("warning")),
            _point_text(_dict(points).get("critical")),
        )
        for stage, points in _ordered_stage_items(collapse_times)
    ]
    return _simple_table(["Stage", "Warning Onset", "Critical Onset"], rows)


def _propagation_table(propagation: dict[str, Any]) -> str:
    scores = propagation.get("propagation_scores", [])
    rows = []
    if isinstance(scores, list):
        for item in scores:
            if not isinstance(item, dict):
                continue
            downstream = str(item.get("downstream_stage", ""))
            if downstream == Stage.OUTCOME.value:
                continue
            edge = (
                f"{_display_stage(str(item.get('upstream_stage', '')))} -> "
                f"{_display_stage(downstream)}"
            )
            rows.append(
                (
                    edge,
                    _format_optional_float(item.get("aggregate_score")),
                    str(item.get("lag", propagation.get("lag", "n/a"))),
                )
            )
    return _simple_table(["Edge", "Aggregate Score", "Lag"], rows)


def _fingerprint_table(fingerprint: dict[str, Any]) -> str:
    stage_scores = _dict(fingerprint.get("stage_scores"))
    rows = [
        (_display_stage(stage), _format_optional_float(stage_scores.get(stage)))
        for stage in _ordered_stage_names(stage_scores)
    ]
    rows.append(("Mean", _format_optional_float(fingerprint.get("mean_robustness"))))
    rows.append(("Run count", str(fingerprint.get("run_count", "n/a"))))
    return _simple_table(["Stage", "Robustness"], rows)


def _fingerprint_text_bars(fingerprint: dict[str, Any], model_id: str) -> str:
    stage_scores = _dict(fingerprint.get("stage_scores"))
    lines = [f"{model_id} Robustness Fingerprint", ""]
    for stage in _ordered_stage_names(stage_scores):
        value = _coerce_float(stage_scores.get(stage))
        if value is None:
            bar = "[??????????]"
            score = "n/a"
        else:
            filled = round(value * 10)
            bar = "[" + ("#" * filled) + ("-" * (10 - filled)) + "]"
            score = f"{value:.2f}"
        lines.append(f"{_display_stage(stage) + ':':<12} {bar} {score}")
    return "\n".join(lines)


def _metadata_for_fingerprint(run_dir: Path, fingerprint: dict[str, Any]) -> dict[str, str]:
    summary_path = run_dir / "pairing_summary.json"
    if summary_path.is_file():
        return _run_metadata(_load_json(summary_path))
    fingerprint_metadata = _dict(
        fingerprint.get("metadata")
        or fingerprint.get("run_metadata")
        or fingerprint.get("stress_metadata")
    )
    return {
        "model_id": _string(_first_present(fingerprint_metadata.get("model_id"), "unknown")),
        "scenario_id": _string(
            _first_present(fingerprint_metadata.get("scenario_id"), "unknown")
        ),
        "condition": _string(
            _first_present(fingerprint_metadata.get("condition"), "unknown")
        ),
        "stress_type": _string(
            _first_present(fingerprint_metadata.get("stress_type"), "unknown")
        ),
        "severity": _string(_first_present(fingerprint_metadata.get("severity"), "n/a")),
        "seed": _string(_first_present(fingerprint_metadata.get("seed"), "n/a")),
    }


def _run_metadata(summary: dict[str, Any]) -> dict[str, str]:
    clean_metadata = _dict(summary.get("clean_metadata"))
    stress_metadata = _dict(summary.get("stress_metadata"))
    return {
        "model_id": _string(
            _first_present(
                stress_metadata.get("model_id"),
                clean_metadata.get("model_id"),
                summary.get("model_id"),
            )
        ),
        "scenario_id": _string(
            _first_present(
                stress_metadata.get("scenario_id"),
                clean_metadata.get("scenario_id"),
                summary.get("scenario_id"),
            )
        ),
        "condition": _string(_first_present(stress_metadata.get("condition"), "stress")),
        "stress_type": _string(_first_present(stress_metadata.get("stress_type"), "none")),
        "severity": _string(_first_present(stress_metadata.get("severity"), "n/a")),
        "seed": _string(_first_present(stress_metadata.get("seed"), summary.get("seed"))),
    }


def _stress_label(meta: dict[str, str]) -> str:
    stress_type = meta.get("stress_type") or "stress"
    label = stress_type.replace("_", " ").title()
    severity = meta.get("severity")
    if severity and severity != "n/a":
        return f"{label} severity {severity}"
    return label


def _failure_phrase(stress_outcome: dict[str, Any]) -> str:
    failures = []
    if stress_outcome.get("collision") is True:
        failures.append("a collision")
    if stress_outcome.get("lane_invasion") is True:
        failures.append("a lane invasion")
    if not failures:
        return "did not record a collision or lane invasion"
    if len(failures) == 1:
        return f"experienced {failures[0]}"
    return f"experienced {failures[0]} and {failures[1]}"


def _propagation_order_sentence(
    first_stage: str,
    first_point: dict[str, Any],
    collapse_times: dict[str, Any],
    propagation: dict[str, Any],
    onset_status: str,
) -> str:
    increase_sentence = _downstream_increase_sentence(
        first_stage,
        onset_status,
        propagation,
    )
    if increase_sentence is not None:
        return increase_sentence

    first_time = _coerce_float(first_point.get("timestamp"))
    first_index = _stage_index(first_stage)
    downstream: list[tuple[float, str, dict[str, Any], str]] = []
    for stage, points in _ordered_stage_items(collapse_times):
        if _stage_index(stage) <= first_index:
            continue
        point, status = _best_onset_after(points, first_time)
        if point is not None:
            timestamp = _coerce_float(point.get("timestamp"))
            if timestamp is not None:
                downstream.append((timestamp, stage, point, status))
    if not downstream:
        return "No downstream stage crossed warning or critical thresholds after that onset."
    downstream.sort(key=lambda item: (item[0], _stage_index(item[1])))
    parts = [
        f"{_display_stage(stage)} {status} at t={_format_time(point.get('timestamp'))}"
        for _, stage, point, status in downstream
    ]
    return "Propagation was observed downstream in the order " + _join_words(parts) + "."


def _propagation_from_primary(
    diagnosis: dict[str, Any],
    collapse_times: dict[str, Any],
    propagation: dict[str, Any],
) -> str:
    primary = diagnosis.get("primary_failure_stage")
    if not primary:
        return "No downstream propagation order was detected."
    point_map = _dict(collapse_times.get(str(primary), {}))
    status = "critical" if point_map.get("critical") is not None else "warning"
    point = point_map.get(status)
    if point is None:
        return "No downstream propagation order was detected."
    return _propagation_order_sentence(
        str(primary),
        _dict(point),
        collapse_times,
        propagation,
        status,
    )


def _downstream_increase_sentence(
    source_stage: str,
    onset_status: str,
    propagation: dict[str, Any],
) -> str | None:
    increases = []
    raw_increases = propagation.get("downstream_increases", [])
    if not isinstance(raw_increases, list):
        return None
    for item in raw_increases:
        if not isinstance(item, dict):
            continue
        if item.get("source_stage") != source_stage:
            continue
        if item.get("onset_status") != onset_status:
            continue
        if item.get("increased") is not True:
            continue
        downstream = str(item.get("downstream_stage", ""))
        if downstream:
            delta = _coerce_float(item.get("delta"))
            increases.append((_stage_index(downstream), downstream, delta))
    if not increases:
        return None
    increases.sort(key=lambda item: item[0])
    parts = [
        f"{_display_stage(stage)} (+{delta:.3f})"
        if delta is not None
        else _display_stage(stage)
        for _, stage, delta in increases
    ]
    return (
        "Propagation evidence shows downstream increases in the order "
        f"{_join_words(parts)} after the {_display_stage(source_stage)} onset."
    )


def _interpretation_sentence(
    primary: str | None,
    collapse_times: dict[str, Any],
    stage_means: dict[str, float],
) -> str:
    if primary is None:
        return "This suggests that the configured thresholds did not identify a pipeline collapse."

    downstream = [
        stage
        for stage in _ordered_stage_names(collapse_times)
        if _stage_index(stage) > _stage_index(primary)
        and _dict(collapse_times.get(stage)).get("warning") is not None
    ]
    upstream_critical = [
        stage
        for stage in _ordered_stage_names(collapse_times)
        if _stage_index(stage) < _stage_index(primary)
        and _dict(collapse_times.get(stage)).get("critical") is not None
    ]
    if primary == Stage.REASONING.value and not upstream_critical:
        return (
            "This suggests that upstream perception remained comparatively stable, "
            "but semantic or intent changes were amplified during reasoning and "
            "propagated into planning and control."
        )
    if downstream:
        return (
            f"This suggests that the stress became operationally important at "
            f"{_display_stage(primary)} and then propagated into "
            f"{_join_words([_display_stage(stage) for stage in downstream])}."
        )
    mean_score = stage_means.get(primary)
    return (
        f"This suggests that {_display_stage(primary)} dominated the observed "
        f"deviation profile with mean deviation {_format_optional_float(mean_score)}."
    )


def _first_onset(
    collapse_times: dict[str, Any],
    status: str,
) -> tuple[str, dict[str, Any]] | None:
    candidates = []
    for stage, points in _ordered_stage_items(collapse_times):
        point = _dict(points).get(status)
        if not isinstance(point, dict):
            continue
        timestamp = _coerce_float(point.get("timestamp"))
        frame_idx = _coerce_float(point.get("frame_idx"))
        if timestamp is None:
            continue
        candidates.append((timestamp, frame_idx or 0.0, stage, point))
    if not candidates:
        return None
    _, _, stage, point = min(candidates, key=lambda item: (item[0], item[1], _stage_index(item[2])))
    return stage, point


def _best_onset_after(
    points: Any,
    first_time: float | None,
) -> tuple[dict[str, Any] | None, str]:
    point_map = _dict(points)
    candidates = []
    for status in ("critical", "warning"):
        point = point_map.get(status)
        if not isinstance(point, dict):
            continue
        timestamp = _coerce_float(point.get("timestamp"))
        if timestamp is None:
            continue
        if first_time is None or timestamp >= first_time:
            candidates.append((timestamp, status, point))
    if not candidates:
        return None, ""
    _, status, point = min(candidates, key=lambda item: (item[0], item[1]))
    return point, status


def _average_stage_scores(fingerprints: list[dict[str, Any]]) -> dict[str, float | None]:
    stage_names = _stage_names_from_fingerprints(fingerprints)
    averaged: dict[str, float | None] = {}
    for stage in stage_names:
        values = [
            value
            for value in (
                _coerce_float(_dict(fingerprint.get("stage_scores")).get(stage))
                for fingerprint in fingerprints
            )
            if value is not None
        ]
        averaged[stage] = None if not values else mean(values)
    return averaged


def _stage_names_from_fingerprints(fingerprints: list[dict[str, Any]]) -> list[str]:
    present: set[str] = set()
    for fingerprint in fingerprints:
        present.update(_dict(fingerprint.get("stage_scores")).keys())
    return _ordered_stage_names({stage: None for stage in present})


def _stage_mean_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = _scores_by_stage(rows)
    return {stage: mean(values) for stage, values in scores.items() if values}


def _stage_max_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = _scores_by_stage(rows)
    return {stage: max(values) for stage, values in scores.items() if values}


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    scores = _scores_by_stage(rows)
    return {stage: len(values) for stage, values in scores.items()}


def _scores_by_stage(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if bool(row.get("missing", False)):
            continue
        stage = str(row.get("stage", ""))
        score = _coerce_float(row.get("normalized_score"))
        if stage and score is not None:
            scores[stage].append(score)
    return scores


def _status_from_score(score: float | None, thresholds: dict[str, Any]) -> str:
    if score is None:
        return "missing"
    warning = _coerce_float(thresholds.get("warning")) or 0.4
    critical = _coerce_float(thresholds.get("critical")) or 0.7
    if score >= critical:
        return "critical"
    if score >= warning:
        return "warning"
    return "healthy"


def _ordered_stage_items(data: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(stage, data.get(stage)) for stage in _ordered_stage_names(data)]


def _ordered_stage_names(data: dict[str, Any]) -> list[str]:
    present = set(str(stage) for stage in data)
    ordered = [
        stage.value
        for stage in Stage.ordered()
        if stage != Stage.OUTCOME and stage.value in present
    ]
    return ordered + sorted(present - set(ordered))


def _stage_index(stage: str) -> int:
    ordered = [item.value for item in Stage.ordered()]
    return ordered.index(stage) if stage in ordered else len(ordered)


def _point_text(point: Any) -> str:
    if not isinstance(point, dict):
        return "n/a"
    return (
        f"t={_format_time(point.get('timestamp'))}, "
        f"frame {_format_value(point.get('frame_idx'))}, "
        f"score {_format_optional_float(point.get('score'))}"
    )


def _simple_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    lines = [_table_header(headers), _table_separator(len(headers))]
    if not rows:
        rows = [tuple("n/a" for _ in headers)]
    for row in rows:
        lines.append(_table_row([_format_value(item) for item in row]))
    return "\n".join(lines)


def _table_header(headers: list[str]) -> str:
    return _table_row(headers)


def _table_separator(count: int) -> str:
    return "| " + " | ".join("---" for _ in range(count)) + " |"


def _table_row(values: list[Any]) -> str:
    return "| " + " | ".join(_md(_format_value(value)) for value in values) + " |"


def _relative_link(report_path: Path, target_path: Path) -> str:
    relative = os.path.relpath(target_path, start=report_path.parent)
    return relative.replace(os.sep, "/")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _format_time(value: Any) -> str:
    number = _coerce_float(value)
    return "n/a" if number is None else f"{number:.3f}s"


def _format_optional_float(value: Any) -> str:
    number = _coerce_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def _format_optional_progress_delta(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.1f} pp"


def _format_progress(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    if 0.0 <= number <= 1.5:
        return f"{number * 100:.1f}%"
    return f"{number:.3f}"


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _display_stage(stage: str) -> str:
    return stage.replace("_", " ").title()


def _string(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _md(value: Any) -> str:
    text = _format_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _join_words(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
