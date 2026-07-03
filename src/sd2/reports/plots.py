"""Matplotlib plot generation for SD2 analysis reports."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sd2.core.stage import Stage


logger = logging.getLogger(__name__)


def plot_deviation_timeline(
    deviation_table: Any,
    output_path: str | Path,
    thresholds: dict[str, Any] | None = None,
) -> Path | None:
    """Save a stage-wise deviation timeline PNG and return its path."""

    rows = _load_deviation_rows(deviation_table)
    series = _deviation_series(rows)
    if not series:
        logger.warning("Skipping deviation timeline plot: no deviation records found.")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for stage in _ordered_stage_names(series):
        points = series[stage]
        xs = [point["timestamp"] for point in points]
        ax.plot(
            xs,
            [point["score"] for point in points],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=_display_stage(stage),
        )

    thresholds = thresholds or {}
    _add_threshold_line(ax, thresholds.get("warning"), "warning threshold")
    _add_threshold_line(ax, thresholds.get("critical"), "critical threshold")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized deviation (0-1)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_fingerprint(
    fingerprint: Any,
    output_path: str | Path,
) -> Path | None:
    """Save a robustness fingerprint bar chart PNG and return its path."""

    payload = _load_jsonish(fingerprint)
    stage_scores = payload.get("stage_scores", {}) if isinstance(payload, dict) else {}
    ordered = [
        (stage, _coerce_float(stage_scores.get(stage)))
        for stage in _ordered_stage_names(stage_scores)
    ]
    observed = [(stage, score) for stage, score in ordered if score is not None]
    if not observed:
        logger.warning("Skipping fingerprint plot: no robustness scores found.")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [_display_stage(stage) for stage, _ in observed]
    values = [float(score) for _, score in observed]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(labels[::-1], values[::-1], color="#4e79a7")
    ax.set_xlabel("Robustness score (0-1)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="x", alpha=0.25)
    for index, value in enumerate(values[::-1]):
        ax.text(
            min(value + 0.02, 0.98),
            index,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_propagation(
    propagation_result: Any,
    output_path: str | Path,
) -> Path | None:
    """Save an adjacent-stage propagation bar chart PNG and return its path."""

    payload = _load_jsonish(propagation_result)
    scores = payload.get("propagation_scores", {}) if isinstance(payload, dict) else {}
    if not isinstance(scores, list):
        logger.warning("Skipping propagation plot: malformed propagation scores.")
        return None

    bars: list[tuple[str, float]] = []
    for score in scores:
        if not isinstance(score, dict):
            continue
        upstream = str(score.get("upstream_stage", ""))
        downstream = str(score.get("downstream_stage", ""))
        if not upstream or not downstream or downstream == Stage.OUTCOME.value:
            continue
        aggregate = _coerce_float(score.get("aggregate_score"))
        if aggregate is None:
            continue
        bars.append((f"{_display_stage(upstream)} -> {_display_stage(downstream)}", aggregate))

    if not bars:
        logger.warning("Skipping propagation plot: no aggregate scores found.")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [label for label, _ in bars]
    values = [value for _, value in bars]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, values, color="#59a14f")
    ax.set_ylabel("Aggregate propagation score")
    ax.set_xlabel("Adjacent stage pair")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=25)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _load_deviation_rows(deviation_table: Any) -> list[dict[str, Any]]:
    if deviation_table is None:
        return []
    if hasattr(deviation_table, "to_records"):
        return list(deviation_table.to_records())
    payload = _load_jsonish(deviation_table)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _load_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _deviation_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        if bool(row.get("missing", False)):
            continue
        stage = str(row.get("stage", ""))
        if not stage:
            continue
        score = _coerce_float(row.get("normalized_score"))
        timestamp = _coerce_float(row.get("timestamp"))
        frame_idx = row.get("frame_idx")
        if score is None or timestamp is None or frame_idx is None:
            continue
        grouped[(stage, int(frame_idx), timestamp)].append(score)

    series: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (stage, frame_idx, timestamp), scores in grouped.items():
        series[stage].append(
            {
                "frame_idx": float(frame_idx),
                "timestamp": timestamp,
                "score": mean(scores),
            }
        )
    return {
        stage: sorted(points, key=lambda point: (point["frame_idx"], point["timestamp"]))
        for stage, points in series.items()
        if points
    }


def _ordered_stage_names(data: Any) -> list[str]:
    if isinstance(data, dict):
        present = set(str(stage) for stage in data)
    else:
        present = set()
    ordered = [
        stage.value
        for stage in Stage.ordered()
        if stage != Stage.OUTCOME and stage.value in present
    ]
    return ordered + sorted(present - set(ordered))


def _add_threshold_line(ax: Any, value: Any, label: str) -> None:
    threshold = _coerce_float(value)
    if threshold is None:
        return
    color = "#f28e2b" if "warning" in label else "#e15759"
    ax.axhline(threshold, linestyle="--", linewidth=1.1, color=color, label=label)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return number


def _display_stage(stage: str) -> str:
    return stage.replace("_", " ").title()
