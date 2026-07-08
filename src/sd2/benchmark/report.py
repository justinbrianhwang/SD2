"""Markdown and plot reporting for the synthetic fault benchmark."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sd2.benchmark.runner import NO_FAILURE, PREDICTION_COLUMNS, BenchmarkResult
from sd2.benchmark.synthetic import FAULT_STAGES


@dataclass(frozen=True)
class BenchmarkReportOutput:
    """Paths written by benchmark report generation."""

    report_path: Path
    heatmap_path: Path


def headline_accuracy(result: BenchmarkResult) -> str:
    """Return the benchmark headline accuracy sentence."""

    return f"Primary Failure Stage Diagnosis Accuracy: {result.overall_accuracy * 100:.1f}%"


def generate_benchmark_report(
    result: BenchmarkResult,
    output_dir: str | Path,
    *,
    report_name: str = "benchmark_report.md",
    heatmap_name: str = "confusion_matrix.png",
) -> BenchmarkReportOutput:
    """Write benchmark Markdown report and confusion-matrix heatmap."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    heatmap_path = plot_confusion_matrix(result, root / heatmap_name)
    report_path = root / report_name
    report_path.write_text(
        render_benchmark_markdown(result, report_path, heatmap_path),
        encoding="utf-8",
    )
    return BenchmarkReportOutput(report_path=report_path, heatmap_path=heatmap_path)


def render_benchmark_markdown(
    result: BenchmarkResult,
    report_path: str | Path,
    heatmap_path: str | Path,
) -> str:
    """Render the benchmark Markdown report."""

    report = Path(report_path)
    heatmap = Path(heatmap_path)
    lines = [
        "# SD2 Synthetic Fault Injection Benchmark",
        "",
        f"## {headline_accuracy(result)}",
        "",
        "This benchmark is a framework sanity check: synthetic clean/stress run "
        "pairs are generated with a known primary failure stage, then SD2 scores "
        "only the diagnosis returned by the real offline analysis pipeline.",
        "",
        "## Per-class Accuracy",
        "",
        _per_class_table(result),
        "",
        "## Confusion Matrix",
        "",
        f"![Confusion matrix]({_relative_link(report, heatmap)})",
        "",
        _confusion_table(result),
        "",
        "## Common Confusions",
        "",
        _confusion_analysis(result),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def plot_confusion_matrix(result: BenchmarkResult, output_path: str | Path) -> Path:
    """Save a confusion-matrix heatmap PNG."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [stage.value for stage in FAULT_STAGES]
    columns = _matrix_columns(result)
    matrix = [
        [result.confusion_matrix.get(row, {}).get(column, 0) for column in columns]
        for row in rows
    ]

    fig_width = max(7.0, 0.9 * len(columns) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(columns)), labels=[_display_stage(col) for col in columns])
    ax.set_yticks(range(len(rows)), labels=[_display_stage(row) for row in rows])
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Ground truth stage")
    ax.tick_params(axis="x", labelrotation=35)

    max_value = max([value for row in matrix for value in row] or [0])
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _per_class_table(result: BenchmarkResult) -> str:
    rows = []
    support = Counter(record.target_stage for record in result.records)
    for stage in [stage.value for stage in FAULT_STAGES]:
        rows.append(
            (
                _display_stage(stage),
                f"{result.per_class_accuracy.get(stage, 0.0) * 100:.1f}%",
                str(support.get(stage, 0)),
            )
        )
    return _simple_table(["Class", "Accuracy", "Samples"], rows)


def _confusion_table(result: BenchmarkResult) -> str:
    columns = _matrix_columns(result)
    headers = ["True \\ Predicted", *[_display_stage(column) for column in columns]]
    rows = []
    for stage in [stage.value for stage in FAULT_STAGES]:
        row = [_display_stage(stage)]
        row.extend(str(result.confusion_matrix.get(stage, {}).get(column, 0)) for column in columns)
        rows.append(tuple(row))
    return _simple_table(headers, rows)


def _confusion_analysis(result: BenchmarkResult) -> str:
    misses = Counter()
    for record in result.records:
        if record.correct:
            continue
        predicted = record.predicted_stage or NO_FAILURE
        misses[(record.target_stage, predicted)] += 1

    if not misses:
        return (
            "No off-diagonal confusions were observed. This indicates that the "
            "synthetic injections match the current diagnosis policy assumptions."
        )

    total = len(result.records)
    parts = []
    for (target, predicted), count in misses.most_common(4):
        parts.append(
            f"{_display_stage(target)} -> {_display_stage(predicted)} ({count}/{total})"
        )
    return (
        "Most common misses: "
        + "; ".join(parts)
        + ". These are the classes to inspect first when hardening the diagnosis engine."
    )


def _matrix_columns(result: BenchmarkResult) -> list[str]:
    columns = list(PREDICTION_COLUMNS)
    extras = sorted(
        {
            record.predicted_stage
            for record in result.records
            if record.predicted_stage
            and record.predicted_stage not in columns
        }
    )
    return columns + extras


def _simple_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(_md(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md(value) for value in row) + " |")
    return "\n".join(lines)


def _relative_link(report_path: Path, target_path: Path) -> str:
    relative = os.path.relpath(target_path, start=report_path.parent)
    return relative.replace(os.sep, "/")


def _display_stage(stage: str) -> str:
    if stage == NO_FAILURE:
        return "No Failure"
    return stage.replace("_", " ").title()


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
