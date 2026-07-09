"""Statistical robustness aggregation across repeated SD2 analysis runs."""

from __future__ import annotations

import builtins
import json
from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

from sd2.analysis.fingerprint import (
    compute_robustness_fingerprint,
    load_deviation_table_json,
)
from sd2.core.config import SD2Config
from sd2.core.stage import Stage


@dataclass(frozen=True)
class StageStat:
    """Summary statistics for one stage over multiple analysis runs."""

    stage: Stage | str
    n: int
    mean: float | None
    std: float | None
    ci95_low: float | None
    ci95_high: float | None
    min: float | None
    max: float | None
    values: list[float]


@dataclass(frozen=True)
class DiagnosisStability:
    """Modal primary-stage stability over repeated analysis runs."""

    primary_stage_counts: dict[str, int]
    modal_stage: str | None
    stability: float | None
    n_runs: int


@dataclass(frozen=True)
class StatisticalRobustnessReport:
    """Multi-run robustness statistics and diagnosis stability."""

    run_count: int
    stage_stats: dict[Stage, StageStat]
    mean_robustness_stat: StageStat
    diagnosis_stability: DiagnosisStability

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "run_count": self.run_count,
            "stage_stats": {
                stage.value: _stage_stat_to_dict(stat)
                for stage, stat in self.stage_stats.items()
            },
            "mean_robustness_stat": _stage_stat_to_dict(
                self.mean_robustness_stat
            ),
            "diagnosis_stability": {
                "primary_stage_counts": dict(
                    self.diagnosis_stability.primary_stage_counts
                ),
                "modal_stage": self.diagnosis_stability.modal_stage,
                "stability": self.diagnosis_stability.stability,
                "n_runs": self.diagnosis_stability.n_runs,
            },
        }

    def write_json(self, path: str | Path) -> None:
        """Write the statistical report as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )


def aggregate_run_statistics(
    analysis_dirs: Sequence[str | Path],
    config: SD2Config | None = None,
) -> StatisticalRobustnessReport:
    """Aggregate stage robustness statistics across analysis output directories."""

    run_dirs = [Path(item) for item in analysis_dirs]
    if not run_dirs:
        raise ValueError("statistical aggregation requires at least one analysis run")

    stage_order = _configured_stage_order(config)
    stage_values: dict[Stage, list[float]] = {stage: [] for stage in stage_order}
    mean_values: list[float] = []
    primary_stages: list[str] = []

    for run_dir in run_dirs:
        deviation_table = load_deviation_table_json(run_dir)
        fingerprint = compute_robustness_fingerprint(deviation_table, config)

        for stage in stage_order:
            score = fingerprint.stage_scores.get(stage)
            if score is not None and isfinite(score):
                stage_values[stage].append(float(score))

        if (
            fingerprint.mean_robustness is not None
            and isfinite(fingerprint.mean_robustness)
        ):
            mean_values.append(float(fingerprint.mean_robustness))

        primary_stage = _load_primary_stage(run_dir)
        if primary_stage is not None:
            primary_stages.append(primary_stage)

    return StatisticalRobustnessReport(
        run_count=len(run_dirs),
        stage_stats={
            stage: _stage_stat(stage, values)
            for stage, values in stage_values.items()
        },
        mean_robustness_stat=_stage_stat("mean_robustness", mean_values),
        diagnosis_stability=_diagnosis_stability(primary_stages, len(run_dirs)),
    )


def discover_analysis_dirs(root: str | Path) -> list[Path]:
    """Return analysis directories directly under root containing deviations."""

    root_path = Path(root)
    if (root_path / "deviation_table.json").is_file():
        return [root_path]
    if not root_path.is_dir():
        raise FileNotFoundError(f"analysis directory not found: {root_path}")

    run_dirs = sorted(
        child
        for child in root_path.iterdir()
        if child.is_dir() and (child / "deviation_table.json").is_file()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"no deviation_table.json files found under {root_path}. "
            "Run `sd2 analyze` first."
        )
    return run_dirs


def format_statistical_report(report: StatisticalRobustnessReport) -> str:
    """Render a multi-run statistical robustness report as Markdown."""

    lines = [
        "# SD2 Statistical Robustness Report",
        "",
        "| Stage | n | mean robustness | std | 95% CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage, stat in report.stage_stats.items():
        if stat.n == 0:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _display_stage(stage.value),
                    str(stat.n),
                    _format_optional_float(stat.mean),
                    _format_optional_float(stat.std),
                    _format_ci(stat),
                ]
            )
            + " |"
        )

    overall = report.mean_robustness_stat
    lines.extend(
        [
            "",
            (
                "Overall mean robustness: "
                f"{_format_optional_float(overall.mean)} +/- "
                f"{_format_optional_float(overall.std)} "
                f"(95% CI {_format_ci(overall)}; n={overall.n})."
            ),
            _diagnosis_stability_line(report.diagnosis_stability),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _stage_stat(stage: Stage | str, values: Sequence[float]) -> StageStat:
    finite_values = [float(value) for value in values if isfinite(float(value))]
    n = len(finite_values)
    if n == 0:
        return StageStat(
            stage=stage,
            n=0,
            mean=None,
            std=None,
            ci95_low=None,
            ci95_high=None,
            min=None,
            max=None,
            values=[],
        )

    mean_value = mean(finite_values)
    std_value = stdev(finite_values) if n >= 2 else None
    if std_value is None:
        ci95_low = None
        ci95_high = None
    else:
        half_width = 1.96 * std_value / sqrt(n)
        ci95_low = mean_value - half_width
        ci95_high = mean_value + half_width

    return StageStat(
        stage=stage,
        n=n,
        mean=mean_value,
        std=std_value,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        min=builtins.min(finite_values),
        max=builtins.max(finite_values),
        values=list(finite_values),
    )


def _diagnosis_stability(
    primary_stages: Sequence[str],
    run_count: int,
) -> DiagnosisStability:
    counts = Counter(primary_stages)
    if not counts:
        return DiagnosisStability(
            primary_stage_counts={},
            modal_stage=None,
            stability=None,
            n_runs=run_count,
        )

    modal_stage, modal_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], _stage_sort_key(item[0])),
    )[0]
    return DiagnosisStability(
        primary_stage_counts=dict(sorted(counts.items(), key=lambda item: item[0])),
        modal_stage=modal_stage,
        stability=modal_count / run_count if run_count else None,
        n_runs=run_count,
    )


def _load_primary_stage(run_dir: Path) -> str | None:
    diagnosis_path = run_dir / "diagnosis.json"
    if not diagnosis_path.is_file():
        return None

    payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    primary = payload.get("primary_failure_stage")
    if primary is None and isinstance(payload.get("diagnosis"), dict):
        primary = payload["diagnosis"].get("primary_failure_stage")
    if primary is None:
        return None

    text = str(primary).strip()
    return text or None


def _configured_stage_order(config: SD2Config | None) -> list[Stage]:
    raw_stages = Stage.ordered() if config is None else config.stages
    stages: list[Stage] = []
    for raw_stage in raw_stages:
        stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
        if stage != Stage.OUTCOME:
            stages.append(stage)
    return stages


def _stage_stat_to_dict(stat: StageStat) -> dict[str, Any]:
    stage = stat.stage.value if isinstance(stat.stage, Stage) else str(stat.stage)
    return {
        "stage": stage,
        "n": stat.n,
        "mean": stat.mean,
        "std": stat.std,
        "ci95_low": stat.ci95_low,
        "ci95_high": stat.ci95_high,
        "min": stat.min,
        "max": stat.max,
        "values": list(stat.values),
    }


def _stage_sort_key(stage: str) -> tuple[int, str]:
    ordered = [item.value for item in Stage.ordered()]
    return (
        ordered.index(stage) if stage in ordered else len(ordered),
        stage,
    )


def _diagnosis_stability_line(stability: DiagnosisStability) -> str:
    modal_count = (
        0
        if stability.modal_stage is None
        else stability.primary_stage_counts.get(stability.modal_stage, 0)
    )
    counts = _format_counts(stability.primary_stage_counts)
    return (
        "Diagnosis stability: primary stage = "
        f"{stability.modal_stage or 'n/a'} in {modal_count}/{stability.n_runs} "
        f"runs (stability {_format_optional_float(stability.stability)}); "
        f"counts: {counts}."
    )


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{stage}={count}" for stage, count in counts.items())


def _format_ci(stat: StageStat) -> str:
    if stat.ci95_low is None or stat.ci95_high is None:
        return "n/a"
    return f"[{stat.ci95_low:.3f}, {stat.ci95_high:.3f}]"


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _display_stage(stage: str) -> str:
    return stage.replace("_", " ").title()
