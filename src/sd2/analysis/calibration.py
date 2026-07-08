"""Clean-clean threshold calibration for SD2 deviation status cutoffs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.analysis.deviation import compute_deviation_table
from sd2.analysis.thresholds import ThresholdSet, threshold_set_from_mapping
from sd2.core.config import SD2Config
from sd2.core.run import PairingSummary, PairedRun, RunLog
from sd2.core.schema import PairedFrameLog
from sd2.core.stage import Stage


NEAR_ZERO_STD = 1.0e-12


@dataclass(frozen=True)
class CalibratedStageThresholds:
    """Clean-clean calibration statistics and thresholds for one stage."""

    stage: Stage
    warning: float
    critical: float
    clean_clean_mean: float | None
    clean_clean_std: float | None
    sample_count: int
    fallback_to_config: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning": self.warning,
            "critical": self.critical,
            "clean_clean_mean": self.clean_clean_mean,
            "clean_clean_std": self.clean_clean_std,
            "sample_count": self.sample_count,
            "fallback_to_config": self.fallback_to_config,
        }


@dataclass(frozen=True)
class CalibratedThresholds:
    """Per-stage thresholds calibrated from clean-clean run variance."""

    stages: dict[Stage, CalibratedStageThresholds]
    k_warning: float
    k_critical: float
    fallback_warning: float
    fallback_critical: float
    pair_count: int
    source_run_ids: list[str]
    calibration_type: str = "clean_clean_variance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_type": self.calibration_type,
            "source": self.calibration_type,
            "k_warning": self.k_warning,
            "k_critical": self.k_critical,
            "warning": self.fallback_warning,
            "critical": self.fallback_critical,
            "fallback_thresholds": {
                "warning": self.fallback_warning,
                "critical": self.fallback_critical,
            },
            "pair_count": self.pair_count,
            "source_run_ids": self.source_run_ids,
            "stages": {
                stage.value: threshold.to_dict()
                for stage, threshold in sorted(
                    self.stages.items(),
                    key=lambda item: item[0].index(),
                )
            },
        }

    def to_threshold_set(self) -> ThresholdSet:
        return threshold_set_from_mapping(self.to_dict(), source=self.calibration_type)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output


def calibrate_thresholds(
    clean_runs: Iterable[str | Path | RunLog],
    config: SD2Config,
    k_warning: float = 2.0,
    k_critical: float = 3.0,
) -> CalibratedThresholds:
    """Calibrate per-stage warning/critical thresholds from clean-clean variance.

    All clean-run combinations are paired by shared frame index and evaluated
    with the same configured stage metrics as normal analysis. For each stage,
    the warning threshold is ``mean_cc + k_warning * std_cc`` and the critical
    threshold is ``mean_cc + k_critical * std_cc``, clipped to ``[0, 1]``.
    Stages with no observations or near-zero variance fall back to the static
    config thresholds and record ``fallback_to_config = True``.
    """

    runs = [_load_clean_run(run) for run in clean_runs]
    if len(runs) < 2:
        raise ValueError("calibration requires at least two clean runs")
    if k_warning < 0.0 or k_critical < 0.0:
        raise ValueError("calibration multipliers must be non-negative")
    if k_critical < k_warning:
        raise ValueError("k-critical must be greater than or equal to k-warning")

    fallback = threshold_set_from_mapping(config.thresholds, source="config")
    stage_values: dict[Stage, list[float]] = {stage: [] for stage in _configured_stages(config)}
    pair_count = 0
    for left, right in combinations(runs, 2):
        paired = _pair_clean_runs(left, right)
        if not paired.pairs:
            continue
        pair_count += 1
        table = compute_deviation_table(paired, config, fallback)
        for record in table.records:
            if not record.missing:
                stage_values.setdefault(record.stage, []).append(
                    float(record.normalized_score)
                )

    if pair_count == 0:
        raise ValueError("calibration found no shared frame indices across clean runs")

    calibrated: dict[Stage, CalibratedStageThresholds] = {}
    for stage in _configured_stages(config):
        values = stage_values.get(stage, [])
        stage_fallback = fallback.for_stage(stage)
        if not values:
            calibrated[stage] = CalibratedStageThresholds(
                stage=stage,
                warning=stage_fallback.warning,
                critical=stage_fallback.critical,
                clean_clean_mean=None,
                clean_clean_std=None,
                sample_count=0,
                fallback_to_config=True,
            )
            continue

        mean_cc = mean(values)
        std_cc = pstdev(values)
        fallback_to_config = std_cc <= NEAR_ZERO_STD
        if fallback_to_config:
            warning = stage_fallback.warning
            critical = stage_fallback.critical
        else:
            warning = _clip01(mean_cc + k_warning * std_cc)
            critical = _clip01(mean_cc + k_critical * std_cc)

        calibrated[stage] = CalibratedStageThresholds(
            stage=stage,
            warning=warning,
            critical=critical,
            clean_clean_mean=mean_cc,
            clean_clean_std=std_cc,
            sample_count=len(values),
            fallback_to_config=fallback_to_config,
        )

    return CalibratedThresholds(
        stages=calibrated,
        k_warning=float(k_warning),
        k_critical=float(k_critical),
        fallback_warning=fallback.warning,
        fallback_critical=fallback.critical,
        pair_count=pair_count,
        source_run_ids=[run.metadata.run_id for run in runs],
    )


def format_calibrated_threshold_table(calibrated: CalibratedThresholds) -> str:
    """Format a compact per-stage calibration table for CLI output."""

    lines = [
        "| Stage | Mean CC | Std CC | Warning | Critical | Fallback |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage, thresholds in sorted(
        calibrated.stages.items(),
        key=lambda item: item[0].index(),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    stage.value,
                    _format_optional(thresholds.clean_clean_mean),
                    _format_optional(thresholds.clean_clean_std),
                    f"{thresholds.warning:.6f}",
                    f"{thresholds.critical:.6f}",
                    "yes" if thresholds.fallback_to_config else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _load_clean_run(run: str | Path | RunLog) -> RunLog:
    loaded = run if isinstance(run, RunLog) else load_run_jsonl(run)
    if loaded.metadata.condition.lower() != "clean":
        raise ValueError(f"run {loaded.metadata.run_id!r} is not marked condition=clean")
    if loaded.metadata.severity != 0:
        raise ValueError(f"run {loaded.metadata.run_id!r} is not severity 0")
    return loaded


def _pair_clean_runs(left: RunLog, right: RunLog) -> PairedRun:
    left_by_idx = {frame.frame_idx: frame for frame in left.frames}
    right_by_idx = {frame.frame_idx: frame for frame in right.frames}
    paired_indices = sorted(set(left_by_idx) & set(right_by_idx))
    missing_in_left = sorted(set(right_by_idx) - set(left_by_idx))
    missing_in_right = sorted(set(left_by_idx) - set(right_by_idx))

    pairs = [
        PairedFrameLog(
            pair_key=f"{left.metadata.run_id}:{right.metadata.run_id}:{frame_idx}",
            frame_idx=frame_idx,
            timestamp=left_by_idx[frame_idx].timestamp,
            clean=left_by_idx[frame_idx],
            stress=right_by_idx[frame_idx],
        )
        for frame_idx in paired_indices
    ]
    summary = PairingSummary(
        clean_run_id=left.metadata.run_id,
        stress_run_id=right.metadata.run_id,
        model_id=left.metadata.model_id,
        scenario_id=left.metadata.scenario_id,
        seed=left.metadata.seed,
        clean_frame_count=len(left.frames),
        stress_frame_count=len(right.frames),
        paired_count=len(pairs),
        skipped_count=len(missing_in_left) + len(missing_in_right),
        missing_in_clean=missing_in_left,
        missing_in_stress=missing_in_right,
        clean_metadata=left.metadata.model_dump(mode="json"),
        stress_metadata=right.metadata.model_dump(mode="json"),
    )
    return PairedRun(pairs=pairs, summary=summary)


def _configured_stages(config: SD2Config) -> list[Stage]:
    stages: list[Stage] = []
    for raw_stage in config.stages:
        stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
        if stage != Stage.OUTCOME:
            stages.append(stage)
    return stages


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
