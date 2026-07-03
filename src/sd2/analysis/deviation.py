"""Stage-wise deviation table construction and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

import sd2.metrics  # noqa: F401 - imported to register concrete metrics
from sd2.core.config import SD2Config
from sd2.core.run import PairedRun
from sd2.core.stage import Stage
from sd2.metrics.base import StageMetric, build_metric


HEALTHY = "healthy"
WARNING = "warning"
CRITICAL = "critical"
MISSING = "missing"


@dataclass(frozen=True)
class DeviationRecord:
    """One metric result for one frame and stage."""

    pair_key: str
    frame_idx: int
    timestamp: float
    stage: Stage
    metric: str
    raw_score: float
    normalized_score: float
    status: str
    missing: bool
    details: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "pair_key": self.pair_key,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "stage": self.stage.value,
            "metric": self.metric,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "status": self.status,
            "missing": self.missing,
            "details": self.details,
        }


@dataclass(frozen=True)
class DeviationTable:
    """Collection of stage-wise deviation records."""

    records: list[DeviationRecord]

    def to_records(self) -> list[dict[str, Any]]:
        """Return records suitable for JSON serialization."""

        return [record.to_record() for record in self.records]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame suitable for CSV export."""

        rows = self.to_records()
        for row in rows:
            row["details"] = json.dumps(row["details"], sort_keys=True)
        return pd.DataFrame(rows)

    def write_json(self, path: str | Path) -> None:
        """Write the deviation table as JSON records."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_records(), indent=2) + "\n",
            encoding="utf-8",
        )

    def write_csv(self, path: str | Path) -> None:
        """Write the deviation table as CSV."""

        self.to_dataframe().to_csv(path, index=False)


def compute_deviation_table(paired_run: PairedRun, config: SD2Config) -> DeviationTable:
    """Compute deviations for every paired frame and configured non-outcome stage."""

    metrics = _build_stage_metrics(config)
    records: list[DeviationRecord] = []
    for paired_frame in paired_run.pairs:
        for stage, metric in metrics:
            clean_state = paired_frame.clean.states.get(stage)
            stress_state = paired_frame.stress.states.get(stage)
            result = metric.compute(clean_state, stress_state)
            normalized_score = _apply_normalization(
                result.normalized_score,
                config.normalization,
            )
            status = MISSING if result.missing else classify_status(
                normalized_score,
                config.thresholds,
            )
            records.append(
                DeviationRecord(
                    pair_key=paired_frame.pair_key,
                    frame_idx=paired_frame.frame_idx,
                    timestamp=paired_frame.timestamp,
                    stage=stage,
                    metric=metric.name,
                    raw_score=float(result.raw_score),
                    normalized_score=normalized_score,
                    status=status,
                    missing=result.missing,
                    details=result.details,
                )
            )
    return DeviationTable(records=records)


def classify_status(score: float, thresholds: dict[str, Any]) -> str:
    """Classify a normalized deviation score using warning/critical thresholds."""

    warning = float(thresholds.get("warning", 0.4))
    critical = float(thresholds.get("critical", 0.7))
    if score >= critical:
        return CRITICAL
    if score >= warning:
        return WARNING
    return HEALTHY


def _build_stage_metrics(config: SD2Config) -> list[tuple[Stage, StageMetric]]:
    metrics: list[tuple[Stage, StageMetric]] = []
    for raw_stage in config.stages:
        stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
        if stage == Stage.OUTCOME:
            continue
        metric_config = config.metrics.get(stage.value)
        if metric_config is None:
            continue
        metrics.append((stage, build_metric(stage, metric_config)))
    return metrics


def _apply_normalization(score: float, normalization: dict[str, Any]) -> float:
    method = str(normalization.get("method", "minmax"))
    min_ref = float(normalization.get("min_ref", 0.0))
    max_ref = float(normalization.get("max_ref", 1.0))
    clip_min = float(normalization.get("clip_min", 0.0))
    clip_max = float(normalization.get("clip_max", 1.0))

    normalized = float(score)
    if method == "minmax":
        if max_ref == min_ref:
            raise ValueError("normalization max_ref must differ from min_ref")
        normalized = (normalized - min_ref) / (max_ref - min_ref)
    if not isfinite(normalized):
        normalized = clip_max
    return min(max(normalized, clip_min), clip_max)
