"""Threshold resolution for static and calibrated SD2 status decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sd2.core.config import SD2Config
from sd2.core.stage import Stage


DEFAULT_WARNING = 0.4
DEFAULT_CRITICAL = 0.7


@dataclass(frozen=True)
class StageThresholds:
    """Warning and critical cutoffs for one stage."""

    warning: float
    critical: float

    def to_dict(self) -> dict[str, float]:
        return {
            "warning": self.warning,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class ThresholdSet:
    """Resolved warning/critical thresholds.

    ``warning`` and ``critical`` are the global fallback values used by legacy
    configs and outputs. ``per_stage`` optionally overrides those values for
    individual stages, as produced by clean-clean calibration.
    """

    warning: float = DEFAULT_WARNING
    critical: float = DEFAULT_CRITICAL
    per_stage: dict[Stage, StageThresholds] = field(default_factory=dict)
    source: str = "config"

    def for_stage(self, stage: Stage | str | None) -> StageThresholds:
        """Return stage-specific thresholds or the global fallback."""

        if stage is not None:
            try:
                stage_value = stage if isinstance(stage, Stage) else Stage(str(stage))
            except ValueError:
                stage_value = None
            if stage_value is not None and stage_value in self.per_stage:
                return self.per_stage[stage_value]
        return StageThresholds(warning=self.warning, critical=self.critical)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Static config thresholds serialize to the original two-field mapping.
        Calibrated thresholds add ``source`` and ``per_stage`` while preserving
        the legacy ``warning`` and ``critical`` keys.
        """

        payload: dict[str, Any] = {
            "warning": self.warning,
            "critical": self.critical,
        }
        if self.per_stage:
            payload["source"] = self.source
            payload["per_stage"] = {
                stage.value: thresholds.to_dict()
                for stage, thresholds in sorted(
                    self.per_stage.items(),
                    key=lambda item: item[0].index(),
                )
            }
        return payload


def threshold_set_from_config(config: SD2Config) -> ThresholdSet:
    """Build a static threshold set from ``config.thresholds``."""

    return threshold_set_from_mapping(config.thresholds, source="config")


def threshold_set_from_mapping(
    data: Mapping[str, Any] | None,
    *,
    source: str = "mapping",
) -> ThresholdSet:
    """Build thresholds from a legacy or calibrated mapping."""

    mapping = data or {}
    warning = _coerce_threshold(mapping.get("warning"), DEFAULT_WARNING)
    critical = _coerce_threshold(mapping.get("critical"), DEFAULT_CRITICAL)

    raw_per_stage = mapping.get("per_stage")
    if raw_per_stage is None:
        raw_per_stage = mapping.get("stages")

    per_stage: dict[Stage, StageThresholds] = {}
    if isinstance(raw_per_stage, Mapping):
        for raw_stage, raw_thresholds in raw_per_stage.items():
            if not isinstance(raw_thresholds, Mapping):
                continue
            try:
                stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
            except ValueError:
                continue
            per_stage[stage] = StageThresholds(
                warning=_coerce_threshold(raw_thresholds.get("warning"), warning),
                critical=_coerce_threshold(raw_thresholds.get("critical"), critical),
            )

    return ThresholdSet(
        warning=warning,
        critical=critical,
        per_stage=per_stage,
        source=str(mapping.get("source") or source),
    )


def load_threshold_set(path: str | Path, config: SD2Config) -> ThresholdSet:
    """Load per-stage thresholds from JSON, using config values as fallbacks."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"threshold file must contain a JSON object: {path}")

    fallback = threshold_set_from_config(config)
    merged: dict[str, Any] = {
        "warning": fallback.warning,
        "critical": fallback.critical,
        **dict(payload),
    }
    if "source" not in merged:
        merged["source"] = str(path)
    return threshold_set_from_mapping(merged, source=str(path))


def resolve_threshold_set(
    config: SD2Config,
    thresholds_path: str | Path | None = None,
) -> ThresholdSet:
    """Resolve static config thresholds or an optional calibrated JSON file."""

    if thresholds_path is None:
        return threshold_set_from_config(config)
    return load_threshold_set(thresholds_path, config)


def classify_with_thresholds(
    score: float,
    thresholds: ThresholdSet,
    stage: Stage | str | None = None,
) -> str:
    """Classify a normalized score as healthy, warning, or critical."""

    stage_thresholds = thresholds.for_stage(stage)
    if score >= stage_thresholds.critical:
        return "critical"
    if score >= stage_thresholds.warning:
        return "warning"
    return "healthy"


def _coerce_threshold(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)
