"""Robustness fingerprint analysis for deviation tables."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from sd2.analysis.deviation import DeviationRecord, DeviationTable
from sd2.core.config import SD2Config
from sd2.core.stage import Stage


@dataclass(frozen=True)
class RobustnessFingerprint:
    """Stage-wise robustness scores where higher means more robust."""

    stage_scores: dict[Stage, float | None]
    mean_robustness: float | None
    run_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fingerprint."""

        return {
            "stage_scores": {
                stage.value: score for stage, score in self.stage_scores.items()
            },
            "mean_robustness": self.mean_robustness,
            "run_count": self.run_count,
        }

    def write_json(self, path: str | Path) -> None:
        """Write the fingerprint as JSON."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )


DeviationTableInput = DeviationTable | str | Path


def compute_robustness_fingerprint(
    deviation_table: DeviationTable,
    config: SD2Config | None = None,
) -> RobustnessFingerprint:
    """Compute one run-pair robustness fingerprint.

    Each stage score is ``1 - mean(normalized_deviation)`` clipped to ``[0, 1]``.
    Stages with no non-missing deviation records are reported as ``None`` rather
    than zero so lower-observability logs do not look non-robust.
    """

    stage_order = _configured_stage_order(config)
    scores_by_stage = _scores_by_stage(deviation_table)
    stage_scores: dict[Stage, float | None] = {}

    for stage in stage_order:
        scores = scores_by_stage.get(stage, [])
        stage_scores[stage] = None if not scores else _clip01(1.0 - mean(scores))

    observed_scores = [
        score for score in stage_scores.values() if score is not None
    ]
    mean_robustness = None if not observed_scores else mean(observed_scores)
    return RobustnessFingerprint(
        stage_scores=stage_scores,
        mean_robustness=mean_robustness,
        run_count=1,
    )


def aggregate_robustness_fingerprints(
    inputs: Sequence[DeviationTableInput],
    config: SD2Config | None = None,
) -> RobustnessFingerprint:
    """Average per-run-pair fingerprints into a model-level fingerprint.

    ``inputs`` may contain ``DeviationTable`` instances or paths to
    ``deviation_table.json`` files. Directory paths are treated as analysis
    output directories and resolved to ``deviation_table.json`` inside them.
    Missing stage scores are ignored for the stage average.
    """

    fingerprints = [
        compute_robustness_fingerprint(_coerce_deviation_table(item), config)
        for item in inputs
    ]
    stage_order = _configured_stage_order(config)
    aggregated_scores: dict[Stage, float | None] = {}

    for stage in stage_order:
        observed_scores = [
            fingerprint.stage_scores.get(stage)
            for fingerprint in fingerprints
            if fingerprint.stage_scores.get(stage) is not None
        ]
        aggregated_scores[stage] = (
            None if not observed_scores else mean(observed_scores)
        )

    observed_aggregates = [
        score for score in aggregated_scores.values() if score is not None
    ]
    mean_robustness = None if not observed_aggregates else mean(observed_aggregates)
    return RobustnessFingerprint(
        stage_scores=aggregated_scores,
        mean_robustness=mean_robustness,
        run_count=len(fingerprints),
    )


def load_deviation_table_json(path: str | Path) -> DeviationTable:
    """Load a JSON deviation table written by ``DeviationTable.write_json``."""

    json_path = Path(path)
    if json_path.is_dir():
        json_path = json_path / "deviation_table.json"

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"deviation table JSON must contain a list: {json_path}")

    return DeviationTable(
        records=[
            DeviationRecord(
                pair_key=str(row["pair_key"]),
                frame_idx=int(row["frame_idx"]),
                timestamp=float(row["timestamp"]),
                stage=Stage(str(row["stage"])),
                metric=str(row["metric"]),
                raw_score=float(row["raw_score"]),
                normalized_score=float(row["normalized_score"]),
                status=str(row["status"]),
                missing=bool(row["missing"]),
                details=dict(row.get("details", {})),
            )
            for row in rows
        ]
    )


def _coerce_deviation_table(item: DeviationTableInput) -> DeviationTable:
    if isinstance(item, DeviationTable):
        return item
    return load_deviation_table_json(item)


def _scores_by_stage(deviation_table: DeviationTable) -> dict[Stage, list[float]]:
    scores_by_stage: dict[Stage, list[float]] = defaultdict(list)
    for record in deviation_table.records:
        if record.missing or not isfinite(record.normalized_score):
            continue
        scores_by_stage[record.stage].append(float(record.normalized_score))
    return scores_by_stage


def _configured_stage_order(config: SD2Config | None) -> list[Stage]:
    raw_stages = Stage.ordered() if config is None else config.stages
    stages: list[Stage] = []
    for raw_stage in raw_stages:
        stage = raw_stage if isinstance(raw_stage, Stage) else Stage(str(raw_stage))
        if stage != Stage.OUTCOME:
            stages.append(stage)
    return stages


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
