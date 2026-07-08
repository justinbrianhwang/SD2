"""Runner for the synthetic primary-failure-stage benchmark."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sd2.analysis.pipeline import run_analysis
from sd2.benchmark.synthetic import (
    FAULT_STAGES,
    SyntheticRunPair,
    generate_synthetic_pairs,
    materialize_pair,
)


logger = logging.getLogger(__name__)
NO_FAILURE = "no_failure"
PREDICTION_COLUMNS = [stage.value for stage in FAULT_STAGES] + [NO_FAILURE]


@dataclass(frozen=True)
class BenchmarkRecord:
    """One benchmark sample diagnosis result."""

    run_id: str
    target_stage: str
    predicted_stage: str | None
    correct: bool
    onset_frame: int
    sample_dir: str
    diagnosis_status: str | None = None
    fallback_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return {
            "run_id": self.run_id,
            "target_stage": self.target_stage,
            "predicted_stage": self.predicted_stage,
            "correct": self.correct,
            "onset_frame": self.onset_frame,
            "sample_dir": self.sample_dir,
            "diagnosis_status": self.diagnosis_status,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate synthetic benchmark result."""

    records: list[BenchmarkRecord]
    overall_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    n_per_class: int | None = None
    seed: int | None = None
    frame_count: int | None = None
    config_path: str | None = None
    work_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable benchmark result."""

        return {
            "overall_accuracy": self.overall_accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "confusion_matrix": self.confusion_matrix,
            "records": [record.to_dict() for record in self.records],
            "n_per_class": self.n_per_class,
            "seed": self.seed,
            "frame_count": self.frame_count,
            "config_path": self.config_path,
            "work_dir": self.work_dir,
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the result JSON and return the output path."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output


def run_fault_benchmark(
    config_path: str | Path,
    work_dir: str | Path,
    n_per_class: int = 20,
    seed: int = 42,
    *,
    frame_count: int = 30,
    profile: str = "realistic",
) -> BenchmarkResult:
    """Run the labeled synthetic benchmark through the real SD2 pipeline."""

    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    pairs = generate_synthetic_pairs(
        n_per_class=n_per_class,
        seed=seed,
        frame_count=frame_count,
        profile=profile,
    )

    records: list[BenchmarkRecord] = []
    per_class_seen: dict[str, int] = {stage.value: 0 for stage in FAULT_STAGES}
    for pair in pairs:
        class_index = per_class_seen[pair.target_stage.value]
        per_class_seen[pair.target_stage.value] += 1
        sample_dir = root / pair.target_stage.value / f"{class_index:03d}"
        paths = materialize_pair(pair, sample_dir)
        analysis_dir = sample_dir / "analysis"
        output = run_analysis(
            clean_path=paths.clean_path,
            stress_path=paths.stress_path,
            config_path=config_path,
            output_dir=analysis_dir,
            report=False,
        )
        diagnosis = json.loads(output.diagnosis_path.read_text(encoding="utf-8"))
        predicted = diagnosis.get("primary_failure_stage")
        predicted_stage = str(predicted) if predicted else None
        if predicted_stage is None:
            logger.warning("No primary failure stage diagnosed for %s", pair.run_id)
        records.append(_record_from_pair(pair, predicted_stage, diagnosis, sample_dir))

    return compute_benchmark_result(
        records,
        n_per_class=n_per_class,
        seed=seed,
        frame_count=frame_count,
        config_path=str(config_path),
        work_dir=str(root),
    )


def compute_benchmark_result(
    records: Iterable[BenchmarkRecord],
    *,
    n_per_class: int | None = None,
    seed: int | None = None,
    frame_count: int | None = None,
    config_path: str | None = None,
    work_dir: str | None = None,
) -> BenchmarkResult:
    """Compute accuracy and confusion summaries from per-sample records."""

    record_list = list(records)
    total = len(record_list)
    correct_count = sum(1 for record in record_list if record.correct)
    overall_accuracy = 0.0 if total == 0 else correct_count / total

    per_class_accuracy: dict[str, float] = {}
    for stage in [stage.value for stage in FAULT_STAGES]:
        stage_records = [
            record for record in record_list if record.target_stage == stage
        ]
        if not stage_records:
            per_class_accuracy[stage] = 0.0
            continue
        per_class_accuracy[stage] = (
            sum(1 for record in stage_records if record.correct) / len(stage_records)
        )

    confusion = _empty_confusion_matrix()
    for record in record_list:
        target = record.target_stage
        if target not in confusion:
            confusion[target] = {column: 0 for column in PREDICTION_COLUMNS}
        predicted = record.predicted_stage or NO_FAILURE
        if predicted not in confusion[target]:
            confusion[target][predicted] = 0
        confusion[target][predicted] += 1

    return BenchmarkResult(
        records=record_list,
        overall_accuracy=overall_accuracy,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion,
        n_per_class=n_per_class,
        seed=seed,
        frame_count=frame_count,
        config_path=config_path,
        work_dir=work_dir,
    )


def _record_from_pair(
    pair: SyntheticRunPair,
    predicted_stage: str | None,
    diagnosis: dict[str, Any],
    sample_dir: Path,
) -> BenchmarkRecord:
    target = pair.target_stage.value
    return BenchmarkRecord(
        run_id=pair.run_id,
        target_stage=target,
        predicted_stage=predicted_stage,
        correct=predicted_stage == target,
        onset_frame=pair.onset_frame,
        sample_dir=str(sample_dir),
        diagnosis_status=diagnosis.get("status"),
        fallback_used=diagnosis.get("fallback_used"),
    )


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        stage.value: {column: 0 for column in PREDICTION_COLUMNS}
        for stage in FAULT_STAGES
    }
