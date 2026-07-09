"""End-to-end offline analysis pipeline helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.analysis.diagnosis import compute_failure_diagnosis
from sd2.analysis.deviation import compute_deviation_table
from sd2.analysis.fingerprint import compute_robustness_fingerprint
from sd2.analysis.propagation import compute_propagation_analysis
from sd2.analysis.thresholds import resolve_threshold_set
from sd2.core.config import load_config
from sd2.core.run import (
    DEFAULT_PAIRING_MODE,
    DEFAULT_PROGRESS_TOLERANCE,
    DEFAULT_TIMESTAMP_TOLERANCE,
    pair_runs,
)


@dataclass(frozen=True)
class AnalysisOutput:
    """Filesystem paths written by one SD2 analysis run."""

    output_dir: Path
    paired_frames_path: Path
    pairing_summary_path: Path
    deviation_json_path: Path
    deviation_csv_path: Path
    propagation_path: Path
    diagnosis_path: Path
    fingerprint_path: Path
    report_path: Path | None = None


def run_analysis(
    clean_path: str | Path,
    stress_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    report: bool = False,
    thresholds_path: str | Path | None = None,
) -> AnalysisOutput:
    """Run pairing, deviation, propagation, diagnosis, and fingerprint analysis."""

    config = load_config(config_path)
    thresholds = resolve_threshold_set(config, thresholds_path)
    clean = load_run_jsonl(clean_path)
    stress = load_run_jsonl(stress_path)
    paired_run = pair_runs(clean, stress, **_pairing_options(config.pairing))
    deviation_table = compute_deviation_table(paired_run, config, thresholds)
    propagation_result = compute_propagation_analysis(deviation_table, config, thresholds)
    diagnosis_result = compute_failure_diagnosis(
        deviation_table,
        propagation_result,
        paired_run,
        config,
    )
    fingerprint = compute_robustness_fingerprint(deviation_table, config)

    analysis_dir = Path(output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    paired_frames_path = analysis_dir / "paired_frames.json"
    pairing_summary_path = analysis_dir / "pairing_summary.json"
    deviation_json_path = analysis_dir / "deviation_table.json"
    deviation_csv_path = analysis_dir / "deviation_table.csv"
    propagation_path = analysis_dir / "propagation.json"
    diagnosis_path = analysis_dir / "diagnosis.json"
    fingerprint_path = analysis_dir / "fingerprint.json"

    paired_payload = [
        paired.model_dump(mode="json")
        for paired in paired_run.pairs
    ]
    summary_payload = paired_run.summary.model_dump(mode="json")

    paired_frames_path.write_text(
        json.dumps(paired_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    pairing_summary_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    deviation_table.write_json(deviation_json_path)
    deviation_table.write_csv(deviation_csv_path)
    propagation_result.write_json(propagation_path)
    diagnosis_result.write_json(diagnosis_path)
    fingerprint.write_json(fingerprint_path)

    report_path = None
    if report:
        from sd2.reports.markdown import generate_report

        report_path = generate_report(analysis_dir)

    return AnalysisOutput(
        output_dir=analysis_dir,
        paired_frames_path=paired_frames_path,
        pairing_summary_path=pairing_summary_path,
        deviation_json_path=deviation_json_path,
        deviation_csv_path=deviation_csv_path,
        propagation_path=propagation_path,
        diagnosis_path=diagnosis_path,
        fingerprint_path=fingerprint_path,
        report_path=report_path,
    )


def _pairing_options(pairing: dict[str, object]) -> dict[str, object]:
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
