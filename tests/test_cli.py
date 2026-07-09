import json
import os
import subprocess
import sys
from pathlib import Path

from sd2.analysis.deviation import DeviationRecord, DeviationTable
from sd2.core.stage import Stage


def test_cli_analyze_sample_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "analyze",
            "--clean",
            "data/sample/clean_run.jsonl",
            "--stress",
            "data/sample/stress_run.jsonl",
            "--config",
            "configs/mvp.yaml",
            "--output",
            str(output_dir),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr

    paired_path = output_dir / "paired_frames.json"
    summary_path = output_dir / "pairing_summary.json"
    deviation_json_path = output_dir / "deviation_table.json"
    deviation_csv_path = output_dir / "deviation_table.csv"
    propagation_path = output_dir / "propagation.json"
    diagnosis_path = output_dir / "diagnosis.json"
    fingerprint_path = output_dir / "fingerprint.json"
    assert paired_path.is_file()
    assert summary_path.is_file()
    assert deviation_json_path.is_file()
    assert deviation_csv_path.is_file()
    assert propagation_path.is_file()
    assert diagnosis_path.is_file()
    assert fingerprint_path.is_file()

    paired_frames = json.loads(paired_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    deviation_rows = json.loads(deviation_json_path.read_text(encoding="utf-8"))
    propagation = json.loads(propagation_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert len(paired_frames) == 30
    assert summary["paired_count"] == 30
    assert summary["skipped_count"] == 0
    assert len(deviation_rows) == 30 * 5
    assert set(deviation_rows[0]) == {
        "pair_key",
        "frame_idx",
        "timestamp",
        "stage",
        "metric",
        "raw_score",
        "normalized_score",
        "status",
        "missing",
        "details",
    }
    assert len(deviation_csv_path.read_text(encoding="utf-8").splitlines()) == (30 * 5) + 1
    assert "reasoning" in propagation["collapse_onsets"]
    assert "ratio_clipped" in propagation["propagation_scores"][0]
    assert "collapse_order" in propagation["propagation_scores"][0]
    assert diagnosis["diagnosis_type"] == "temporal_correlational"
    assert diagnosis["primary_failure_stage"] == "reasoning"
    assert diagnosis["driving_failure"] is True
    assert diagnosis["deviation_precedes_driving_failure"] is True
    assert set(fingerprint["stage_scores"]) == {
        "vision",
        "semantic",
        "reasoning",
        "planning",
        "control",
    }


def test_cli_aggregate_explicit_runs(tmp_path: Path) -> None:
    run_a = _write_cli_analysis_run(
        tmp_path / "run_a",
        {Stage.VISION: 0.9, Stage.REASONING: 0.4},
        "reasoning",
    )
    run_b = _write_cli_analysis_run(
        tmp_path / "run_b",
        {Stage.VISION: 0.8, Stage.REASONING: 0.6},
        "reasoning",
    )
    run_c = _write_cli_analysis_run(
        tmp_path / "run_c",
        {Stage.VISION: 0.7, Stage.REASONING: 0.8},
        "vision",
    )
    output_path = tmp_path / "aggregate.md"
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "aggregate",
            "--run",
            str(run_a),
            "--run",
            str(run_b),
            "--run",
            str(run_c),
            "--output",
            str(output_path),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Overall mean robustness 0.700 +/- 0.050" in result.stdout
    assert "primary stage = reasoning in 2/3 runs" in result.stdout

    assert output_path.is_file()
    json_path = output_path.with_suffix(".json")
    assert json_path.is_file()

    markdown = output_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "| Stage | n | mean robustness | std | 95% CI |" in markdown
    assert payload["run_count"] == 3
    assert payload["diagnosis_stability"]["modal_stage"] == "reasoning"


def _write_cli_analysis_run(
    path: Path,
    robustness_scores: dict[Stage, float],
    primary_stage: str,
) -> Path:
    path.mkdir()
    records = []
    for index, (stage, robustness_score) in enumerate(robustness_scores.items()):
        deviation_score = 1.0 - robustness_score
        records.append(
            DeviationRecord(
                pair_key=f"pair:{index}",
                frame_idx=0,
                timestamp=0.0,
                stage=stage,
                metric="synthetic",
                raw_score=deviation_score,
                normalized_score=deviation_score,
                status="healthy",
                missing=False,
                details={},
            )
        )
    DeviationTable(records).write_json(path / "deviation_table.json")
    (path / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": primary_stage}) + "\n",
        encoding="utf-8",
    )
    return path
