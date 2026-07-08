import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sd2.analysis.pipeline import run_analysis
from sd2.reports.markdown import (
    aggregate_fingerprint_files,
    generate_fingerprint_summary,
)


def test_report_generation_from_sample_analysis_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    result = run_analysis(
        clean_path="data/sample/clean_run.jsonl",
        stress_path="data/sample/stress_run.jsonl",
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
        report=True,
    )

    report_path = result.report_path
    assert report_path is not None
    assert report_path.is_file()

    text = report_path.read_text(encoding="utf-8")
    assert "Reasoning" in text
    assert "temporal-correlational" in text
    assert "earliest critical deviation" in text
    assert "preceding downstream" in text
    assert "caused" not in text.lower()
    for header in [
        "## Summary Diagnosis",
        "## Final Outcome Comparison",
        "## Stage-wise Mean Deviation",
        "## Collapse Onset Times",
        "## Propagation Summary",
        "## Robustness Fingerprint",
        "## Embedded Plots",
    ]:
        assert header in text

    for name in [
        "deviation_timeline.png",
        "robustness_fingerprint.png",
        "propagation_scores.png",
    ]:
        path = output_dir / "plots" / name
        assert path.is_file()
        assert path.stat().st_size > 0

    links = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    assert links
    for link in links:
        assert (report_path.parent / link).is_file()


def test_cli_report_empty_analysis_dir_errors(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "report",
            "--analysis-dir",
            str(empty_dir),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Run `sd2 analyze` first" in result.stderr


def test_fingerprint_aggregation_over_two_synthetic_files(tmp_path: Path) -> None:
    _write_analysis_fingerprint(
        tmp_path / "run_a",
        {"vision": 0.8, "semantic": 0.5},
    )
    _write_analysis_fingerprint(
        tmp_path / "run_b",
        {"vision": 0.6, "semantic": None},
    )

    aggregates = aggregate_fingerprint_files(tmp_path)

    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.run_count == 2
    assert aggregate.stage_scores["vision"] == pytest.approx(0.7)
    assert aggregate.stage_scores["semantic"] == pytest.approx(0.5)
    assert aggregate.mean_robustness == pytest.approx(0.6)

    summary_path = generate_fingerprint_summary(
        tmp_path,
        tmp_path / "fingerprint_summary.md",
    )
    text = summary_path.read_text(encoding="utf-8")
    assert "model_a" in text
    assert "gaussian_noise" in text
    assert "0.700" in text


def _write_analysis_fingerprint(path: Path, stage_scores: dict[str, float | None]) -> None:
    path.mkdir()
    fingerprint = {
        "stage_scores": stage_scores,
        "mean_robustness": None,
        "run_count": 1,
    }
    pairing_summary = {
        "model_id": "model_a",
        "scenario_id": "scenario_a",
        "seed": 42,
        "clean_metadata": {
            "model_id": "model_a",
            "scenario_id": "scenario_a",
            "condition": "clean",
            "severity": 0,
            "seed": 42,
        },
        "stress_metadata": {
            "model_id": "model_a",
            "scenario_id": "scenario_a",
            "condition": "stress",
            "stress_type": "gaussian_noise",
            "severity": 3,
            "seed": 42,
        },
    }
    (path / "fingerprint.json").write_text(
        json.dumps(fingerprint),
        encoding="utf-8",
    )
    (path / "pairing_summary.json").write_text(
        json.dumps(pairing_summary),
        encoding="utf-8",
    )
