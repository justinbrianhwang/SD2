import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sd2.analysis.pipeline import run_analysis
from sd2.benchmark.runner import (
    BenchmarkRecord,
    compute_benchmark_result,
    run_fault_benchmark,
)
from sd2.benchmark.synthetic import generate_synthetic_pairs, materialize_pair


def test_clean_cut_synthetic_faults_hit_each_target_stage(tmp_path: Path) -> None:
    pairs = generate_synthetic_pairs(
        n_per_class=1,
        seed=7,
        frame_count=18,
        profile="clean_cut",
    )

    predictions = {}
    for pair in pairs:
        sample_dir = tmp_path / pair.target_stage.value
        paths = materialize_pair(pair, sample_dir)
        output = run_analysis(
            clean_path=paths.clean_path,
            stress_path=paths.stress_path,
            config_path="configs/mvp.yaml",
            output_dir=sample_dir / "analysis",
            report=False,
        )
        diagnosis = json.loads(output.diagnosis_path.read_text(encoding="utf-8"))
        predictions[pair.target_stage.value] = diagnosis["primary_failure_stage"]

    assert predictions == {
        "vision": "vision",
        "semantic": "semantic",
        "reasoning": "reasoning",
        "planning": "planning",
        "control": "control",
    }


def test_benchmark_accuracy_and_confusion_matrix_from_records() -> None:
    records = [
        _record("run_1", "vision", "vision"),
        _record("run_2", "vision", "semantic"),
        _record("run_3", "semantic", None),
        _record("run_4", "semantic", "semantic"),
        _record("run_5", "control", "control"),
    ]

    result = compute_benchmark_result(records)

    assert result.overall_accuracy == pytest.approx(3 / 5)
    assert result.per_class_accuracy["vision"] == pytest.approx(0.5)
    assert result.per_class_accuracy["semantic"] == pytest.approx(0.5)
    assert result.per_class_accuracy["control"] == pytest.approx(1.0)
    assert result.confusion_matrix["vision"]["vision"] == 1
    assert result.confusion_matrix["vision"]["semantic"] == 1
    assert result.confusion_matrix["semantic"]["no_failure"] == 1
    assert result.confusion_matrix["semantic"]["semantic"] == 1


def test_fault_benchmark_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_fault_benchmark(
        config_path="configs/mvp.yaml",
        work_dir=tmp_path / "first",
        n_per_class=1,
        seed=11,
        frame_count=18,
    )
    second = run_fault_benchmark(
        config_path="configs/mvp.yaml",
        work_dir=tmp_path / "second",
        n_per_class=1,
        seed=11,
        frame_count=18,
    )

    first_signature = [
        (record.target_stage, record.predicted_stage, record.onset_frame)
        for record in first.records
    ]
    second_signature = [
        (record.target_stage, record.predicted_stage, record.onset_frame)
        for record in second.records
    ]
    assert first_signature == second_signature


def test_benchmark_cli_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "benchmark",
            "--config",
            "configs/mvp.yaml",
            "--output",
            str(output_dir),
            "--n-per-class",
            "1",
            "--seed",
            "5",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Primary Failure Stage Diagnosis Accuracy:" in result.stdout
    assert (output_dir / "benchmark_result.json").is_file()
    assert (output_dir / "benchmark_report.md").is_file()
    assert (output_dir / "confusion_matrix.png").is_file()


def _record(
    run_id: str,
    target_stage: str,
    predicted_stage: str | None,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id,
        target_stage=target_stage,
        predicted_stage=predicted_stage,
        correct=target_stage == predicted_stage,
        onset_frame=10,
        sample_dir=".",
    )
