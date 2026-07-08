import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sd2.analysis.calibration import calibrate_thresholds
from sd2.analysis.pipeline import run_analysis
from sd2.benchmark.synthetic import (
    generate_repeated_clean_runs,
    materialize_clean_runs,
)
from sd2.core.config import SD2Config, load_config
from sd2.core.run import RunLog
from sd2.core.schema import FrameLog, RunMetadata


def test_calibration_computes_clean_clean_thresholds_and_fallbacks() -> None:
    config = SD2Config.model_validate(
        {
            "stages": ["control", "outcome"],
            "thresholds": {"warning": 0.4, "critical": 0.7},
            "metrics": {"control": {"type": "weighted_action_mae"}},
        }
    )
    runs = [
        _control_run("clean_a", [0.0, 0.0, 0.0]),
        _control_run("clean_b", [0.2, 0.2, 0.2]),
        _control_run("clean_c", [0.4, 0.4, 0.4]),
    ]

    calibrated = calibrate_thresholds(runs, config, k_warning=2.0, k_critical=3.0)
    control = calibrated.stages[next(iter(calibrated.stages))]

    assert control.clean_clean_mean == pytest.approx(0.0666666667)
    assert control.clean_clean_std == pytest.approx(0.023570226)
    assert control.warning == pytest.approx(0.113807119)
    assert control.critical == pytest.approx(0.137377345)
    assert control.fallback_to_config is False

    fallback = calibrate_thresholds(
        [
            _control_run("zero_a", [0.0, 0.0]),
            _control_run("zero_b", [0.0, 0.0]),
        ],
        config,
    )
    fallback_control = fallback.stages[next(iter(fallback.stages))]
    assert fallback_control.fallback_to_config is True
    assert fallback_control.warning == pytest.approx(0.4)
    assert fallback_control.critical == pytest.approx(0.7)


def test_analyze_thresholds_changes_status_on_constructed_case(tmp_path: Path) -> None:
    config = SD2Config.model_validate(
        {
            "stages": ["control", "outcome"],
            "thresholds": {"warning": 0.4, "critical": 0.7},
            "metrics": {"control": {"type": "weighted_action_mae"}},
            "diagnosis": {"downstream_window": 2},
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "stages: [control, outcome]",
                "thresholds:",
                "  warning: 0.4",
                "  critical: 0.7",
                "metrics:",
                "  control:",
                "    type: weighted_action_mae",
                "diagnosis:",
                "  downstream_window: 2",
            ]
        ),
        encoding="utf-8",
    )
    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    _write_control_jsonl(clean_path, "clean", [0.0])
    _write_control_jsonl(stress_path, "stress", [1.0])

    static = run_analysis(
        clean_path=clean_path,
        stress_path=stress_path,
        config_path=config_path,
        output_dir=tmp_path / "static",
    )
    thresholds_path = tmp_path / "calibrated_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "warning": 0.4,
                "critical": 0.7,
                "stages": {
                    "control": {
                        "warning": 0.1,
                        "critical": 0.3,
                        "clean_clean_mean": 0.05,
                        "clean_clean_std": 0.02,
                        "fallback_to_config": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calibrated = run_analysis(
        clean_path=clean_path,
        stress_path=stress_path,
        config_path=config_path,
        output_dir=tmp_path / "calibrated",
        thresholds_path=thresholds_path,
    )

    static_rows = json.loads(static.deviation_json_path.read_text(encoding="utf-8"))
    calibrated_rows = json.loads(calibrated.deviation_json_path.read_text(encoding="utf-8"))
    assert static_rows[0]["normalized_score"] == pytest.approx(0.25)
    assert static_rows[0]["status"] == "healthy"
    assert calibrated_rows[0]["status"] == "warning"


def test_calibrate_cli_writes_table_and_json(tmp_path: Path) -> None:
    runs = generate_repeated_clean_runs(count=3, seed=123, frame_count=12)
    clean_paths = materialize_clean_runs(runs, tmp_path / "clean_runs")
    output_dir = tmp_path / "calibration"
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "calibrate",
            "--clean",
            str(clean_paths[0]),
            "--clean",
            str(clean_paths[1]),
            "--clean",
            str(clean_paths[2]),
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
    assert "| Stage | Mean CC | Std CC | Warning | Critical | Fallback |" in result.stdout
    payload = json.loads((output_dir / "calibrated_thresholds.json").read_text())
    assert payload["calibration_type"] == "clean_clean_variance"
    assert "reasoning" in payload["stages"]
    assert load_config("configs/mvp.yaml")


def _control_run(run_id: str, steers: list[float]) -> RunLog:
    metadata = RunMetadata(
        run_id=run_id,
        model_id="model",
        scenario_id="scenario",
        condition="clean",
        severity=0,
        seed=1,
    )
    return RunLog(
        metadata=metadata,
        frames=[
            FrameLog.model_validate(
                {
                    "run_id": run_id,
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx * 0.1,
                    "states": {
                        "control": {
                            "steer": steer,
                            "throttle": 0.5,
                            "brake": 0.0,
                        }
                    },
                }
            )
            for frame_idx, steer in enumerate(steers)
        ],
    )


def _write_control_jsonl(path: Path, run_id: str, steers: list[float]) -> None:
    run = _control_run(run_id, steers)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "run_metadata",
                        **run.metadata.model_dump(mode="json"),
                        "condition": "clean" if run_id == "clean" else "stress",
                        "stress_type": None if run_id == "clean" else "constructed",
                        "severity": 0 if run_id == "clean" else 1,
                    }
                ),
                *[
                    json.dumps({"type": "frame", **frame.model_dump(mode="json")})
                    for frame in run.frames
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
