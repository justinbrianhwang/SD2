import json
import os
import subprocess
import sys
from pathlib import Path


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
    assert paired_path.is_file()
    assert summary_path.is_file()

    paired_frames = json.loads(paired_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(paired_frames) == 30
    assert summary["paired_count"] == 30
    assert summary["skipped_count"] == 0
