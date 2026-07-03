"""Run the SD2 MVP sample analysis and report end to end."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import generate_sample_data
from sd2.analysis.pipeline import run_analysis


def main() -> None:
    generate_sample_data.main()

    result = run_analysis(
        clean_path=REPO_ROOT / "data" / "sample" / "clean_run.jsonl",
        stress_path=REPO_ROOT / "data" / "sample" / "stress_run.jsonl",
        config_path=REPO_ROOT / "configs" / "mvp.yaml",
        output_dir=REPO_ROOT / "outputs" / "sample_analysis",
        report=True,
    )

    print(f"Analysis directory: {_rel(result.output_dir)}")
    print(f"Report: {_rel(result.report_path)}")
    print(f"Plots: {_rel(result.output_dir / 'plots')}")


def _rel(path: Path | None) -> str:
    if path is None:
        return "n/a"
    return path.relative_to(REPO_ROOT).as_posix()


if __name__ == "__main__":
    main()
