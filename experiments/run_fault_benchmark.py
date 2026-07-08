"""Run the SD2 synthetic fault-injection benchmark."""

from __future__ import annotations

from pathlib import Path

from sd2.benchmark.report import generate_benchmark_report, headline_accuracy
from sd2.benchmark.runner import run_fault_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "fault_benchmark"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp.yaml"


def main() -> None:
    result = run_fault_benchmark(
        config_path=CONFIG_PATH,
        work_dir=OUTPUT_DIR,
        n_per_class=20,
        seed=42,
    )
    result_path = result.write_json(OUTPUT_DIR / "benchmark_result.json")
    report_output = generate_benchmark_report(result, OUTPUT_DIR)

    print(headline_accuracy(result))
    print(f"Result JSON: {_display_path(result_path)}")
    print(f"Report: {_display_path(report_output.report_path)}")
    print(f"Confusion plot: {_display_path(report_output.heatmap_path)}")


def _display_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


if __name__ == "__main__":
    main()
