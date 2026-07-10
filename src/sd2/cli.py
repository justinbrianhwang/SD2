"""Command-line interface for SD2."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Sequence

from sd2 import __version__
from sd2.analysis.calibration import (
    calibrate_thresholds,
    format_calibrated_threshold_table,
)
from sd2.analysis.pipeline import run_analysis
from sd2.analysis.intervention import (
    run_intervention_analysis,
    run_single_run_share_analysis,
)
from sd2.analysis.robustness_stats import (
    aggregate_run_statistics,
    discover_analysis_dirs,
    format_statistical_report,
)
from sd2.benchmark.report import generate_benchmark_report, headline_accuracy
from sd2.benchmark.runner import run_fault_benchmark
from sd2.core.config import load_config
from sd2.reports.markdown import generate_fingerprint_summary, generate_report
from sd2.stressors.pipeline import run_stress


def build_parser() -> argparse.ArgumentParser:
    """Build the SD2 argument parser."""

    parser = argparse.ArgumentParser(prog="sd2", description="System Deviation Diagnosis")
    parser.add_argument("--version", action="version", version=f"sd2 {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="pair clean and stress run logs")
    analyze.add_argument("--clean", required=True, help="path to clean run JSONL")
    analyze.add_argument("--stress", required=True, help="path to stress run JSONL")
    analyze.add_argument("--config", required=True, help="path to YAML config")
    analyze.add_argument("--output", required=True, help="output directory")
    analyze.add_argument(
        "--thresholds",
        help="optional calibrated_thresholds.json for per-stage status thresholds",
    )
    analyze.add_argument(
        "--report",
        action="store_true",
        help="generate report.md and plots after analysis",
    )

    calibrate = subparsers.add_parser(
        "calibrate",
        help="calibrate warning/critical thresholds from repeated clean runs",
    )
    calibrate.add_argument(
        "--clean",
        action="append",
        default=[],
        help="path to a clean run JSONL; repeat for two or more runs",
    )
    calibrate.add_argument(
        "--clean-glob",
        action="append",
        default=[],
        help="glob pattern for clean run JSONL files",
    )
    calibrate.add_argument("--config", required=True, help="path to YAML config")
    calibrate.add_argument(
        "--output",
        required=True,
        help="output file or directory for calibrated_thresholds.json",
    )
    calibrate.add_argument(
        "--k-warning",
        type=float,
        default=2.0,
        help="warning threshold multiplier, default: 2.0",
    )
    calibrate.add_argument(
        "--k-critical",
        type=float,
        default=3.0,
        help="critical threshold multiplier, default: 3.0",
    )

    report = subparsers.add_parser("report", help="generate a Markdown report")
    report.add_argument("--analysis-dir", required=True, help="analysis output directory")
    report.add_argument(
        "--output",
        help="report path, default: <analysis-dir>/report.md",
    )

    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="aggregate fingerprint.json files into a Markdown summary",
    )
    fingerprint.add_argument(
        "--analysis-dir",
        required=True,
        help="analysis directory or parent directory containing analyses",
    )
    fingerprint.add_argument("--output", required=True, help="summary Markdown path")

    aggregate = subparsers.add_parser(
        "aggregate",
        help="aggregate statistical robustness over multiple analysis runs",
    )
    aggregate.add_argument(
        "--analysis-dir",
        help="analysis directory or parent directory containing analyses",
    )
    aggregate.add_argument(
        "--run",
        action="append",
        default=[],
        help="explicit analysis output directory; repeat for multiple runs",
    )
    aggregate.add_argument("--output", required=True, help="output path base, .md, or .json")

    stress = subparsers.add_parser("stress", help="apply input stress to image frames")
    stress.add_argument("--input", required=True, help="input directory of PNG/JPG images")
    stress.add_argument("--config", required=True, help="path to stress YAML config")
    stress.add_argument("--output", required=True, help="output directory")
    stress.add_argument(
        "--seed",
        type=int,
        default=42,
        help="deterministic RNG seed, default: 42",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run the synthetic primary-failure-stage benchmark",
    )
    benchmark.add_argument("--config", required=True, help="path to YAML config")
    benchmark.add_argument("--output", required=True, help="output directory")
    benchmark.add_argument(
        "--n-per-class",
        type=int,
        default=20,
        help="synthetic samples per target stage, default: 20",
    )
    benchmark.add_argument(
        "--seed",
        type=int,
        default=42,
        help="deterministic RNG seed, default: 42",
    )
    benchmark.add_argument(
        "--profile",
        choices=["realistic", "hard"],
        default="realistic",
        help="synthetic benchmark profile, default: realistic",
    )

    intervention = subparsers.add_parser(
        "intervention",
        help="analyze a counterfactual stage-intervention run",
    )
    intervention.add_argument(
        "--baseline-clean",
        help="path to the baseline clean run JSONL",
    )
    intervention.add_argument("--stress", help="path to stress run JSONL")
    intervention.add_argument(
        "--intervened",
        help="path to intervened run JSONL",
    )
    intervention.add_argument(
        "--none-run",
        help="single --intervene-stage none run JSONL for common-denominator stage shares",
    )
    intervention.add_argument(
        "--clean-replicates",
        nargs="*",
        default=[],
        help="optional clean replicate JSONL paths for outcome recovery noise-floor calibration",
    )
    intervention.add_argument("--config", help="path to YAML config")
    intervention.add_argument("--output", required=True, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SD2 CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "calibrate":
        return _run_calibrate(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "fingerprint":
        return _run_fingerprint(args)
    if args.command == "aggregate":
        return _run_aggregate(args)
    if args.command == "stress":
        return _run_stress(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "intervention":
        return _run_intervention(args)

    parser.print_help()
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    run_analysis(
        clean_path=args.clean,
        stress_path=args.stress,
        config_path=args.config,
        output_dir=args.output,
        report=bool(args.report),
        thresholds_path=args.thresholds,
    )
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    try:
        clean_paths = _resolve_clean_paths(args.clean, args.clean_glob)
        config = load_config(args.config)
        result = calibrate_thresholds(
            clean_paths,
            config,
            k_warning=args.k_warning,
            k_critical=args.k_critical,
        )
        output = _calibration_output_path(args.output)
        result.write_json(output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_calibrated_threshold_table(result))
    print(f"Wrote {output}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    try:
        generate_report(
            analysis_dir=args.analysis_dir,
            output_path=Path(args.output) if args.output else None,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_fingerprint(args: argparse.Namespace) -> int:
    try:
        generate_fingerprint_summary(args.analysis_dir, args.output)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    try:
        run_dirs = _resolve_analysis_dirs(args.analysis_dir, args.run)
        report = aggregate_run_statistics(run_dirs)
        markdown_path, json_path = _aggregate_output_paths(args.output)
        report.write_json(json_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            format_statistical_report(report),
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_aggregate_headline(report))
    return 0


def _run_stress(args: argparse.Namespace) -> int:
    try:
        run_stress(
            input_path=args.input,
            config_path=args.config,
            output_dir=args.output,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    try:
        result = run_fault_benchmark(
            config_path=args.config,
            work_dir=args.output,
            n_per_class=args.n_per_class,
            seed=args.seed,
            profile=args.profile,
        )
        output_dir = Path(args.output)
        result.write_json(output_dir / "benchmark_result.json")
        generate_benchmark_report(result, output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(headline_accuracy(result))
    return 0


def _run_intervention(args: argparse.Namespace) -> int:
    try:
        if args.none_run:
            output = run_single_run_share_analysis(
                none_run=args.none_run,
                output_dir=args.output,
            )
        else:
            missing = [
                name
                for name in ("baseline_clean", "stress", "intervened", "config")
                if getattr(args, name) is None
            ]
            if missing:
                formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
                raise ValueError(f"intervention analysis requires {formatted}")
            output = run_intervention_analysis(
                baseline_clean=args.baseline_clean,
                stress=args.stress,
                intervened=args.intervened,
                config_path=args.config,
                output_dir=args.output,
                clean_replicates=args.clean_replicates,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {output.json_path}")
    print(f"Wrote {output.markdown_path}")
    return 0


def _resolve_clean_paths(
    explicit_paths: list[str],
    patterns: list[str],
) -> list[Path]:
    paths = [Path(path) for path in explicit_paths]
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if len(unique) < 2:
        raise ValueError("calibration requires at least two --clean paths")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"clean run not found: {missing[0]}")
    return unique


def _calibration_output_path(raw_output: str) -> Path:
    output = Path(raw_output)
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / "calibrated_thresholds.json"


def _resolve_analysis_dirs(
    analysis_dir: str | None,
    explicit_runs: list[str],
) -> list[Path]:
    paths: list[Path] = []
    if analysis_dir:
        paths.extend(discover_analysis_dirs(analysis_dir))
    paths.extend(Path(path) for path in explicit_runs)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)

    if not unique:
        raise ValueError("aggregate requires --analysis-dir or at least one --run")
    return unique


def _aggregate_output_paths(raw_output: str) -> tuple[Path, Path]:
    output = Path(raw_output)
    suffix = output.suffix.lower()
    if suffix == ".md":
        return output, output.with_suffix(".json")
    if suffix == ".json":
        return output.with_suffix(".md"), output
    return Path(f"{output}.md"), Path(f"{output}.json")


def _aggregate_headline(report) -> str:
    overall = report.mean_robustness_stat
    stability = report.diagnosis_stability
    modal_count = (
        0
        if stability.modal_stage is None
        else stability.primary_stage_counts.get(stability.modal_stage, 0)
    )
    return (
        "Overall mean robustness "
        f"{_format_headline_float(overall.mean)} +/- "
        f"{_format_headline_float(overall.std)}; "
        f"primary stage = {stability.modal_stage or 'n/a'} in "
        f"{modal_count}/{stability.n_runs} runs "
        f"(stability {_format_headline_float(stability.stability)})"
    )


def _format_headline_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
