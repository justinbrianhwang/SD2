"""Command-line interface for SD2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sd2 import __version__
from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.core.config import load_config
from sd2.core.run import pair_runs


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SD2 CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _run_analyze(args)

    parser.print_help()
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    load_config(args.config)
    clean = load_run_jsonl(args.clean)
    stress = load_run_jsonl(args.stress)
    paired_run = pair_runs(clean, stress)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_payload = [
        paired.model_dump(mode="json")
        for paired in paired_run.pairs
    ]
    summary_payload = paired_run.summary.model_dump(mode="json")

    (output_dir / "paired_frames.json").write_text(
        json.dumps(paired_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pairing_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
