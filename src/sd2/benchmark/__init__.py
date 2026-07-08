"""Synthetic validation benchmarks for SD2."""

from sd2.benchmark.runner import BenchmarkRecord, BenchmarkResult, run_fault_benchmark
from sd2.benchmark.synthetic import SyntheticRunPair, generate_synthetic_pairs

__all__ = [
    "BenchmarkRecord",
    "BenchmarkResult",
    "SyntheticRunPair",
    "generate_synthetic_pairs",
    "run_fault_benchmark",
]
