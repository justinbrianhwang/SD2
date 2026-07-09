import json
from pathlib import Path
from statistics import mean

import pytest

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.analysis.pipeline import run_analysis
from sd2.core.run import RunLog, pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage
from sd2.metrics.control import ControlWeightedMAEMetric


def _metadata(run_id: str, condition: str, seed: int = 42) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        model_id="openemma",
        scenario_id="town05_route01",
        condition=condition,
        stress_type=None if condition == "clean" else "gaussian_noise",
        severity=0 if condition == "clean" else 3,
        seed=seed,
        timestamp_start="2026-01-01T00:00:00",
    )


def _frame(
    run_id: str,
    frame_idx: int,
    *,
    timestamp: float | None = None,
    steer: float = 0.0,
    throttle: float = 0.2,
    brake: float = 0.0,
    route_progress: float | None = None,
) -> FrameLog:
    states = {
        "control": {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
        }
    }
    if route_progress is not None:
        states["outcome"] = {"route_progress": route_progress}
    return FrameLog(
        run_id=run_id,
        frame_idx=frame_idx,
        timestamp=frame_idx * 0.1 if timestamp is None else timestamp,
        states=states,
    )


def _control_values(route_progress: float) -> tuple[float, float, float]:
    return (route_progress * 2.0) - 1.0, route_progress, 0.0


def _progress_frame(run_id: str, frame_idx: int, route_progress: float) -> FrameLog:
    steer, throttle, brake = _control_values(route_progress)
    return _frame(
        run_id,
        frame_idx,
        steer=steer,
        throttle=throttle,
        brake=brake,
        route_progress=route_progress,
    )


def _mean_control_deviation(paired) -> float:
    metric = ControlWeightedMAEMetric(Stage.CONTROL)
    scores = [
        metric.compute(
            pair.clean.states[Stage.CONTROL],
            pair.stress.states[Stage.CONTROL],
        ).normalized_score
        for pair in paired.pairs
    ]
    return mean(scores)


def _write_run_jsonl(path: Path, run: RunLog) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "run_metadata", **run.metadata.model_dump(mode="json")}),
                *[
                    json.dumps({"type": "frame", **frame.model_dump(mode="json")})
                    for frame in run.frames
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_pair_runs_matches_shared_frame_indices() -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[_frame("clean_run", idx) for idx in [0, 1, 2]],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[_frame("stress_run", idx) for idx in [0, 1, 2]],
    )

    paired = pair_runs(clean, stress)

    assert paired.summary.paired_count == 3
    assert paired.summary.skipped_count == 0
    assert paired.summary.mode == "frame_idx"
    assert paired.summary.mean_anchor_mismatch == 0.0
    assert paired.summary.max_anchor_mismatch == 0.0
    assert [pair.frame_idx for pair in paired.pairs] == [0, 1, 2]
    assert paired.pairs[0].pair_key == "openemma:town05_route01:42:0"


def test_pair_runs_default_and_frame_idx_match_sample_data() -> None:
    clean = load_run_jsonl("data/sample/clean_run.jsonl")
    stress = load_run_jsonl("data/sample/stress_run.jsonl")

    default = pair_runs(clean, stress)
    explicit = pair_runs(clean, stress, mode="frame_idx")

    assert [pair.model_dump(mode="json") for pair in default.pairs] == [
        pair.model_dump(mode="json") for pair in explicit.pairs
    ]
    assert default.summary.model_dump(mode="json") == explicit.summary.model_dump(
        mode="json"
    )
    assert default.summary.mode == "frame_idx"
    assert default.summary.paired_count == 30
    assert default.summary.skipped_count == 0
    assert default.summary.mean_anchor_mismatch == 0.0
    assert default.summary.max_anchor_mismatch == 0.0
    assert all(
        pair.frame_idx == pair.clean.frame_idx == pair.stress.frame_idx
        for pair in default.pairs
    )


def test_pair_runs_skips_missing_frames_and_reports_counts() -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[_frame("clean_run", idx) for idx in [0, 1, 3]],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[_frame("stress_run", idx) for idx in [0, 2, 3]],
    )

    paired = pair_runs(clean, stress)

    assert [pair.frame_idx for pair in paired.pairs] == [0, 3]
    assert paired.summary.paired_count == 2
    assert paired.summary.skipped_count == 2
    assert paired.summary.missing_in_clean == [2]
    assert paired.summary.missing_in_stress == [1]


def test_route_progress_pairing_reduces_constructed_lag_control_deviation() -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[
            _progress_frame("clean_run", idx, idx / 10)
            for idx in range(10)
        ],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[
            _progress_frame("stress_run", idx, max(0.0, (idx - 3) / 10))
            for idx in range(13)
        ],
    )

    by_frame_idx = pair_runs(clean, stress, mode="frame_idx")
    by_progress = pair_runs(
        clean,
        stress,
        mode="route_progress",
        progress_tolerance=1.0e-9,
    )
    frame_idx_mean = _mean_control_deviation(by_frame_idx)
    progress_mean = _mean_control_deviation(by_progress)

    assert by_frame_idx.pairs[5].frame_idx == 5
    assert by_frame_idx.pairs[5].stress.frame_idx == 5
    assert by_progress.pairs[5].frame_idx == 5
    assert by_progress.pairs[5].stress.frame_idx == 8
    assert by_progress.summary.mode == "route_progress"
    assert by_progress.summary.skipped_count == 0
    assert by_progress.summary.mean_anchor_mismatch == pytest.approx(0.0)
    assert by_progress.summary.max_anchor_mismatch == pytest.approx(0.0)
    assert progress_mean == pytest.approx(0.0)
    assert frame_idx_mean > progress_mean


def test_timestamp_pairing_uses_nearest_stress_frame_and_skips_by_tolerance() -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[
            _frame("clean_run", 0, timestamp=0.0),
            _frame("clean_run", 1, timestamp=0.1),
            _frame("clean_run", 2, timestamp=0.2),
        ],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[
            _frame("stress_run", 10, timestamp=0.02),
            _frame("stress_run", 11, timestamp=0.11),
            _frame("stress_run", 12, timestamp=0.50),
        ],
    )

    paired = pair_runs(
        clean,
        stress,
        mode="timestamp",
        timestamp_tolerance=0.03,
    )

    assert [pair.frame_idx for pair in paired.pairs] == [0, 1]
    assert [pair.timestamp for pair in paired.pairs] == [0.0, 0.1]
    assert [pair.stress.frame_idx for pair in paired.pairs] == [10, 11]
    assert paired.pairs[0].pair_key == "openemma:town05_route01:42:0"
    assert paired.summary.mode == "timestamp"
    assert paired.summary.skipped_count == 1
    assert paired.summary.missing_in_clean == [12]
    assert paired.summary.missing_in_stress == [2]
    assert paired.summary.mean_anchor_mismatch == pytest.approx(0.015)
    assert paired.summary.max_anchor_mismatch == pytest.approx(0.02)


def test_run_analysis_honors_pairing_mode_from_config(tmp_path: Path) -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[
            _frame("clean_run", 0, timestamp=0.0),
            _frame("clean_run", 1, timestamp=0.1),
        ],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[
            _frame("stress_run", 10, timestamp=0.01),
            _frame("stress_run", 11, timestamp=0.12),
        ],
    )
    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    config_path = tmp_path / "config.yaml"
    _write_run_jsonl(clean_path, clean)
    _write_run_jsonl(stress_path, stress)
    config_path.write_text(
        "\n".join(
            [
                "stages: [control, outcome]",
                "pairing:",
                "  mode: timestamp",
                "  timestamp_tolerance: 0.03",
                "thresholds:",
                "  warning: 0.4",
                "  critical: 0.7",
                "metrics:",
                "  control:",
                "    type: weighted_action_mae",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run_analysis(
        clean_path=clean_path,
        stress_path=stress_path,
        config_path=config_path,
        output_dir=tmp_path / "analysis",
    )

    summary = json.loads(output.pairing_summary_path.read_text(encoding="utf-8"))
    pairs = json.loads(output.paired_frames_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "timestamp"
    assert summary["paired_count"] == 2
    assert summary["skipped_count"] == 0
    assert [pair["frame_idx"] for pair in pairs] == [0, 1]
    assert [pair["stress"]["frame_idx"] for pair in pairs] == [10, 11]


def test_route_progress_pairing_requires_route_progress_on_both_runs() -> None:
    clean = RunLog(
        metadata=_metadata("clean_run", "clean"),
        frames=[_frame("clean_run", 0)],
    )
    stress = RunLog(
        metadata=_metadata("stress_run", "stress"),
        frames=[_progress_frame("stress_run", 0, 0.0)],
    )

    with pytest.raises(ValueError) as exc_info:
        pair_runs(clean, stress, mode="route_progress")

    message = str(exc_info.value)
    assert "route_progress" in message
    assert "frame_idx" in message
    assert "Use frame_idx mode" in message


def test_pair_runs_rejects_metadata_mismatch() -> None:
    clean = RunLog(metadata=_metadata("clean_run", "clean"), frames=[])
    stress = RunLog(metadata=_metadata("stress_run", "stress", seed=43), frames=[])

    with pytest.raises(ValueError) as exc_info:
        pair_runs(clean, stress)

    assert "seed mismatch" in str(exc_info.value)
