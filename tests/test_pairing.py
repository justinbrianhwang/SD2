import pytest

from sd2.core.run import RunLog, pair_runs
from sd2.core.schema import FrameLog, RunMetadata


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


def _frame(run_id: str, frame_idx: int) -> FrameLog:
    return FrameLog(
        run_id=run_id,
        frame_idx=frame_idx,
        timestamp=frame_idx * 0.1,
        states={"control": {"steer": 0.0, "throttle": 0.2, "brake": 0.0}},
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
    assert [pair.frame_idx for pair in paired.pairs] == [0, 1, 2]
    assert paired.pairs[0].pair_key == "openemma:town05_route01:42:0"


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


def test_pair_runs_rejects_metadata_mismatch() -> None:
    clean = RunLog(metadata=_metadata("clean_run", "clean"), frames=[])
    stress = RunLog(metadata=_metadata("stress_run", "stress", seed=43), frames=[])

    with pytest.raises(ValueError) as exc_info:
        pair_runs(clean, stress)

    assert "seed mismatch" in str(exc_info.value)
