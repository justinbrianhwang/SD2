from __future__ import annotations

import logging
import sys

import pytest

from sd2.adapters.carla_adapter import (
    Sd2JsonlWriter,
    build_carla_run_metadata,
    carla_frame_to_sd2,
    write_sd2_jsonl,
)
from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_carla_frame_to_sd2_validates_core_stages_and_preserves_ego() -> None:
    frame_record = carla_frame_to_sd2(
        {
            "frame_idx": 3,
            "timestamp": 0.15,
            "ego": {"x": 1.0, "y": 2.0, "z": 0.3, "yaw": 90.0, "speed": 5.2},
            "control": {"steer": 0.1, "throttle": 0.4, "brake": 0.0},
            "planned_waypoints": [[1.0, 2.0], [3.0, 4.0]],
            "target_speed": 5.5,
            "collision": False,
            "lane_invasion": True,
            "off_route": False,
            "route_progress": 0.42,
            "min_ttc": 3.4,
        },
        run_id="carla_run",
    )

    assert frame_record["type"] == "frame"
    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert set(frame.states) == {Stage.PLANNING, Stage.CONTROL, Stage.OUTCOME}
    planning = frame.states[Stage.PLANNING]
    assert planning.waypoints == [[1.0, 2.0], [3.0, 4.0]]
    assert planning.target_speed == 5.5
    assert planning.model_extra["ego"]["yaw"] == 90.0

    control = frame.states[Stage.CONTROL]
    assert control.steer == 0.1
    assert control.throttle == 0.4
    assert control.brake == 0.0

    outcome = frame.states[Stage.OUTCOME]
    assert outcome.collision is False
    assert outcome.lane_invasion is True
    assert outcome.off_route is False
    assert outcome.route_progress == 0.42
    assert outcome.min_ttc == 3.4


def test_carla_frame_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = carla_frame_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "ego": {"x": 1.0, "speed": 0.0},
            "control": {"steer": 0.0},
            "collision": False,
            "lane_invasion": False,
            "route_progress": 1.5,
            "min_ttc": None,
        },
        run_id="carla_run",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert "waypoints" not in frame_record["states"]["planning"]
    assert "min_ttc" not in frame_record["states"]["outcome"]
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0


def test_carla_control_anti_crawl_marker_is_conditional() -> None:
    marked = _synthetic_record(0, steer=0.0)
    marked["control"]["anti_crawl_applied"] = True
    marked["control"]["applied_throttle"] = 0.6

    marked_frame = carla_frame_to_sd2(marked, run_id="carla_run")
    marked_control = marked_frame["states"]["control"]

    assert marked_control["anti_crawl_applied"] is True
    assert marked_control["applied_throttle"] == pytest.approx(0.6)

    clean_frame = carla_frame_to_sd2(_synthetic_record(1, steer=0.0), run_id="carla_run")
    assert set(clean_frame["states"]["control"]) == {"steer", "throttle", "brake"}


def test_build_carla_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_carla_run_metadata(
        run_id="carla_basic_agent_Town10HD_Opt_spawn1_clean_seed42",
        scenario_id="Town10HD_Opt_spawn1_dest10",
        condition="clean",
        stress_type=None,
        severity=0,
        seed=42,
        town="Town10HD_Opt",
    )

    metadata = RunMetadata.model_validate(
        {key: value for key, value in metadata_record.items() if key != "type"}
    )

    assert metadata_record["type"] == "run_metadata"
    assert metadata.model_id == "carla_basic_agent"
    assert metadata.scenario_id == "Town10HD_Opt_spawn1_dest10"
    assert metadata.condition == "clean"
    assert metadata.stress_type is None


def test_carla_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn1_dest10"
    clean_run_id = "carla_basic_agent_Town10HD_Opt_spawn1_clean_seed42"
    stress_run_id = "carla_basic_agent_Town10HD_Opt_spawn1_control_noise_s3_seed42"
    clean_metadata = build_carla_run_metadata(
        clean_run_id,
        scenario_id,
        "clean",
        None,
        0,
        42,
        "Town10HD_Opt",
    )
    stress_metadata = build_carla_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "control_noise",
        3,
        42,
        "Town10HD_Opt",
    )

    clean_frames = [
        carla_frame_to_sd2(_synthetic_record(idx, steer=0.0), clean_run_id)
        for idx in range(5)
    ]
    stress_frames = [
        carla_frame_to_sd2(_synthetic_record(idx, steer=0.03), stress_run_id)
        for idx in range(5)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    clean_run = load_run_jsonl(clean_path)
    stress_run = load_run_jsonl(stress_path)
    paired = pair_runs(clean_run, stress_run)

    assert paired.summary.paired_count == 5
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"carla_basic_agent:{scenario_id}:42:0"


def test_sd2_jsonl_writer_matches_write_sd2_jsonl_bytes(tmp_path) -> None:
    metadata, frames = _sample_run(frame_count=3)
    streaming_path = tmp_path / "streaming.jsonl"
    batch_path = tmp_path / "batch.jsonl"

    with Sd2JsonlWriter(streaming_path, metadata) as writer:
        for frame in frames:
            writer.write_frame(frame)
    write_sd2_jsonl(batch_path, metadata, frames)

    assert streaming_path.read_bytes() == batch_path.read_bytes()


def test_sd2_jsonl_writer_leaves_partial_on_exception(tmp_path, caplog) -> None:
    metadata, frames = _sample_run(frame_count=3)
    path = tmp_path / "failed.jsonl"
    partial_path = path.with_suffix(path.suffix + ".partial")
    expected_path = tmp_path / "expected_partial.jsonl"
    caplog.set_level(logging.WARNING, logger="sd2.adapters.carla_adapter")

    with pytest.raises(RuntimeError, match="sensor timeout"):
        with Sd2JsonlWriter(path, metadata) as writer:
            writer.write_frame(frames[0])
            writer.write_frame(frames[1])
            raise RuntimeError("sensor timeout")

    write_sd2_jsonl(expected_path, metadata, frames[:2])
    assert not path.exists()
    assert partial_path.exists()
    assert partial_path.read_bytes() == expected_path.read_bytes()
    assert str(partial_path) in caplog.text


def test_sd2_jsonl_writer_completed_run_renames_partial(tmp_path) -> None:
    metadata, frames = _sample_run(frame_count=2)
    path = tmp_path / "complete.jsonl"
    partial_path = path.with_suffix(path.suffix + ".partial")

    with Sd2JsonlWriter(path, metadata) as writer:
        for frame in frames:
            writer.write_frame(frame)

    assert path.exists()
    assert not partial_path.exists()


def test_sd2_jsonl_writer_rejects_invalid_frame_at_write_time(tmp_path) -> None:
    metadata, frames = _sample_run(frame_count=2)
    path = tmp_path / "invalid.jsonl"
    partial_path = path.with_suffix(path.suffix + ".partial")
    invalid_frame = {**frames[1], "frame_idx": -1}

    with pytest.raises(ValueError):
        with Sd2JsonlWriter(path, metadata) as writer:
            writer.write_frame(frames[0])
            writer.write_frame(invalid_frame)

    assert not path.exists()
    assert partial_path.exists()
    assert len(partial_path.read_text(encoding="utf-8").splitlines()) == 2


def test_carla_adapter_does_not_import_carla() -> None:
    assert "carla" not in sys.modules
    assert "agents.navigation.basic_agent" not in sys.modules


def _sample_run(frame_count: int) -> tuple[dict, list[dict]]:
    scenario_id = "Town10HD_Opt_spawn1_dest10"
    run_id = "carla_basic_agent_Town10HD_Opt_spawn1_clean_seed42"
    metadata = build_carla_run_metadata(
        run_id,
        scenario_id,
        "clean",
        None,
        0,
        42,
        "Town10HD_Opt",
    )
    frames = [
        carla_frame_to_sd2(_synthetic_record(idx, steer=0.0), run_id)
        for idx in range(frame_count)
    ]
    return metadata, frames


def _synthetic_record(frame_idx: int, steer: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "ego": {
            "x": float(frame_idx),
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "speed": 4.0,
        },
        "control": {"steer": steer, "throttle": 0.4, "brake": 0.0},
        "planned_waypoints": [[frame_idx + step, 0.0] for step in range(5)],
        "target_speed": 5.0,
        "collision": False,
        "lane_invasion": False,
        "route_progress": frame_idx / 4,
        "min_ttc": None,
    }
