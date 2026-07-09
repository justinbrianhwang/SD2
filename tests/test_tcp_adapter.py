from __future__ import annotations

import sys

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.adapters.tcp_adapter import (
    build_tcp_run_metadata,
    tcp_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_tcp_record_to_sd2_omits_semantic_and_records_dual_branch_control() -> None:
    frame_record = tcp_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_perception_feature",
                "input_simplification": "single_front_rgb_256x900",
            },
            "planning": {
                "pred_wp": [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2], [7.0, 0.2]],
                "target_speed": 4.5,
                "target_point": [9.0, 1.0],
                "command": 4,
                "planning_source": "pred_wp",
            },
            "control": {
                "steer": 0.1,
                "throttle": 0.4,
                "brake": 0.0,
                "planner_type": "only_traj",
                "details": {
                    "selected_branch": "only_traj",
                    "traj_branch": {"steer": 0.1, "throttle": 0.4, "brake": 0.0},
                    "ctrl_branch": {"steer": -0.2, "throttle": 0.3, "brake": 0.0},
                },
            },
            "outcome": {
                "collision": False,
                "lane_invasion": True,
                "route_progress": 0.25,
                "min_ttc": 3.2,
            },
            "ego": {"x": 1.0, "y": 2.0, "yaw": 15.0, "speed": 5.0},
        },
        run_id="tcp_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert set(frame.states) == {
        Stage.VISION,
        Stage.PLANNING,
        Stage.CONTROL,
        Stage.OUTCOME,
    }
    assert Stage.SEMANTIC not in frame.states
    assert frame.states[Stage.VISION].feature == [0.1, 0.2, 0.3]
    assert (
        frame.states[Stage.VISION].model_extra["input_simplification"]
        == "single_front_rgb_256x900"
    )

    planning = frame.states[Stage.PLANNING]
    assert planning.waypoints == [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2], [7.0, 0.2]]
    assert planning.target_speed == 4.5
    assert planning.model_extra["target_point"] == [9.0, 1.0]
    assert planning.model_extra["ego"]["yaw"] == 15.0

    control = frame.states[Stage.CONTROL]
    assert control.steer == 0.1
    assert control.throttle == 0.4
    assert control.brake == 0.0
    assert control.model_extra["planner_type"] == "only_traj"
    assert control.model_extra["details"]["traj_branch"]["steer"] == 0.1
    assert control.model_extra["details"]["ctrl_branch"]["steer"] == -0.2

    outcome = frame.states[Stage.OUTCOME]
    assert outcome.lane_invasion is True
    assert outcome.min_ttc == 3.2


def test_tcp_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = tcp_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="tcp_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0
    assert Stage.SEMANTIC not in frame.states
    assert "min_ttc" not in frame_record["states"]["outcome"]


def test_build_tcp_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_tcp_run_metadata(
        run_id="tcp_Town10HD_Opt_spawn0_dest10_clean_seed42",
        scenario_id="Town10HD_Opt_spawn0_dest10",
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
    assert metadata.model_id == "tcp"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.stress_type is None


def test_tcp_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "tcp_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "tcp_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_tcp_run_metadata(clean_run_id, scenario_id, "clean", None, 0, 42)
    stress_metadata = build_tcp_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )
    clean_frames = [
        tcp_record_to_sd2(_synthetic_record(idx, feature_delta=0.0), clean_run_id)
        for idx in range(4)
    ]
    stress_frames = [
        tcp_record_to_sd2(_synthetic_record(idx, feature_delta=0.1), stress_run_id)
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    paired = pair_runs(load_run_jsonl(clean_path), load_run_jsonl(stress_path))

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"tcp:{scenario_id}:42:0"


def test_tcp_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "TCP.model" not in sys.modules
    assert "team_code.tcp_b2d_agent" not in sys.modules


def _synthetic_record(frame_idx: int, feature_delta: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "vision": {
            "image_mean": 0.4,
            "image_std": 0.1,
            "feature": [1.0 + feature_delta, 0.0, 0.0],
            "feature_source": "mean_pooled_perception_feature",
        },
        "planning": {
            "waypoints": [
                [float(frame_idx), 0.0],
                [float(frame_idx + 1), 0.1],
                [float(frame_idx + 2), 0.2],
                [float(frame_idx + 3), 0.3],
            ],
            "target_speed": 4.0,
            "target_point": [5.0, 0.0],
        },
        "control": {
            "steer": 0.0,
            "throttle": 0.3,
            "brake": 0.0,
            "planner_type": "only_traj",
            "details": {
                "selected_branch": "only_traj",
                "traj_branch": {"steer": 0.0, "throttle": 0.3, "brake": 0.0},
                "ctrl_branch": {"steer": 0.1, "throttle": 0.2, "brake": 0.0},
            },
        },
        "outcome": {
            "collision": False,
            "lane_invasion": False,
            "route_progress": frame_idx / 3,
            "min_ttc": None,
        },
        "ego": {
            "x": float(frame_idx),
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "speed": 4.0,
        },
    }
