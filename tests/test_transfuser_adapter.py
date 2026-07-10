from __future__ import annotations

import sys

import pytest

from sd2.adapters.jsonl_adapter import load_run_jsonl
from sd2.adapters.transfuser_adapter import (
    build_transfuser_run_metadata,
    transfuser_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.core.run import pair_runs
from sd2.core.schema import FrameLog, RunMetadata
from sd2.core.stage import Stage


def test_transfuser_record_to_sd2_validates_all_observed_stages() -> None:
    frame_record = transfuser_record_to_sd2(
        {
            "frame_idx": 2,
            "timestamp": 0.1,
            "vision": {
                "image_mean": 0.42,
                "image_std": 0.11,
                "feature": [0.1, 0.2, 0.3],
                "feature_source": "mean_pooled_fused_features",
            },
            "semantic": {
                "rotated_bb": [
                    {
                        "bbox": [
                            [1.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [3.0, 1.0, 0.0],
                            [3.0, -1.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.5, 0.0, 0.0],
                        ],
                        "confidence": 0.9,
                        "brake": 0.0,
                    },
                    (
                        [
                            [4.0, -0.5, 0.0],
                            [4.0, 0.5, 0.0],
                            [5.0, 0.5, 0.0],
                            [5.0, -0.5, 0.0],
                            [4.5, 0.0, 0.0],
                            [5.0, 0.0, 0.0],
                        ],
                        1.0,
                        0.6,
                    ),
                ],
                "bev_seg_summary": {"road": 120, "sidewalk": 7},
            },
            "planning": {
                "pred_wp": [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2]],
                "target_point": [9.0, 1.0],
                "command": 4,
            },
            "control": {"steer": 0.1, "throttle": 0.4, "brake": 0.0},
            "outcome": {
                "collision": False,
                "lane_invasion": True,
                "off_route": True,
                "route_progress": 0.25,
                "min_ttc": 3.2,
            },
            "ego": {"x": 1.0, "y": 2.0, "yaw": 15.0, "speed": 5.0},
        },
        run_id="transfuser_clean",
    )

    assert frame_record["type"] == "frame"
    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert set(frame.states) == {
        Stage.VISION,
        Stage.SEMANTIC,
        Stage.PLANNING,
        Stage.CONTROL,
        Stage.OUTCOME,
    }
    vision = frame.states[Stage.VISION]
    assert vision.feature == [0.1, 0.2, 0.3]
    assert vision.model_extra["image_mean"] == 0.42
    assert vision.model_extra["feature_source"] == "mean_pooled_fused_features"

    semantic = frame.states[Stage.SEMANTIC]
    assert semantic.objects == ["vehicle"]
    assert semantic.critical_object == "vehicle"
    assert semantic.model_extra["num_objects"] == 2
    assert semantic.model_extra["per_class_counts"] == {"vehicle": 2}
    assert semantic.model_extra["object_density_summary"]["vehicle"] == 2
    assert semantic.model_extra["object_density_summary"]["occupied_cells"] == 2
    assert semantic.model_extra["bev_seg_summary"] == {"road": 120, "sidewalk": 7}
    assert len(semantic.model_extra["detections"]) == 2

    planning = frame.states[Stage.PLANNING]
    assert planning.waypoints == [[1.0, 0.0], [3.0, 0.0], [5.0, 0.2]]
    assert planning.target_speed == pytest.approx(4.0)
    assert planning.model_extra["ego"]["yaw"] == 15.0
    assert planning.model_extra["target_point"] == [9.0, 1.0]

    control = frame.states[Stage.CONTROL]
    assert control.steer == 0.1
    assert control.throttle == 0.4
    assert control.brake == 0.0

    outcome = frame.states[Stage.OUTCOME]
    assert outcome.collision is False
    assert outcome.lane_invasion is True
    assert outcome.off_route is True
    assert outcome.route_progress == 0.25
    assert outcome.min_ttc == 3.2


def test_transfuser_record_to_sd2_handles_missing_optional_fields() -> None:
    frame_record = transfuser_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"image_mean": 0.5, "image_std": 0.2, "feature": None},
            "semantic": {},
            "planning": {},
            "control": {"steer": 0.0},
            "outcome": {"route_progress": 2.0, "min_ttc": None},
        },
        run_id="transfuser_clean",
    )

    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.VISION].feature == [0.5, 0.2]
    assert frame.states[Stage.VISION].model_extra["feature_source"] == "image_stats"
    assert frame.states[Stage.SEMANTIC].objects is None
    assert frame.states[Stage.PLANNING].waypoints is None
    assert frame.states[Stage.CONTROL].throttle == 0.0
    assert frame.states[Stage.OUTCOME].route_progress == 1.0
    assert "min_ttc" not in frame_record["states"]["outcome"]


def test_transfuser_control_anti_crawl_marker_is_conditional() -> None:
    marked = _synthetic_record(0, feature_delta=0.0)
    marked["control"]["anti_crawl_applied"] = True
    marked["control"]["applied_throttle"] = 0.6

    marked_frame = transfuser_record_to_sd2(marked, run_id="transfuser_clean")
    marked_control = marked_frame["states"]["control"]

    assert marked_control["anti_crawl_applied"] is True
    assert marked_control["applied_throttle"] == pytest.approx(0.6)

    clean_frame = transfuser_record_to_sd2(
        _synthetic_record(1, feature_delta=0.0),
        run_id="transfuser_clean",
    )
    assert set(clean_frame["states"]["control"]) == {"steer", "throttle", "brake"}


def test_transfuser_record_to_sd2_handles_empty_rotated_bboxes() -> None:
    frame_record = transfuser_record_to_sd2(
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "vision": {"feature": [0.0, 1.0]},
            "semantic": {"rotated_bb": []},
            "planning": {"waypoints": [[0.0, 0.0], [0.5, 0.0]]},
            "control": {},
            "outcome": {},
        },
        run_id="transfuser_clean",
    )
    frame = FrameLog.model_validate(
        {key: value for key, value in frame_record.items() if key != "type"}
    )

    assert frame.states[Stage.SEMANTIC].objects == []
    assert frame.states[Stage.SEMANTIC].model_extra["num_objects"] == 0
    assert frame.states[Stage.SEMANTIC].model_extra["object_density_summary"][
        "occupied_cells"
    ] == 0


def test_build_transfuser_run_metadata_validates_run_metadata() -> None:
    metadata_record = build_transfuser_run_metadata(
        run_id="transfuser_Town10HD_Opt_spawn0_dest10_clean_seed42",
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
    assert metadata.model_id == "transfuser"
    assert metadata.scenario_id == "Town10HD_Opt_spawn0_dest10"
    assert metadata.condition == "clean"
    assert metadata.stress_type is None


def test_transfuser_jsonl_round_trips_and_pairs(tmp_path) -> None:
    scenario_id = "Town10HD_Opt_spawn0_dest10"
    clean_run_id = "transfuser_Town10HD_Opt_spawn0_dest10_clean_seed42"
    stress_run_id = "transfuser_Town10HD_Opt_spawn0_dest10_gaussian_noise_s3_seed42"
    clean_metadata = build_transfuser_run_metadata(
        clean_run_id,
        scenario_id,
        "clean",
        None,
        0,
        42,
    )
    stress_metadata = build_transfuser_run_metadata(
        stress_run_id,
        scenario_id,
        "stress",
        "gaussian_noise",
        3,
        42,
    )

    clean_frames = [
        transfuser_record_to_sd2(_synthetic_record(idx, feature_delta=0.0), clean_run_id)
        for idx in range(4)
    ]
    stress_frames = [
        transfuser_record_to_sd2(_synthetic_record(idx, feature_delta=0.1), stress_run_id)
        for idx in range(4)
    ]

    clean_path = tmp_path / "clean.jsonl"
    stress_path = tmp_path / "stress.jsonl"
    write_sd2_jsonl(clean_path, clean_metadata, clean_frames)
    write_sd2_jsonl(stress_path, stress_metadata, stress_frames)

    clean_run = load_run_jsonl(clean_path)
    stress_run = load_run_jsonl(stress_path)
    paired = pair_runs(clean_run, stress_run)

    assert paired.summary.paired_count == 4
    assert paired.summary.skipped_count == 0
    assert paired.pairs[0].pair_key == f"transfuser:{scenario_id}:42:0"


def test_transfuser_adapter_does_not_import_heavy_runtime_modules() -> None:
    assert "carla" not in sys.modules
    assert "torch" not in sys.modules
    assert "timm" not in sys.modules
    assert "model" not in sys.modules
    assert "team_code_transfuser.model" not in sys.modules


def _synthetic_record(frame_idx: int, feature_delta: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx * 0.05,
        "vision": {
            "image_mean": 0.4,
            "image_std": 0.1,
            "feature": [1.0 + feature_delta, 0.0, 0.0],
        },
        "semantic": {
            "rotated_bb": [
                [
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, -1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.5, 0.0, 0.0],
                ]
            ],
        },
        "planning": {
            "waypoints": [[float(frame_idx), 0.0], [float(frame_idx + 1), 0.1]],
            "target_speed": 4.0,
        },
        "control": {"steer": 0.0, "throttle": 0.3, "brake": 0.0},
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
