import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import _carla_e2e_common as e2e
from sd2.adapters.aim_adapter import aim_record_to_sd2
from sd2.adapters.carla_adapter import write_sd2_jsonl
from sd2.adapters.interfuser_adapter import interfuser_record_to_sd2
from sd2.adapters.neat_adapter import neat_record_to_sd2
from sd2.adapters.tcp_adapter import tcp_record_to_sd2
from sd2.adapters.transfuser_adapter import transfuser_record_to_sd2
from sd2.analysis.intervention import (
    MIN_OUTCOME_EFFECT,
    SINGLE_INPUT_REASON,
    run_intervention_analysis,
    run_single_run_share_analysis,
)
from sd2.core.schema import FrameLog
from sd2.core.stage import Stage


@pytest.mark.parametrize(
    ("adapter", "model_id"),
    [
        (interfuser_record_to_sd2, "interfuser"),
        (neat_record_to_sd2, "neat"),
        (transfuser_record_to_sd2, "transfuser"),
        (aim_record_to_sd2, "aim"),
        (tcp_record_to_sd2, "tcp"),
    ],
)
def test_intervention_block_round_trips_through_adapters(adapter, model_id: str) -> None:
    frame_record = adapter(_synthetic_extracted_frame(), run_id=f"{model_id}_run")
    payload = {key: value for key, value in frame_record.items() if key != "type"}
    frame = FrameLog.model_validate(payload)

    intervention = frame.states[Stage.INTERVENTION]
    assert intervention.stage == "planning"
    assert intervention.direction == "restore"
    assert intervention.applied_source == "clean_forward"
    assert intervention.control_from_stress_forward == {
        "steer": 0.0,
        "throttle": 0.2,
        "brake": 0.0,
    }
    assert intervention.planning_waypoints_clean_forward == [[1.0, 0.0], [2.0, 0.0]]
    assert intervention.semantic_clean_forward == {"red_light_occ": 0.0}


def test_intervention_analysis_computes_outcome_recovery(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.4])
    intervened = _write_run(
        tmp_path / "restore_planning.jsonl",
        "interfuser",
        "stress",
        [0.7],
        intervention_stage="planning",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "planning", "status": "failure_detected"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    assert result["outcome_recovery"]["raw"] == pytest.approx(0.5)
    assert result["outcome_recovery"]["clipped"] == pytest.approx(0.5)
    assert result["outcome_recovery"]["threshold"] == pytest.approx(MIN_OUTCOME_EFFECT)
    assert result["outcome_recovery"]["threshold_source"] == "default"
    assert result["outcome_recovery"]["null_reason"] is None
    assert result["agreement_with_sd2"]["agrees"] is True


def test_intervention_analysis_nulls_recovery_when_stress_does_not_degrade(
    tmp_path: Path,
) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [0.5])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.6])
    intervened = _write_run(
        tmp_path / "restore_semantic.jsonl",
        "interfuser",
        "stress",
        [0.6],
        intervention_stage="semantic",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "semantic", "status": "failure_detected"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    assert result["outcome_recovery"]["raw"] is None
    assert result["outcome_recovery"]["clipped"] is None
    assert result["outcome_recovery"]["denominator"] == pytest.approx(-0.1)
    assert result["outcome_recovery"]["null_reason"] == "stress_did_not_degrade_outcome"


def test_intervention_analysis_nulls_recovery_below_default_noise_floor(
    tmp_path: Path,
) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [0.53])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.50])
    intervened = _write_run(
        tmp_path / "restore_semantic.jsonl",
        "interfuser",
        "stress",
        [0.52],
        intervention_stage="semantic",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "semantic", "status": "failure_detected"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    assert result["outcome_recovery"]["raw"] is None
    assert result["outcome_recovery"]["clipped"] is None
    assert result["outcome_recovery"]["denominator"] == pytest.approx(0.03)
    assert result["outcome_recovery"]["threshold"] == pytest.approx(MIN_OUTCOME_EFFECT)
    assert result["outcome_recovery"]["threshold_source"] == "default"
    assert result["outcome_recovery"]["null_reason"] == "effect_below_noise_floor"


def test_intervention_analysis_nulls_recovery_for_missing_outcome(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.4])
    intervened = _write_run(
        tmp_path / "restore_semantic.jsonl",
        "interfuser",
        "stress",
        [None],
        intervention_stage="semantic",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "semantic", "status": "failure_detected"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    assert result["outcome_recovery"]["raw"] is None
    assert result["outcome_recovery"]["clipped"] is None
    assert result["outcome_recovery"]["denominator"] == pytest.approx(0.6)
    assert result["outcome_recovery"]["null_reason"] == "missing_outcome"


def test_intervention_analysis_uses_clean_replicate_noise_floor(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.85])
    intervened = _write_run(
        tmp_path / "restore_planning.jsonl",
        "interfuser",
        "stress",
        [0.9],
        intervention_stage="planning",
    )
    replicate_a = _write_run(tmp_path / "clean_a.jsonl", "interfuser", "clean", [0.9])
    replicate_b = _write_run(tmp_path / "clean_b.jsonl", "interfuser", "clean", [1.0])
    replicate_c = _write_run(tmp_path / "clean_c.jsonl", "interfuser", "clean", [1.1])
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "planning"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
        clean_replicates=[replicate_a, replicate_b, replicate_c],
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    recovery = result["outcome_recovery"]
    assert recovery["raw"] is None
    assert recovery["clipped"] is None
    assert recovery["denominator"] == pytest.approx(0.15)
    assert recovery["threshold"] == pytest.approx(1.96 * 0.1)
    assert recovery["threshold_source"] == "clean_replicates"
    assert recovery["null_reason"] == "effect_below_noise_floor"


def test_control_decomposition_null_for_transfuser(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "transfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "transfuser", "stress", [0.3])
    intervened = _write_run(
        tmp_path / "restore_planning.jsonl",
        "transfuser",
        "stress",
        [0.6],
        intervention_stage="planning",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "planning"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    assert result["control_decomposition"] is None
    assert result["control_decomposition_reason"] == SINGLE_INPUT_REASON


def test_control_decomposition_populated_for_interfuser(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.2])
    intervened = _write_run(
        tmp_path / "restore_planning.jsonl",
        "interfuser",
        "stress",
        [0.6],
        intervention_stage="planning",
        applied_control={"steer": 0.5, "throttle": 0.0, "brake": 0.0},
        clean_forward_control={"steer": 1.0, "throttle": 0.0, "brake": 0.0},
        stress_forward_control={"steer": 0.0, "throttle": 0.0, "brake": 0.0},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "planning"}),
        encoding="utf-8",
    )

    run_intervention_analysis(
        baseline_clean=clean,
        stress=stress,
        intervened=intervened,
        config_path="configs/mvp.yaml",
        output_dir=output_dir,
    )

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    decomposition = result["control_decomposition"]
    assert decomposition["intervened_stage"] == "planning"
    assert decomposition["mean_abs_clean_vs_stress_control"]["l1"] == pytest.approx(1.0)
    assert decomposition["mean_abs_stage_intervention_effect"]["l1"] == pytest.approx(0.5)
    assert decomposition["share_of_total_control_change"] == pytest.approx(0.5)


def test_single_run_shares_use_one_common_denominator(tmp_path: Path) -> None:
    none_run = _write_none_run_with_hybrids(tmp_path / "none.jsonl")
    output_dir = tmp_path / "out"

    run_single_run_share_analysis(none_run=none_run, output_dir=output_dir)

    result = json.loads((output_dir / "intervention.json").read_text(encoding="utf-8"))
    shares = result["control_decomposition"]["single_run_shares"]
    assert shares["common_denominator"]["l1"] == pytest.approx(2.0)
    assert shares["denominator_frame_count"] == 2
    assert shares["stages"]["planning"]["mean_abs_hybrid_effect"]["l1"] == pytest.approx(1.0)
    assert shares["stages"]["planning"]["share_of_total_control_change"] == pytest.approx(0.5)
    assert shares["stages"]["semantic"]["mean_abs_hybrid_effect"]["l1"] == pytest.approx(0.5)
    assert shares["stages"]["semantic"]["share_of_total_control_change"] == pytest.approx(0.25)
    assert "need not sum to 1" in shares["non_additivity_note"]


def test_scene_sidecar_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"

    e2e.write_scene_sidecar(
        output,
        town="Town10HD_Opt",
        spawn_index=0,
        dest_index=50,
        seed=7,
        vehicles={"requested": 40, "spawned": 37},
        walkers={"requested": 20, "spawned": 18},
        frames=400,
        delta=0.05,
        route_length_meters=1234.5,
    )

    payload = json.loads(
        Path(f"{output}.scene.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "town": "Town10HD_Opt",
        "spawn_index": 0,
        "dest_index": 50,
        "seed": 7,
        "vehicles": {"requested": 40, "spawned": 37},
        "walkers": {"requested": 20, "spawned": 18},
        "frames": 400,
        "delta": 0.05,
        "route_length_meters": 1234.5,
    }


def test_cli_intervention_writes_outputs(tmp_path: Path) -> None:
    clean = _write_run(tmp_path / "clean.jsonl", "interfuser", "clean", [1.0])
    stress = _write_run(tmp_path / "stress.jsonl", "interfuser", "stress", [0.4])
    intervened = _write_run(
        tmp_path / "restore_planning.jsonl",
        "interfuser",
        "stress",
        [0.7],
        intervention_stage="planning",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnosis.json").write_text(
        json.dumps({"primary_failure_stage": "planning"}),
        encoding="utf-8",
    )
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sd2.cli",
            "intervention",
            "--baseline-clean",
            str(clean),
            "--stress",
            str(stress),
            "--intervened",
            str(intervened),
            "--config",
            "configs/mvp.yaml",
            "--output",
            str(output_dir),
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "intervention.json").is_file()
    assert (output_dir / "intervention.md").is_file()


@pytest.mark.parametrize(
    ("model_id", "stage", "message"),
    [
        ("transfuser", "semantic", "detection head is off the causal path"),
        ("aim", "semantic", "no semantic head"),
        ("tcp", "semantic", "no semantic head"),
        ("cilrs", "planning", "no planning stage"),
    ],
)
def test_unsupported_intervention_combinations_raise_clear_errors(
    model_id: str,
    stage: str,
    message: str,
) -> None:
    args = argparse.Namespace(
        intervene_stage=stage,
        intervene_direction="restore",
        stress="gaussian_noise",
        stress_severity=3,
    )

    with pytest.raises(ValueError, match=message):
        e2e.InterventionPolicy.from_args(args, model_id)


def _synthetic_extracted_frame() -> dict:
    return {
        "frame_idx": 0,
        "timestamp": 0.0,
        "vision": {"image_mean": 0.4, "image_std": 0.1, "feature": [0.1, 0.2]},
        "semantic": {"objects": ["vehicle"], "red_light_occ": 0.2},
        "planning": {"waypoints": [[0.0, 0.0], [1.0, 0.0]], "target_speed": 1.0},
        "control": {"steer": 0.1, "throttle": 0.3, "brake": 0.0},
        "outcome": {"route_progress": 0.4, "collision": False, "lane_invasion": False},
        "intervention": {
            "stage": "planning",
            "direction": "restore",
            "applied_source": "clean_forward",
            "control_from_stress_forward": {"steer": 0.0, "throttle": 0.2, "brake": 0.0},
            "control_from_clean_forward": {"steer": 0.1, "throttle": 0.3, "brake": 0.0},
            "planning_waypoints_clean_forward": [[1.0, 0.0], [2.0, 0.0]],
            "semantic_clean_forward": {"red_light_occ": 0.0},
        },
    }


def _write_run(
    path: Path,
    model_id: str,
    condition: str,
    route_progress: list[float | None],
    *,
    intervention_stage: str | None = None,
    applied_control: dict[str, float] | None = None,
    clean_forward_control: dict[str, float] | None = None,
    stress_forward_control: dict[str, float] | None = None,
) -> Path:
    metadata = {
        "type": "run_metadata",
        "run_id": f"{model_id}_{condition}_{path.stem}",
        "model_id": model_id,
        "scenario_id": "TownTest_routeA",
        "condition": condition,
        "stress_type": None if condition == "clean" else "gaussian_noise",
        "severity": 0 if condition == "clean" else 3,
        "seed": 7,
    }
    frames = []
    for idx, progress in enumerate(route_progress):
        control = applied_control or {"steer": 0.2, "throttle": 0.1, "brake": 0.0}
        states = {
            "vision": {"feature": [0.1 + idx, 0.2]},
            "semantic": {"objects": ["vehicle"], "red_light_occ": 0.0},
            "planning": {"waypoints": [[0.0, 0.0], [1.0, 0.0]], "target_speed": 1.0},
            "control": control,
            "outcome": {
                "collision": False,
                "lane_invasion": idx == 0 and condition == "stress",
            },
        }
        if progress is not None:
            states["outcome"]["route_progress"] = progress
        if intervention_stage is not None:
            states["intervention"] = {
                "stage": intervention_stage,
                "direction": "restore",
                "applied_source": "clean_forward",
                "control_from_stress_forward": stress_forward_control
                or {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
                "control_from_clean_forward": clean_forward_control
                or {"steer": 0.2, "throttle": 0.1, "brake": 0.0},
                "planning_waypoints_clean_forward": [[1.0, 0.0], [2.0, 0.0]],
            }
        frames.append(
            {
                "type": "frame",
                "run_id": metadata["run_id"],
                "frame_idx": idx,
                "timestamp": float(idx),
                "states": states,
            }
        )
    write_sd2_jsonl(path, metadata, frames)
    if route_progress[-1] is not None:
        assert not math.isnan(route_progress[-1])
    return path


def _write_none_run_with_hybrids(path: Path) -> Path:
    metadata = {
        "type": "run_metadata",
        "run_id": "interfuser_stress_none",
        "model_id": "interfuser",
        "scenario_id": "TownTest_routeA",
        "condition": "stress",
        "stress_type": "gaussian_noise",
        "severity": 3,
        "seed": 7,
    }
    controls = [
        {
            "stress": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
            "clean": {"steer": 1.0, "throttle": 0.0, "brake": 0.0},
            "planning": {"steer": 0.5, "throttle": 0.0, "brake": 0.0},
            "semantic": {"steer": 0.25, "throttle": 0.0, "brake": 0.0},
        },
        {
            "stress": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
            "clean": {"steer": 0.0, "throttle": 3.0, "brake": 0.0},
            "planning": {"steer": 0.0, "throttle": 1.5, "brake": 0.0},
            "semantic": {"steer": 0.0, "throttle": 0.75, "brake": 0.0},
        },
    ]
    frames = []
    for idx, item in enumerate(controls):
        frames.append(
            {
                "type": "frame",
                "run_id": metadata["run_id"],
                "frame_idx": idx,
                "timestamp": float(idx),
                "states": {
                    "control": item["stress"],
                    "outcome": {
                        "route_progress": 0.2 + idx * 0.1,
                        "collision": False,
                        "lane_invasion": False,
                    },
                    "intervention": {
                        "stage": "none",
                        "direction": None,
                        "applied_source": "stress_forward",
                        "control_from_stress_forward": item["stress"],
                        "control_from_clean_forward": item["clean"],
                        "control_hybrid_planning_clean": item["planning"],
                        "control_hybrid_semantic_clean": item["semantic"],
                    },
                },
            }
        )
    write_sd2_jsonl(path, metadata, frames)
    return path
