"""Labeled synthetic fault-injection run generation for SD2 benchmarks."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd2.core.stage import Stage


FAULT_STAGES: tuple[Stage, ...] = (
    Stage.VISION,
    Stage.SEMANTIC,
    Stage.REASONING,
    Stage.PLANNING,
    Stage.CONTROL,
)
_NOVEL_OBJECTS = [
    "construction_barrel",
    "fallen_cargo",
    "emergency_vehicle",
    "temporary_sign",
    "road_worker",
    "traffic_cone",
]


@dataclass(frozen=True)
class SyntheticRunPair:
    """One labeled clean/stress synthetic run pair."""

    run_id: str
    target_stage: Stage
    sample_index: int
    onset_frame: int
    frame_count: int
    pair_seed: int
    clean_metadata: dict[str, Any]
    stress_metadata: dict[str, Any]
    clean_frames: list[dict[str, Any]]
    stress_frames: list[dict[str, Any]]
    params: dict[str, Any]


@dataclass(frozen=True)
class SyntheticPairPaths:
    """Filesystem paths written for one synthetic run pair."""

    clean_path: Path
    stress_path: Path
    label_path: Path


def generate_synthetic_pairs(
    n_per_class: int = 20,
    seed: int = 42,
    frame_count: int = 30,
    *,
    profile: str = "realistic",
) -> list[SyntheticRunPair]:
    """Generate deterministic labeled synthetic run pairs.

    ``profile="realistic"`` varies onset, target magnitude, downstream
    propagation, and small run-to-run noise. ``profile="clean_cut"`` keeps the
    same SD2 JSONL path but uses larger margins for high-separability tests.
    """

    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")
    if frame_count < 12:
        raise ValueError("frame_count must be at least 12")
    if profile not in {"realistic", "clean_cut"}:
        raise ValueError("profile must be 'realistic' or 'clean_cut'")

    master_rng = random.Random(seed)
    pairs: list[SyntheticRunPair] = []
    for target_stage in FAULT_STAGES:
        for sample_index in range(n_per_class):
            pair_seed = master_rng.randrange(1, 2_147_483_647)
            pairs.append(
                _build_pair(
                    target_stage=target_stage,
                    sample_index=sample_index,
                    frame_count=frame_count,
                    pair_seed=pair_seed,
                    profile=profile,
                )
            )
    return pairs


def materialize_pair(pair: SyntheticRunPair, output_dir: str | Path) -> SyntheticPairPaths:
    """Write one synthetic pair as clean/stress JSONL plus a label sidecar."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    clean_path = root / "clean_run.jsonl"
    stress_path = root / "stress_run.jsonl"
    label_path = root / "label.json"

    _write_run(clean_path, pair.clean_metadata, pair.clean_frames)
    _write_run(stress_path, pair.stress_metadata, pair.stress_frames)
    label_payload = {
        "run_id": pair.run_id,
        "target_stage": pair.target_stage.value,
        "onset_frame": pair.onset_frame,
        "frame_count": pair.frame_count,
        "pair_seed": pair.pair_seed,
        "params": pair.params,
    }
    label_path.write_text(json.dumps(label_payload, indent=2) + "\n", encoding="utf-8")
    return SyntheticPairPaths(
        clean_path=clean_path,
        stress_path=stress_path,
        label_path=label_path,
    )


def _build_pair(
    target_stage: Stage,
    sample_index: int,
    frame_count: int,
    pair_seed: int,
    profile: str,
) -> SyntheticRunPair:
    rng = random.Random(pair_seed)
    onset_frame = _choose_onset(rng, frame_count, profile)
    route_phase = rng.uniform(-0.6, 0.6)
    curve_amplitude = rng.uniform(0.22, 0.46)
    target_score = _target_score(rng, target_stage, profile)
    downstream_stages = _choose_downstream_stages(rng, target_stage, profile)
    downstream_delays = {
        stage.value: (1 if profile == "clean_cut" else rng.randint(1, 3))
        for stage in downstream_stages
    }
    downstream_scores = {
        stage.value: _downstream_score(rng, target_score, profile)
        for stage in downstream_stages
    }
    run_id = f"synthetic_{target_stage.value}_{sample_index:03d}_seed{pair_seed}"
    clean_run_id = f"{run_id}_clean"
    stress_run_id = f"{run_id}_stress"
    common_metadata = {
        "model_id": "synthetic_sd2",
        "scenario_id": f"fault_{target_stage.value}_{sample_index:03d}",
        "seed": pair_seed,
        "timestamp_start": "2026-01-01T00:00:00",
    }
    clean_metadata = {
        "run_id": clean_run_id,
        "condition": "clean",
        "stress_type": None,
        "severity": 0,
        **common_metadata,
    }
    stress_metadata = {
        "run_id": stress_run_id,
        "condition": "stress",
        "stress_type": f"synthetic_{target_stage.value}_fault",
        "severity": 5,
        **common_metadata,
    }

    clean_frames = [
        _base_clean_frame(
            run_id=clean_run_id,
            frame_idx=frame_idx,
            frame_count=frame_count,
            rng=rng,
            route_phase=route_phase,
            curve_amplitude=curve_amplitude,
        )
        for frame_idx in range(frame_count)
    ]
    stress_frames = [
        _stress_frame(
            clean_frame=clean_frame,
            stress_run_id=stress_run_id,
            target_stage=target_stage,
            onset_frame=onset_frame,
            frame_count=frame_count,
            target_score=target_score,
            downstream_stages=downstream_stages,
            downstream_delays=downstream_delays,
            downstream_scores=downstream_scores,
            rng=rng,
            profile=profile,
        )
        for clean_frame in clean_frames
    ]

    params = {
        "profile": profile,
        "target_score": target_score,
        "downstream_stages": [stage.value for stage in downstream_stages],
        "downstream_delays": downstream_delays,
        "downstream_scores": downstream_scores,
        "route_phase": route_phase,
        "curve_amplitude": curve_amplitude,
    }
    return SyntheticRunPair(
        run_id=run_id,
        target_stage=target_stage,
        sample_index=sample_index,
        onset_frame=onset_frame,
        frame_count=frame_count,
        pair_seed=pair_seed,
        clean_metadata=clean_metadata,
        stress_metadata=stress_metadata,
        clean_frames=clean_frames,
        stress_frames=stress_frames,
        params=params,
    )


def _base_clean_frame(
    run_id: str,
    frame_idx: int,
    frame_count: int,
    rng: random.Random,
    route_phase: float,
    curve_amplitude: float,
) -> dict[str, Any]:
    timestamp = round(frame_idx * 0.1, 3)
    progress = frame_idx / (frame_count - 1)
    near_crossing = 0.35 <= progress <= 0.65
    intent = "slow_down" if near_crossing else "follow_lane"
    critical_object = "pedestrian" if near_crossing else "vehicle"

    objects = ["lane", "vehicle", "traffic_light", "pedestrian", "crosswalk"]
    if progress > 0.72:
        objects = ["lane", "vehicle", "traffic_light", "road_edge", "speed_sign"]

    waypoints = []
    for step in range(5):
        x = frame_idx * 0.55 + step * 1.25
        y = curve_amplitude * math.sin((frame_idx + step) / 5.0 + route_phase)
        y += 0.015 * step
        waypoints.append([round(x, 3), round(y, 3)])

    steer = curve_amplitude * math.cos(frame_idx / 5.0 + route_phase)
    steer += rng.uniform(-0.012, 0.012)
    throttle = 0.88 + 0.02 * math.sin(frame_idx / 7.0 + route_phase)
    brake = 0.0
    if near_crossing:
        throttle -= 0.025
        brake = 0.015

    embedding_seed = [
        1.0 + 0.02 * math.sin(frame_idx / 6.0),
        0.32 + 0.04 * math.cos(frame_idx / 7.0),
        0.18 + 0.03 * math.sin(frame_idx / 4.0 + route_phase),
        0.42,
        0.16 + 0.02 * math.cos(frame_idx / 5.0),
        0.27,
        0.08 + 0.01 * math.sin(frame_idx / 3.0),
        0.35,
    ]
    embedding = [
        round(value + rng.uniform(-0.006, 0.006), 5)
        for value in embedding_seed
    ]

    text = (
        "A pedestrian is near the crosswalk, so the ego vehicle should slow down."
        if near_crossing
        else "The lane ahead is clear, so the ego vehicle should follow the lane."
    )
    target_speed = 4.8 if near_crossing else 5.8

    return {
        "type": "frame",
        "run_id": run_id,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "states": {
            "vision": {
                "image_path": f"frames/{frame_idx:06d}.png",
                "embedding": embedding,
                "noise_level": round(0.015 + rng.uniform(0.0, 0.006), 4),
            },
            "semantic": {
                "objects": objects,
                "critical_object": critical_object,
                "traffic_light_state": "green",
                "lane_state": "centered",
            },
            "reasoning": {
                "text": text,
                "intent": intent,
                "critical_object_mentioned": True,
            },
            "planning": {
                "waypoints": waypoints,
                "target_speed": round(target_speed, 3),
                "selected_maneuver": intent,
            },
            "control": {
                "steer": round(max(-1.0, min(1.0, steer)), 3),
                "throttle": round(max(0.0, min(1.0, throttle)), 3),
                "brake": round(brake, 3),
            },
            "outcome": {
                "collision": False,
                "lane_invasion": False,
                "route_progress": round(progress, 3),
                "driving_score": 1.0,
                "min_ttc": round(4.5 - 0.8 * progress, 3),
            },
        },
    }


def _stress_frame(
    clean_frame: dict[str, Any],
    stress_run_id: str,
    target_stage: Stage,
    onset_frame: int,
    frame_count: int,
    target_score: float,
    downstream_stages: list[Stage],
    downstream_delays: dict[str, int],
    downstream_scores: dict[str, float],
    rng: random.Random,
    profile: str,
) -> dict[str, Any]:
    frame = copy.deepcopy(clean_frame)
    frame["run_id"] = stress_run_id
    frame_idx = int(frame["frame_idx"])
    frame["states"]["vision"]["image_path"] = f"stress_frames/{frame_idx:06d}.png"

    if frame_idx >= onset_frame:
        _apply_pair_jitter(frame, clean_frame, rng, scale=0.0 if profile == "clean_cut" else 1.0)
        target_frame_score = _oscillating_score(
            target_score,
            frame_idx - onset_frame,
            lower=0.72,
            upper=1.0,
        )
        _apply_stage_deviation(
            frame=frame,
            clean_frame=clean_frame,
            stage=target_stage,
            score=target_frame_score,
            target=True,
        )

        for stage in downstream_stages:
            start = onset_frame + downstream_delays[stage.value]
            if frame_idx < start:
                continue
            ramp = min(1.0, 0.62 + 0.16 * (frame_idx - start))
            score = downstream_scores[stage.value] * ramp
            score = min(score, max(0.12, target_frame_score - 0.12))
            _apply_stage_deviation(
                frame=frame,
                clean_frame=clean_frame,
                stage=stage,
                score=score,
                target=False,
            )

    _apply_driving_failure(frame, frame_count)
    return frame


def _apply_pair_jitter(
    frame: dict[str, Any],
    clean_frame: dict[str, Any],
    rng: random.Random,
    scale: float,
) -> None:
    if scale <= 0.0:
        return

    states = frame["states"]
    clean_states = clean_frame["states"]
    vision_score = rng.uniform(0.002, 0.018) * scale
    states["vision"]["embedding"] = _rotated_embedding(
        clean_states["vision"]["embedding"],
        vision_score,
    )
    states["vision"]["noise_level"] = round(
        float(clean_states["vision"].get("noise_level", 0.015))
        + rng.uniform(0.0, 0.006) * scale,
        4,
    )

    if rng.random() < 0.35:
        states["reasoning"]["text"] = (
            str(clean_states["reasoning"]["text"]) + " The maneuver remains smooth."
        )

    jittered_waypoints = []
    for point in clean_states["planning"]["waypoints"]:
        jittered_waypoints.append(
            [
                round(float(point[0]) + rng.uniform(-0.015, 0.015) * scale, 3),
                round(float(point[1]) + rng.uniform(-0.035, 0.035) * scale, 3),
            ]
        )
    states["planning"]["waypoints"] = jittered_waypoints
    states["planning"]["target_speed"] = round(
        float(clean_states["planning"]["target_speed"]) + rng.uniform(-0.05, 0.05) * scale,
        3,
    )

    control = states["control"]
    clean_control = clean_states["control"]
    control["steer"] = round(
        max(-1.0, min(1.0, float(clean_control["steer"]) + rng.uniform(-0.018, 0.018) * scale)),
        3,
    )
    control["throttle"] = round(
        max(0.0, min(1.0, float(clean_control["throttle"]) + rng.uniform(-0.015, 0.015) * scale)),
        3,
    )
    control["brake"] = round(
        max(0.0, min(1.0, float(clean_control["brake"]) + rng.uniform(0.0, 0.012) * scale)),
        3,
    )


def _apply_stage_deviation(
    frame: dict[str, Any],
    clean_frame: dict[str, Any],
    stage: Stage,
    score: float,
    target: bool,
) -> None:
    clean_state = clean_frame["states"][stage.value]
    state = frame["states"][stage.value]

    if stage == Stage.VISION:
        state["embedding"] = _rotated_embedding(clean_state["embedding"], score)
        state["noise_level"] = round(0.22 + 0.5 * score, 4)
    elif stage == Stage.SEMANTIC:
        _apply_semantic_deviation(state, clean_state, score, target)
    elif stage == Stage.REASONING:
        _apply_reasoning_deviation(state, clean_state, score, target)
    elif stage == Stage.PLANNING:
        _apply_planning_deviation(state, clean_state, score, target)
    elif stage == Stage.CONTROL:
        _apply_control_deviation(state, clean_state, score, target)


def _apply_semantic_deviation(
    state: dict[str, Any],
    clean_state: dict[str, Any],
    score: float,
    target: bool,
) -> None:
    clean_objects = list(clean_state["objects"])
    if score >= 0.7:
        state["objects"] = _NOVEL_OBJECTS[:4]
    elif score >= 0.55:
        state["objects"] = clean_objects[:2]
    elif score >= 0.35:
        state["objects"] = clean_objects[:3]
    else:
        state["objects"] = clean_objects[:4]

    if target or score >= 0.5:
        state["critical_object"] = "construction_barrel"
        state["traffic_light_state"] = "red"
    state["lane_state"] = "uncertain_left_edge" if score >= 0.35 else "centered"


def _apply_reasoning_deviation(
    state: dict[str, Any],
    clean_state: dict[str, Any],
    score: float,
    target: bool,
) -> None:
    if score >= 0.7:
        state["text"] = (
            "The route appears empty and the obstacle is irrelevant, so continue "
            "assertively through the conflict zone."
        )
        state["intent"] = "maintain_speed"
        state["critical_object_mentioned"] = False
    elif score >= 0.45:
        state["text"] = (
            "The lane appears mostly open, so proceed while monitoring traffic."
        )
        state["intent"] = "maintain_speed"
        state["critical_object_mentioned"] = clean_state.get("critical_object_mentioned", True)
    else:
        state["text"] = str(clean_state["text"]) + " Continue smoothly."
        state["intent"] = clean_state["intent"]
        state["critical_object_mentioned"] = clean_state.get("critical_object_mentioned", True)
    if target and score >= 0.7:
        state["decision_text"] = "No hazard requires braking."


def _apply_planning_deviation(
    state: dict[str, Any],
    clean_state: dict[str, Any],
    score: float,
    target: bool,
) -> None:
    ade = 5.0 * score
    direction = -1.0 if target else 1.0
    shifted = []
    for step, point in enumerate(clean_state["waypoints"]):
        lateral = direction * (ade + 0.08 * step)
        forward = 0.25 * score * step if target else 0.08 * score * step
        shifted.append(
            [
                round(float(point[0]) + forward, 3),
                round(float(point[1]) + lateral, 3),
            ]
        )
    state["waypoints"] = shifted
    state["target_speed"] = round(
        max(0.0, float(clean_state["target_speed"]) + (3.0 if target else 1.2) * score),
        3,
    )
    state["selected_maneuver"] = "swerve_left" if target else "offset_follow"


def _apply_control_deviation(
    state: dict[str, Any],
    clean_state: dict[str, Any],
    score: float,
    target: bool,
) -> None:
    clean_steer = float(clean_state["steer"])
    clean_throttle = float(clean_state["throttle"])
    clean_brake = float(clean_state["brake"])

    if target or score >= 0.7:
        state["steer"] = -1.0 if clean_steer >= 0.0 else 1.0
        state["throttle"] = 0.0 if clean_throttle >= 0.5 else 1.0
        state["brake"] = 1.0 if clean_brake <= 0.5 else 0.0
        return

    steer_delta = min(0.75, 1.35 * score)
    throttle_delta = min(0.38, score)
    brake_delta = min(0.42, score)
    state["steer"] = round(_bounded_shift(clean_steer, steer_delta, -1.0, 1.0), 3)
    state["throttle"] = round(_bounded_shift(clean_throttle, -throttle_delta, 0.0, 1.0), 3)
    state["brake"] = round(_bounded_shift(clean_brake, brake_delta, 0.0, 1.0), 3)


def _apply_driving_failure(frame: dict[str, Any], frame_count: int) -> None:
    frame_idx = int(frame["frame_idx"])
    failure_start = max(0, frame_count - 4)
    collision_start = max(0, frame_count - 2)
    outcome = frame["states"]["outcome"]
    if frame_idx >= failure_start:
        denom = max(1, frame_count - 1 - failure_start)
        severity = (frame_idx - failure_start) / denom
        outcome["lane_invasion"] = True
        outcome["route_progress"] = round(max(0.0, float(outcome["route_progress"]) - 0.07 - 0.05 * severity), 3)
        outcome["driving_score"] = round(max(0.0, 0.86 - 0.22 * severity), 3)
        outcome["min_ttc"] = round(max(0.1, float(outcome["min_ttc"]) - 2.0 * severity), 3)
    if frame_idx >= collision_start:
        outcome["collision"] = True
        outcome["driving_score"] = round(min(float(outcome["driving_score"]), 0.42), 3)
        outcome["min_ttc"] = 0.1


def _choose_onset(rng: random.Random, frame_count: int, profile: str) -> int:
    if profile == "clean_cut":
        return max(4, frame_count // 3)
    lower = max(5, frame_count // 4)
    upper = max(lower, frame_count - 8)
    return rng.randint(lower, upper)


def _target_score(rng: random.Random, target_stage: Stage, profile: str) -> float:
    if profile == "clean_cut":
        return 0.96
    if target_stage == Stage.CONTROL:
        return rng.uniform(0.80, 0.96)
    return rng.uniform(0.74, 0.94)


def _choose_downstream_stages(
    rng: random.Random,
    target_stage: Stage,
    profile: str,
) -> list[Stage]:
    candidates = [
        stage
        for stage in FAULT_STAGES
        if stage.index() > target_stage.index()
    ]
    if profile == "clean_cut":
        return candidates
    chosen = [stage for stage in candidates if rng.random() < 0.65]
    if candidates and not chosen:
        chosen = [rng.choice(candidates)]
    return chosen


def _downstream_score(rng: random.Random, target_score: float, profile: str) -> float:
    if profile == "clean_cut":
        return min(0.58, target_score - 0.18)
    upper = min(0.64, target_score - 0.10)
    return rng.uniform(0.28, max(0.32, upper))


def _oscillating_score(score: float, offset: int, lower: float, upper: float) -> float:
    varied = score + 0.025 * math.sin(offset * 0.9)
    return max(lower, min(upper, varied))


def _rotated_embedding(clean_embedding: list[float], distance: float) -> list[float]:
    vector = [float(value) for value in clean_embedding]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector = [1.0] + [0.0 for _ in vector[1:]]
        norm = 1.0
    unit = [value / norm for value in vector]

    axis = [0.0 for _ in unit]
    axis[0] = 1.0
    if abs(unit[0]) > 0.90 and len(axis) > 1:
        axis[0] = 0.0
        axis[1] = 1.0
    dot = sum(a * b for a, b in zip(axis, unit))
    orthogonal = [a - dot * b for a, b in zip(axis, unit)]
    orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
    if orthogonal_norm == 0.0:
        orthogonal = [0.0 for _ in unit]
        orthogonal[-1] = 1.0
        orthogonal_norm = 1.0
    orthogonal = [value / orthogonal_norm for value in orthogonal]

    cosine = max(-1.0, min(1.0, 1.0 - distance))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    rotated = [
        norm * (cosine * u_value + sine * o_value)
        for u_value, o_value in zip(unit, orthogonal)
    ]
    return [round(value, 5) for value in rotated]


def _bounded_shift(value: float, delta: float, lower: float, upper: float) -> float:
    candidate = value + delta
    if lower <= candidate <= upper:
        return candidate
    return max(lower, min(upper, value - delta))


def _write_run(path: Path, metadata: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "run_metadata", **metadata}, separators=(",", ":")) + "\n")
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
