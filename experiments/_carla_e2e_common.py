"""Shared CARLA recorder helpers for classic E2E baselines.

This module intentionally avoids importing ``carla`` or ``torch`` at import
time. Model recorder scripts import heavy runtime packages inside guarded
``_import_runtime_modules`` functions and pass those modules into this helper.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Mapping

from sd2.stressors import ImageStressor, build_stressor, validate_severity


REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_PYTHON_API = (
    REPO_ROOT / "external" / "Carla" / "CARLA_0.9.16" / "PythonAPI" / "carla"
)
DEFAULT_TARGET_SPEED_KMH = 25.0
STRESS_CHOICES = ["none", "gaussian_noise", "motion_blur", "brightness", "fog"]
INTERVENTION_STAGE_CHOICES = ("none", "planning", "semantic")
INTERVENTION_DIRECTION_CHOICES = ("restore", "inject")
INTERVENTION_SUPPORT: dict[str, set[str]] = {
    "interfuser": {"planning", "semantic"},
    "neat": {"planning", "semantic"},
    "transfuser": {"planning"},
    "aim": {"planning"},
    "tcp": {"planning"},
    "cilrs": set(),
}
SINGLE_INPUT_CONTROLLER_MODELS = {"aim", "tcp", "transfuser"}
MULTI_INPUT_CONTROLLER_MODELS = {"interfuser", "neat"}
TRANSFUSER_SEMANTIC_ERROR = (
    "TransFuser semantic intervention is structurally vacuous: the detection "
    "head is off the causal path to control."
)
PLANNING_ONLY_SEMANTIC_ERROR = (
    "{model_label} semantic intervention is unsupported because this recorder "
    "has no semantic head on the causal path to control."
)
CILRS_PLANNING_ERROR = (
    "CILRS planning intervention is unsupported because CILRS has no planning "
    "stage; it regresses control directly."
)
SENSOR_SPEC_BOOKKEEPING_KEYS = frozenset(
    {
        "type",
        "id",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "sensor_tick",
        "reading_frequency",
    }
)
SENSOR_SPEC_ATTRIBUTE_ALIASES = {
    "width": "image_size_x",
    "height": "image_size_y",
}
OPTIONAL_SENSOR_SPEC_ATTRIBUTES = frozenset({"sensor_tick", "reading_frequency"})
RGB_CAMERA_LEADERBOARD_ATTRIBUTES = {
    "lens_circle_multiplier": 3.0,
    "lens_circle_falloff": 3.0,
    "chromatic_aberration_intensity": 0.5,
    "chromatic_aberration_offset": 0,
}
LIDAR_RAY_CAST_SEMANTIC_LEADERBOARD_ATTRIBUTES = {
    "range": 85,
    "rotation_frequency": 10,
    "channels": 64,
    "upper_fov": 10,
    "lower_fov": -30,
    "points_per_second": 600000,
}
LIDAR_RAY_CAST_LEADERBOARD_ATTRIBUTES = {
    **LIDAR_RAY_CAST_SEMANTIC_LEADERBOARD_ATTRIBUTES,
    "atmosphere_attenuation_rate": 0.004,
    "dropoff_general_rate": 0.45,
    "dropoff_intensity_limit": 0.8,
    "dropoff_zero_intensity": 0.4,
}
GNSS_LEADERBOARD_ATTRIBUTES = {
    "noise_alt_bias": 0.0,
    "noise_lat_bias": 0.0,
    "noise_lon_bias": 0.0,
}


@dataclass(frozen=True)
class InterventionPolicy:
    """Counterfactual stage-intervention policy for same-pose dual-forward runs.

    The recorder always logs all-stress and all-clean forward candidates. For
    InterFuser and NEAT, controller-level stage swaps are empirical because the
    controller consumes both planning and semantic outputs. For AIM, TCP, and
    TransFuser, control is a deterministic function of planning and velocity;
    planning restoration therefore restores per-tick control arithmetically.
    Those single-input controllers are only informative at the closed-loop
    outcome level.
    """

    model_id: str
    stage: str
    direction: str | None
    stress_type: str | None
    stress_severity: int

    @classmethod
    def from_args(cls, args: argparse.Namespace, model_id: str) -> "InterventionPolicy":
        stage = str(getattr(args, "intervene_stage", "none") or "none")
        direction = str(getattr(args, "intervene_direction", "restore") or "restore")
        stress_type = getattr(args, "stress", "none")
        severity = int(getattr(args, "stress_severity", 0) or 0)
        policy = cls(
            model_id=str(model_id),
            stage=stage,
            direction=None if stage == "none" else direction,
            stress_type=None if stress_type in (None, "", "none") else str(stress_type),
            stress_severity=0 if stress_type in (None, "", "none") else severity,
        )
        policy.validate()
        return policy

    @property
    def enabled(self) -> bool:
        return self.stage != "none"

    @property
    def is_restore(self) -> bool:
        return self.direction == "restore"

    @property
    def is_inject(self) -> bool:
        return self.direction == "inject"

    @property
    def base_source(self) -> str:
        return "clean_forward" if self.is_inject else "stress_forward"

    @property
    def applied_source(self) -> str:
        if not self.enabled:
            return "stress_forward"
        return "clean_forward" if self.is_restore else "stress_forward"

    @property
    def metadata_condition(self) -> tuple[str, str | None, int]:
        if self.is_inject:
            return "clean", None, 0
        if self.stress_type is None:
            return "clean", None, 0
        return "stress", self.stress_type, self.stress_severity

    def source_for_stage(self, stage: str) -> str:
        normalized_stage = str(stage)
        if not self.enabled:
            return "stress_forward"
        if normalized_stage == self.stage:
            return self.applied_source
        return self.base_source

    def validate(self) -> None:
        normalized_model = self.model_id.lower()
        if self.stage not in INTERVENTION_STAGE_CHOICES:
            allowed = ", ".join(INTERVENTION_STAGE_CHOICES)
            raise ValueError(f"unsupported intervention stage {self.stage!r}; expected one of: {allowed}")
        if self.direction is not None and self.direction not in INTERVENTION_DIRECTION_CHOICES:
            allowed = ", ".join(INTERVENTION_DIRECTION_CHOICES)
            raise ValueError(
                f"unsupported intervention direction {self.direction!r}; expected one of: {allowed}"
            )
        supported = INTERVENTION_SUPPORT.get(normalized_model)
        if supported is None:
            raise ValueError(f"unknown intervention model_id {self.model_id!r}")
        if self.stage == "none":
            return
        if self.stress_type is None:
            raise ValueError(
                "--intervene-stage requires --stress to name the stressor used for "
                "the counterfactual forward"
            )
        if self.stage in supported:
            return
        raise ValueError(unsupported_intervention_message(normalized_model, self.stage))

    def config_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "model_id": self.model_id,
            "stage": self.stage,
            "direction": self.direction,
            "stress_type": self.stress_type,
            "stress_severity": self.stress_severity,
            "base_source": self.base_source,
            "applied_source": self.applied_source,
        }
        if self.model_id in SINGLE_INPUT_CONTROLLER_MODELS:
            record["analytic_mediation_caveat"] = (
                "controller consumes planning only; per-tick planning-control "
                "decomposition is analytic, not empirical"
            )
        if self.model_id == "transfuser":
            record["semantic_head_limit"] = (
                "TransFuser's detection head is logged but is off the causal "
                "path to control."
            )
        return record


class AntiCrawlNudger:
    """Applied-throttle nudge for models trapped in a cold-start crawl.

    The RECORDED control stage keeps the model's raw output; only the control
    actually sent to the simulator is overridden, and every nudged frame is
    flagged in the recorded control state so the nudge is auditable offline.
    """

    def __init__(self, args) -> None:
        self._enabled = bool(getattr(args, "anti_crawl", False))
        self.creep_speed = float(getattr(args, "creep_speed", 2.0))
        self.creep_frames = int(getattr(args, "creep_frames", 5))
        self.creep_throttle = float(getattr(args, "creep_throttle", 0.6))
        self.creep_duration = int(getattr(args, "creep_duration", 40))
        self.crawl_counter = 0
        self.burst_remaining = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply(self, control, extracted, current_speed) -> bool:
        """Mutate `control` in place if a nudge is engaged. Return whether it nudged."""
        self.crawl_counter = self.crawl_counter + 1 if current_speed < self.creep_speed else 0
        if self.enabled and self.burst_remaining == 0 and self.crawl_counter >= self.creep_frames:
            self.burst_remaining = self.creep_duration
            self.crawl_counter = 0
        if self.enabled and self.burst_remaining > 0:
            control.throttle = self.creep_throttle
            control.brake = 0.0
            self.burst_remaining -= 1
            if isinstance(extracted.get("control"), dict):
                extracted["control"]["anti_crawl_applied"] = True
                extracted["control"]["applied_throttle"] = self.creep_throttle
            return True
        return False


def unsupported_intervention_message(model_id: str, stage: str) -> str:
    normalized_model = str(model_id).lower()
    normalized_stage = str(stage)
    if normalized_model == "transfuser" and normalized_stage == "semantic":
        return TRANSFUSER_SEMANTIC_ERROR
    if normalized_model in {"aim", "tcp"} and normalized_stage == "semantic":
        return PLANNING_ONLY_SEMANTIC_ERROR.format(model_label=normalized_model.upper())
    if normalized_model == "cilrs" and normalized_stage == "planning":
        return CILRS_PLANNING_ERROR
    if normalized_model == "cilrs" and normalized_stage == "semantic":
        return PLANNING_ONLY_SEMANTIC_ERROR.format(model_label="CILRS")
    supported = ", ".join(sorted(INTERVENTION_SUPPORT.get(normalized_model, set()))) or "none"
    return (
        f"{normalized_model} does not support {normalized_stage!r} intervention; "
        f"supported intervention stages: {supported}"
    )


class SensorBuffer:
    def __init__(self, sensor_ids: list[str]) -> None:
        self._queues: dict[str, Queue[tuple[int, Any]]] = {
            sensor_id: Queue() for sensor_id in sensor_ids
        }

    def callback(self, sensor_id: str, converter: Callable[[Any], Any]) -> Callable[[Any], None]:
        def _callback(data: Any) -> None:
            self._queues[sensor_id].put((int(data.frame), converter(data)))

        return _callback

    def read(self, frame_id: int, timeout: float = 10.0) -> dict[str, tuple[int, Any]]:
        packet: dict[str, tuple[int, Any]] = {}
        for sensor_id, queue in self._queues.items():
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for sensor {sensor_id!r} at CARLA frame {frame_id}"
                    )
                try:
                    sensor_frame, payload = queue.get(timeout=remaining)
                except Empty as exc:
                    raise TimeoutError(
                        f"timed out waiting for sensor {sensor_id!r} at CARLA frame {frame_id}"
                    ) from exc
                if sensor_frame < frame_id:
                    continue
                if sensor_frame > frame_id:
                    raise RuntimeError(
                        f"sensor {sensor_id!r} skipped target frame {frame_id}; "
                        f"received future frame {sensor_frame}"
                    )
                packet[sensor_id] = (sensor_frame, payload)
                break
        return packet


class RouteProgressTracker:
    def __init__(
        self,
        locations: list[Any],
        *,
        search_window_points: int = 60,
        max_lateral_m: float = 15.0,
        off_route_frames: int = 20,
    ) -> None:
        self.locations = locations
        self.search_window_points = max(0, int(search_window_points))
        self.max_lateral_m = float(max_lateral_m)
        self.off_route_frames = max(1, int(off_route_frames))
        self.last_index = 0
        self.initial_remaining = max(self._polyline_distance(locations), 1.0)
        self.last_lateral_error_m: float | None = None
        self.corridor_gate_failed = False
        self.off_route = False
        self._consecutive_gate_failures = 0
        self._max_progress = 0.0

    @property
    def consecutive_gate_failures(self) -> int:
        return self._consecutive_gate_failures

    def reset_initial(self, current_location: Any) -> None:
        self.initial_remaining = max(self.remaining_distance(current_location), 1.0)
        self._max_progress = 0.0
        self._consecutive_gate_failures = 0
        self.corridor_gate_failed = False
        self.off_route = False

    def progress(self, current_location: Any) -> float:
        remaining = self.remaining_distance(current_location)
        current_progress = clamp(1.0 - remaining / self.initial_remaining, 0.0, 1.0)
        self._max_progress = max(self._max_progress, current_progress)
        return self._max_progress

    def remaining_distance(self, current_location: Any) -> float:
        if not self.locations:
            return 0.0

        nearest_index, lateral_error = self._nearest_route_candidate(current_location)
        self.last_lateral_error_m = lateral_error
        if lateral_error <= self.max_lateral_m:
            self.last_index = nearest_index
            self._consecutive_gate_failures = 0
            self.corridor_gate_failed = False
            self.off_route = False
        else:
            self._consecutive_gate_failures += 1
            self.corridor_gate_failed = True
            self.off_route = self._consecutive_gate_failures >= self.off_route_frames

        remaining = distance(current_location, self.locations[self.last_index])
        remaining += self._polyline_distance(self.locations[self.last_index :])
        return remaining

    def _nearest_route_index(self, current_location: Any) -> int:
        nearest_index, _lateral_error = self._nearest_route_candidate(current_location)
        return nearest_index

    def _nearest_route_candidate(self, current_location: Any) -> tuple[int, float]:
        if not self.locations:
            return 0, math.inf
        start = min(self.last_index, len(self.locations) - 1)
        end = min(len(self.locations), start + self.search_window_points + 1)
        candidates = range(start, end)
        nearest_index = min(
            candidates,
            key=lambda idx: distance(current_location, self.locations[idx]),
        )
        return nearest_index, distance(current_location, self.locations[nearest_index])

    @staticmethod
    def _polyline_distance(locations: list[Any]) -> float:
        if len(locations) < 2:
            return 0.0
        return sum(distance(left, right) for left, right in zip(locations, locations[1:]))


def parse_record_args(
    argv: list[str] | None,
    *,
    description: str,
    default_checkpoint: Path | None,
    model_id: str | None = None,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
    checkpoint_required: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    checkpoint_help = None
    if model_id == "interfuser":
        checkpoint_help = (
            "Path to the InterFuser weights. Defaults to the $INTERFUSER_CKPT "
            "environment variable; required when that is unset."
        )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        required=checkpoint_required,
        help=checkpoint_help,
    )
    parser.add_argument("--stress", choices=STRESS_CHOICES, default="none")
    parser.add_argument("--stress-severity", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument(
        "--dest-index",
        type=int,
        default=None,
        help="optional destination spawn point index; omitted preserves the default opposite-spawn route",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=0,
        help="number of deterministic NPC vehicles to request (default 0)",
    )
    parser.add_argument(
        "--num-walkers",
        type=int,
        default=0,
        help="number of deterministic NPC walkers to request (default 0)",
    )
    add_intervention_args(parser)
    # Anti-crawl driving aid for models without a native creep controller
    # (AIM/CILRS/TCP fall into a cold-start crawl limit-cycle). When enabled, the
    # APPLIED throttle is nudged while the ego is crawling so it gets a rolling
    # start; the RECORDED control stage still holds the model's raw output, and
    # each nudged frame is flagged in the control state for transparency.
    parser.add_argument(
        "--anti-crawl",
        action="store_true",
        help="nudge applied throttle while the ego crawls, so cold-start-trapped "
        "models drive the route (recorded control stays the model's raw output)",
    )
    parser.add_argument(
        "--creep-speed",
        type=float,
        default=2.0,
        help="speed (m/s) below which the ego counts as crawling (default 2.0)",
    )
    parser.add_argument(
        "--creep-frames",
        type=int,
        default=5,
        help="consecutive crawl frames before the throttle nudge engages (default 5)",
    )
    parser.add_argument(
        "--creep-throttle",
        type=float,
        default=0.6,
        help="applied throttle during an anti-crawl nudge (default 0.6)",
    )
    parser.add_argument(
        "--creep-duration",
        type=int,
        default=40,
        help="frames each anti-crawl throttle burst is sustained once engaged "
        "(builds momentum instead of fighting per-frame braking; default 40)",
    )
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.delta <= 0:
        parser.error("--delta must be positive")
    if args.num_vehicles < 0:
        parser.error("--num-vehicles must be non-negative")
    if args.num_walkers < 0:
        parser.error("--num-walkers must be non-negative")
    if args.stress != "none":
        try:
            validate_severity(args.stress_severity)
        except ValueError as exc:
            parser.error(str(exc))
    if model_id is not None:
        validate_intervention_args(parser, args, model_id)
    return args


def add_intervention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--intervene-stage",
        choices=INTERVENTION_STAGE_CHOICES,
        default="none",
        help=(
            "counterfactual stage to swap using the same-pose clean/stress "
            "dual forward (default: none)"
        ),
    )
    parser.add_argument(
        "--intervene-direction",
        choices=INTERVENTION_DIRECTION_CHOICES,
        default="restore",
        help=(
            "restore uses a clean-forward stage inside a stress run; inject "
            "uses a stressed-forward stage inside a clean run (ignored when "
            "--intervene-stage none)"
        ),
    )


def validate_intervention_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    model_id: str,
) -> InterventionPolicy:
    try:
        return InterventionPolicy.from_args(args, model_id)
    except ValueError as exc:
        parser.error(str(exc))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_recording(
    args: argparse.Namespace,
    modules: Any,
    runtime_factory: Callable[[argparse.Namespace, Any, ImageStressor | None, Any | None], Any],
    *,
    model_id: str,
    model_label: str,
    record_to_sd2: Callable[[dict[str, Any], str], dict[str, Any]],
    build_run_metadata: Callable[..., dict[str, Any]],
    jsonl_writer_cls: Callable[[Path, dict[str, Any]], Any],
    logger: logging.Logger,
) -> int:
    intervention_policy = InterventionPolicy.from_args(args, model_id)
    stressor, stress_rng = build_image_stressor(args, modules, logger)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    client = None
    world = None
    traffic_manager = None
    ego_vehicle = None
    sensors: list[Any] = []
    npc_vehicles: list[Any] = []
    npc_walkers: list[Any] = []
    npc_walker_controllers: list[Any] = []
    scene_counts = {
        "vehicles": {"requested": int(getattr(args, "num_vehicles", 0) or 0), "spawned": 0},
        "walkers": {"requested": int(getattr(args, "num_walkers", 0) or 0), "spawned": 0},
    }
    frames: list[dict[str, Any]] = []
    collision_frames: set[int] = set()
    lane_invasion_frames: set[int] = set()
    event_lock = Lock()

    try:
        client = modules.carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        world = client.load_world(args.town)
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta
        world.apply_settings(settings)
        world.set_weather(modules.carla.WeatherParameters.ClearNoon)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"town {args.town!r} has no vehicle spawn points")

        spawn_index = select_spawn_index(args.spawn_index, len(spawn_points))
        destination_index = select_destination_index(
            spawn_index,
            len(spawn_points),
            getattr(args, "dest_index", None),
        )
        ego_vehicle = spawn_ego_vehicle(
            world,
            blueprint_library,
            spawn_points,
            spawn_index,
            rng,
        )
        world.tick()

        npc_vehicles, npc_walkers, npc_walker_controllers = spawn_npc_traffic(
            modules.carla,
            world,
            blueprint_library,
            traffic_manager,
            spawn_points,
            ego_spawn_index=spawn_index,
            num_vehicles=scene_counts["vehicles"]["requested"],
            num_walkers=scene_counts["walkers"]["requested"],
            seed=args.seed,
            rng=rng,
            logger=logger,
        )
        scene_counts["vehicles"]["spawned"] = len(npc_vehicles)
        scene_counts["walkers"]["spawned"] = len(npc_walkers)

        runtime = runtime_factory(args, modules, stressor, stress_rng)
        event_sensors = attach_event_sensors(
            modules.carla,
            world,
            blueprint_library,
            ego_vehicle,
            collision_frames,
            lane_invasion_frames,
            event_lock,
        )
        sensor_buffer, model_sensors = attach_model_sensors(
            modules,
            world,
            blueprint_library,
            ego_vehicle,
            runtime.sensor_specs(),
            model_label,
            logger,
            delta=args.delta,
        )
        sensors = [*event_sensors, *model_sensors]

        destination = spawn_points[destination_index].location
        basic_agent = modules.BasicAgent(ego_vehicle, target_speed=DEFAULT_TARGET_SPEED_KMH)
        basic_agent.set_destination(destination)
        route = trace_route(modules, world, ego_vehicle.get_location(), destination)
        if not route:
            route = route_from_basic_agent(basic_agent)
        route_transforms = [(waypoint.transform, road_option) for waypoint, road_option in route]
        if len(route_transforms) < 2:
            raise RuntimeError(f"failed to build a {model_label} global route")

        gps_plan = build_sparse_gps_plan(world, route_transforms)
        route_locations = [transform.location for transform, _road_option in route_transforms]
        route_length_meters = route_length(route_locations)
        progress_tracker = RouteProgressTracker(route_locations)
        runtime.set_global_plan(gps_plan)

        logger.info(
            "Route ready: town=%s spawn=%d dest=%d route_length_m=%.1f dense_points=%d sparse_points=%d",
            args.town,
            spawn_index,
            destination_index,
            route_length_meters,
            len(route_transforms),
            len(gps_plan),
        )

        for warmup_idx in range(args.warmup):
            frame_id = world.tick()
            packet = sensor_buffer.read(frame_id)
            packet["speed"] = (frame_id, {"speed": ego_speed(ego_vehicle)})
            control, _extracted = runtime.run_step(
                packet,
                timestamp=warmup_idx * args.delta,
                frame_id=frame_id,
            )
            ego_vehicle.apply_control(control)

        progress_tracker.reset_initial(ego_vehicle.get_location())

        scenario_id = f"{args.town}_spawn{spawn_index}_dest{destination_index}"
        condition, stress_type, severity = intervention_policy.metadata_condition
        run_id = build_run_id(model_id, scenario_id, condition, stress_type, severity, args.seed)
        if intervention_policy.enabled:
            run_id = f"{run_id}_iv-{intervention_policy.direction}-{intervention_policy.stage}"
        metadata = build_run_metadata(
            run_id=run_id,
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
            town=args.town,
        )

        anti_crawl_nudger = AntiCrawlNudger(args)

        with jsonl_writer_cls(args.output, metadata) as jsonl_writer:
            for frame_idx in range(args.frames):
                frame_id = world.tick()
                packet = sensor_buffer.read(frame_id)
                current_speed = ego_speed(ego_vehicle)
                packet["speed"] = (frame_id, {"speed": current_speed})
                control, extracted = runtime.run_step(
                    packet,
                    timestamp=frame_idx * args.delta,
                    frame_id=frame_id,
                )

                anti_crawl_nudger.apply(control, extracted, current_speed)

                ego_vehicle.apply_control(control)

                current_location = ego_vehicle.get_location()
                route_progress = progress_tracker.progress(current_location)
                extracted["frame_idx"] = frame_idx
                extracted["timestamp"] = round(frame_idx * args.delta, 6)
                extracted["ego"] = ego_measurement(ego_vehicle)
                extracted["outcome"] = {
                    "collision": event_seen(collision_frames, frame_id, event_lock),
                    "lane_invasion": event_seen(lane_invasion_frames, frame_id, event_lock),
                    "route_progress": route_progress,
                    "off_route": progress_tracker.off_route,
                    "min_ttc": None,
                }
                frame = record_to_sd2(extracted, run_id=run_id)
                jsonl_writer.write_frame(frame)
                frames.append(frame)

        write_intervention_sidecar(args.output, intervention_policy)
        write_scene_sidecar(
            args.output,
            town=args.town,
            spawn_index=spawn_index,
            dest_index=destination_index,
            seed=args.seed,
            vehicles=scene_counts["vehicles"],
            walkers=scene_counts["walkers"],
            frames=args.frames,
            delta=args.delta,
            route_length_meters=route_length_meters,
        )
        print_summary(args.output, frames)
        return 0
    finally:
        try:
            cleanup(
                modules.carla if "modules" in locals() else None,
                world,
                traffic_manager,
                sensors,
                ego_vehicle,
                npc_vehicles,
                npc_walkers,
                npc_walker_controllers,
            )
        finally:
            # Drop CARLA wrappers before frame teardown picks an arbitrary order.
            extracted = None
            _extracted = None
            control = None
            packet = None
            runtime = None
            progress_tracker = None
            route_locations = None
            gps_plan = None
            route_transforms = None
            route = None
            basic_agent = None
            destination = None
            sensor_buffer = None
            model_sensors = None
            event_sensors = None
            sensors = []
            npc_walker_controllers = []
            npc_walkers = []
            npc_vehicles = []
            ego_vehicle = None
            spawn_points = None
            blueprint_library = None
            settings = None
            traffic_manager = None
            world = None
            client = None


def build_image_stressor(
    args: argparse.Namespace,
    modules: Any,
    logger: logging.Logger,
) -> tuple[ImageStressor | None, Any | None]:
    if args.stress == "none":
        return None, None
    stressor_type = "brightness_shift" if args.stress == "brightness" else args.stress
    stressor = build_stressor(stressor_type)
    if not isinstance(stressor, ImageStressor):
        raise RuntimeError(f"stress {args.stress!r} is not an image stressor")
    rng = modules.np.random.default_rng(args.seed)
    logger.info("Using visual stressor: %s", stressor.describe(args.stress_severity))
    return stressor, rng


def write_intervention_sidecar(path: str | Path, policy: InterventionPolicy) -> None:
    output_path = Path(f"{path}.intervention.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(policy.config_record(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_scene_sidecar(
    path: str | Path,
    *,
    town: str,
    spawn_index: int,
    dest_index: int,
    seed: int,
    vehicles: Mapping[str, Any],
    walkers: Mapping[str, Any],
    frames: int,
    delta: float,
    route_length_meters: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "town": str(town),
        "spawn_index": int(spawn_index),
        "dest_index": int(dest_index),
        "seed": int(seed),
        "vehicles": {
            "requested": int(vehicles.get("requested", 0)),
            "spawned": int(vehicles.get("spawned", 0)),
        },
        "walkers": {
            "requested": int(walkers.get("requested", 0)),
            "spawned": int(walkers.get("spawned", 0)),
        },
        "frames": int(frames),
        "delta": float(delta),
    }
    if route_length_meters is not None:
        payload["route_length_meters"] = float(route_length_meters)

    output_path = Path(f"{path}.scene.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_visual_stress(
    image: Any,
    stressor: ImageStressor | None,
    severity: int,
    rng: Any | None,
) -> Any:
    if stressor is None:
        return image
    return stressor.apply_image(image, severity, rng)


def control_to_dict(control: Any) -> dict[str, float]:
    return {
        "steer": float(getattr(control, "steer", 0.0)),
        "throttle": float(getattr(control, "throttle", 0.0)),
        "brake": float(getattr(control, "brake", 0.0)),
    }


def vehicle_control_from_dict(carla: Any, command: Mapping[str, Any]) -> Any:
    control = carla.VehicleControl()
    control.steer = float(command.get("steer", 0.0))
    control.throttle = float(command.get("throttle", 0.0))
    control.brake = float(command.get("brake", 0.0))
    return control


def build_intervention_block(
    policy: InterventionPolicy,
    *,
    control_from_stress_forward: Mapping[str, Any],
    control_from_clean_forward: Mapping[str, Any],
    planning_waypoints_clean_forward: Any = None,
    semantic_clean_forward: Mapping[str, Any] | None = None,
    control_hybrid_planning_clean: Mapping[str, Any] | None = None,
    control_hybrid_semantic_clean: Mapping[str, Any] | None = None,
    applied_source: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "stage": policy.stage,
        "direction": policy.direction,
        "applied_source": applied_source or policy.applied_source,
        "control_from_stress_forward": jsonable(dict(control_from_stress_forward)),
        "control_from_clean_forward": jsonable(dict(control_from_clean_forward)),
        "planning_waypoints_clean_forward": jsonable(planning_waypoints_clean_forward),
        "config": policy.config_record(),
    }
    if semantic_clean_forward is not None:
        block["semantic_clean_forward"] = jsonable(dict(semantic_clean_forward))
    if control_hybrid_planning_clean is not None:
        block["control_hybrid_planning_clean"] = jsonable(
            dict(control_hybrid_planning_clean)
        )
    if control_hybrid_semantic_clean is not None:
        block["control_hybrid_semantic_clean"] = jsonable(
            dict(control_hybrid_semantic_clean)
        )
    return block


def snapshot_pid_controllers(owner: Any) -> dict[str, Any]:
    """Copy only PID controllers, not arbitrary controller or model state.

    This helper is intentionally limited to the classic ``turn_controller`` and
    ``speed_controller`` attributes. It does not isolate other mutable fields on
    ``owner``.
    """

    state: dict[str, Any] = {}
    for attr in ("turn_controller", "speed_controller"):
        if hasattr(owner, attr):
            state[attr] = deepcopy(getattr(owner, attr))
    return state


def restore_pid_controllers(owner: Any, state: Mapping[str, Any]) -> None:
    for attr, value in state.items():
        setattr(owner, attr, deepcopy(value))


def attach_event_sensors(
    carla: Any,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
    collision_frames: set[int],
    lane_invasion_frames: set[int],
    event_lock: Lock,
) -> list[Any]:
    sensors: list[Any] = []
    collision_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    collision_sensor.listen(
        lambda event: mark_event(collision_frames, event.frame, event_lock)
    )
    sensors.append(collision_sensor)

    lane_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    lane_sensor.listen(
        lambda event: mark_event(lane_invasion_frames, event.frame, event_lock)
    )
    sensors.append(lane_sensor)
    return sensors


def validate_sensor_ticks(sensor_specs: tuple[dict[str, Any], ...], delta: float) -> None:
    """Every sensor must be able to produce a reading on every simulation tick.

    CARLA fires a sensor once its accumulated elapsed time reaches ``sensor_tick``. A
    ``sensor_tick`` at or above the world delta therefore skips a tick eventually, to
    floating-point accumulation. In synchronous mode the recorder then blocks in
    ``SensorBuffer.read`` waiting for a reading that can never arrive, because no
    further tick is issued while it waits. Fail loudly at startup instead.
    """

    for spec in sensor_specs:
        tick = spec.get("sensor_tick")
        if tick is None:
            continue
        if float(tick) >= float(delta):
            raise ValueError(
                f"sensor {spec.get('id')!r} has sensor_tick={tick} >= world delta={delta}; "
                f"it will eventually skip a tick and hang the recorder. Use sensor_tick=0.0."
            )


@dataclass(frozen=True)
class BlueprintAttributeRequest:
    value: Any
    source_key: str | None
    source_is_spec: bool
    required: bool


def _leaderboard_blueprint_attributes(sensor_type: str) -> dict[str, Any]:
    if sensor_type.startswith("sensor.camera.semantic_segmentation"):
        return {}
    if sensor_type.startswith("sensor.camera.depth"):
        return {}
    if sensor_type.startswith("sensor.camera"):
        return dict(RGB_CAMERA_LEADERBOARD_ATTRIBUTES)
    if sensor_type.startswith("sensor.lidar.ray_cast_semantic"):
        return dict(LIDAR_RAY_CAST_SEMANTIC_LEADERBOARD_ATTRIBUTES)
    if sensor_type.startswith("sensor.lidar"):
        return dict(LIDAR_RAY_CAST_LEADERBOARD_ATTRIBUTES)
    if sensor_type.startswith("sensor.other.gnss"):
        return dict(GNSS_LEADERBOARD_ATTRIBUTES)
    return {}


def _sensor_spec_attribute_name(key: str) -> str:
    return SENSOR_SPEC_ATTRIBUTE_ALIASES.get(key, key)


def _blueprint_attribute_type_name(attribute: Any) -> str:
    attribute_type = getattr(attribute, "type", "")
    type_name = getattr(attribute_type, "name", None)
    if type_name is None:
        type_name = str(attribute_type).rsplit(".", maxsplit=1)[-1]
    return str(type_name).lower()


def _expected_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _blueprint_attribute_value(attribute: Any, requested_value: Any) -> tuple[Any, Any, str]:
    type_name = _blueprint_attribute_type_name(attribute)
    if type_name.endswith("int"):
        return attribute.as_int(), int(requested_value), "as_int()"
    if type_name.endswith("float"):
        return attribute.as_float(), float(requested_value), "as_float()"
    if type_name.endswith("bool"):
        return attribute.as_bool(), _expected_bool(requested_value), "as_bool()"
    return attribute.as_str(), str(requested_value), "as_str()"


def _assert_blueprint_attribute_round_trip(
    blueprint: Any,
    sensor_id: str,
    attr_name: str,
    request: BlueprintAttributeRequest,
) -> None:
    attribute = blueprint.get_attribute(attr_name)
    actual, expected, accessor = _blueprint_attribute_value(attribute, request.value)
    if isinstance(expected, float):
        # CARLA stores blueprint float attributes as binary32, so exact float64
        # round-trips reject correctly-applied values that are not representable.
        matches = math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6)
    else:
        matches = actual == expected
    if matches:
        return

    source = (
        f"spec key {request.source_key!r}"
        if request.source_is_spec
        else "leaderboard default"
    )
    raise ValueError(
        f"sensor {sensor_id!r} blueprint attribute {attr_name!r} from {source} did not "
        f"round-trip via {accessor}: requested {request.value!r}, got {actual!r}"
    )


def configure_sensor_blueprint(blueprint: Any, sensor_spec: Mapping[str, Any]) -> None:
    """Apply leaderboard-equivalent blueprint attributes for a recorder sensor spec."""

    sensor_type = str(sensor_spec["type"])
    sensor_id = str(sensor_spec.get("id", sensor_type))
    requests = {
        attr_name: BlueprintAttributeRequest(
            value=value,
            source_key=None,
            source_is_spec=False,
            required=False,
        )
        for attr_name, value in _leaderboard_blueprint_attributes(sensor_type).items()
    }

    for key, value in sensor_spec.items():
        if key in SENSOR_SPEC_BOOKKEEPING_KEYS and key not in OPTIONAL_SENSOR_SPEC_ATTRIBUTES:
            continue
        attr_name = _sensor_spec_attribute_name(str(key))
        requests[attr_name] = BlueprintAttributeRequest(
            value=value,
            source_key=str(key),
            source_is_spec=True,
            required=str(key) not in OPTIONAL_SENSOR_SPEC_ATTRIBUTES,
        )

    for attr_name, request in requests.items():
        if not blueprint.has_attribute(attr_name):
            if request.required:
                raise ValueError(
                    f"sensor {sensor_id!r} spec key {request.source_key!r} maps to "
                    f"blueprint attribute {attr_name!r}, but {sensor_type!r} has no such "
                    "attribute"
                )
            continue
        blueprint.set_attribute(attr_name, str(request.value))
        _assert_blueprint_attribute_round_trip(blueprint, sensor_id, attr_name, request)


def attach_model_sensors(
    modules: Any,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
    sensor_specs: tuple[dict[str, Any], ...],
    model_label: str,
    logger: logging.Logger,
    delta: float | None = None,
) -> tuple[SensorBuffer, list[Any]]:
    if delta is not None:
        validate_sensor_ticks(sensor_specs, delta)
    active_specs = [spec for spec in sensor_specs if spec["type"] != "sensor.speedometer"]
    buffer = SensorBuffer([str(spec["id"]) for spec in active_specs])
    sensors: list[Any] = []
    for spec in active_specs:
        blueprint = blueprint_library.find(spec["type"])
        configure_sensor_blueprint(blueprint, spec)
        transform = modules.carla.Transform(
            modules.carla.Location(
                x=float(spec.get("x", 0.0)),
                y=float(spec.get("y", 0.0)),
                z=float(spec.get("z", 0.0)),
            ),
            modules.carla.Rotation(
                roll=float(spec.get("roll", 0.0)),
                pitch=float(spec.get("pitch", 0.0)),
                yaw=float(spec.get("yaw", 0.0)),
            ),
        )
        sensor = world.spawn_actor(blueprint, transform, attach_to=ego_vehicle)
        sensor_id = str(spec["id"])
        sensor.listen(buffer.callback(sensor_id, sensor_converter(modules, spec, model_label)))
        sensors.append(sensor)
    logger.info(
        "Attached %s sensor rig: %s",
        model_label,
        ", ".join(str(spec["id"]) for spec in sensor_specs),
    )
    return buffer, sensors


def sensor_converter(
    modules: Any,
    spec: Mapping[str, Any],
    model_label: str,
) -> Callable[[Any], Any]:
    sensor_type = str(spec["type"])
    if sensor_type == "sensor.camera.rgb":
        return lambda image: modules.np.frombuffer(image.raw_data, dtype=modules.np.uint8).reshape(
            (image.height, image.width, 4)
        ).copy()
    if sensor_type == "sensor.lidar.ray_cast":
        return lambda lidar: modules.np.frombuffer(lidar.raw_data, dtype=modules.np.float32).reshape(
            (-1, 4)
        ).copy()
    if sensor_type == "sensor.other.imu":
        return lambda imu: modules.np.array(
            [
                imu.accelerometer.x,
                imu.accelerometer.y,
                imu.accelerometer.z,
                imu.gyroscope.x,
                imu.gyroscope.y,
                imu.gyroscope.z,
                imu.compass,
            ],
            dtype=modules.np.float32,
        )
    if sensor_type == "sensor.other.gnss":
        return lambda gps: modules.np.array(
            [gps.latitude, gps.longitude, gps.altitude],
            dtype=modules.np.float64,
        )
    raise ValueError(f"unsupported {model_label} sensor type {sensor_type!r}")


def trace_route(modules: Any, world: Any, start: Any, destination: Any) -> list[Any]:
    try:
        planner = modules.GlobalRoutePlanner(world.get_map(), 1.0)
        return planner.trace_route(start, destination)
    except TypeError:
        if modules.GlobalRoutePlannerDAO is None:
            raise
        dao = modules.GlobalRoutePlannerDAO(world.get_map(), 1.0)
        planner = modules.GlobalRoutePlanner(dao)
        planner.setup()
        return planner.trace_route(start, destination)


def route_from_basic_agent(agent: Any) -> list[Any]:
    local_planner = getattr(agent, "_local_planner", None)
    queue = getattr(local_planner, "_waypoints_queue", None)
    if queue is None:
        return []
    return list(queue)


def build_sparse_gps_plan(
    world: Any,
    route_transforms: list[tuple[Any, Any]],
    sample_factor: float = 50.0,
) -> list[tuple[dict[str, float], Any]]:
    sampled_indices = downsample_route(route_transforms, sample_factor)
    sparse_route = [route_transforms[index] for index in sampled_indices]
    lat_ref, lon_ref = get_latlon_ref(world)
    return [
        (location_to_gps(lat_ref, lon_ref, transform.location), road_option)
        for transform, road_option in sparse_route
    ]


def downsample_route(route: list[tuple[Any, Any]], sample_factor: float) -> list[int]:
    ids_to_sample: list[int] = []
    prev_option = None
    distance_since_sample = 0.0
    for idx, point in enumerate(route):
        curr_option = point[1]
        curr_value = getattr(curr_option, "value", curr_option)
        prev_value = getattr(prev_option, "value", prev_option)
        lane_change = curr_value in (5, 6)
        prev_lane_change = prev_value in (5, 6)
        if lane_change:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif prev_option is None or (prev_value != curr_value and not prev_lane_change):
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif distance_since_sample > sample_factor:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        elif idx == len(route) - 1:
            ids_to_sample.append(idx)
            distance_since_sample = 0.0
        else:
            current_location = point[0].location
            previous_location = route[idx - 1][0].location
            distance_since_sample += distance(current_location, previous_location)
        prev_option = curr_option
    return sorted(set(ids_to_sample))


def location_to_gps(lat_ref: float, lon_ref: float, location: Any) -> dict[str, float]:
    earth_radius_equator = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * earth_radius_equator / 180.0
    my = scale * earth_radius_equator * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    # CARLA 0.9.16's GNSS sensor reports latitude increasing with +y. The CARLA 0.9.10
    # leaderboard code this was ported from negated y here; keeping that negation
    # mirrors the whole global plan about y=0, so next_wp -- and therefore
    # target_point -- points at a reflected goal and the ego drives off route.
    my += location.y
    lon = mx * 180.0 / (math.pi * earth_radius_equator * scale)
    lat = 360.0 * math.atan(math.exp(my / (earth_radius_equator * scale))) / math.pi - 90.0
    return {"lat": lat, "lon": lon, "z": location.z}


def get_latlon_ref(world: Any) -> tuple[float, float]:
    tree = ET.ElementTree(ET.fromstring(world.get_map().to_opendrive()))
    lat_ref = 42.0
    lon_ref = 2.0
    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if not georef.text:
                    continue
                for item in georef.text.split(" "):
                    if "+lat_0" in item:
                        lat_ref = float(item.split("=")[1])
                    if "+lon_0" in item:
                        lon_ref = float(item.split("=")[1])
    return lat_ref, lon_ref


def select_spawn_index(requested: int, count: int) -> int:
    return int(requested) % count


def select_destination_index(
    spawn_index: int,
    count: int,
    requested: int | None = None,
) -> int:
    if requested is not None:
        return int(requested) % count
    return (spawn_index + max(1, count // 2)) % count


def spawn_ego_vehicle(
    world: Any,
    blueprint_library: Any,
    spawn_points: list[Any],
    spawn_index: int,
    rng: random.Random,
) -> Any:
    blueprints = list(blueprint_library.filter("vehicle.tesla.model3"))
    if not blueprints:
        blueprints = list(blueprint_library.filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("no vehicle blueprints available")

    blueprint = sorted(blueprints, key=lambda item: item.id)[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", colors[rng.randrange(len(colors))])

    for offset in range(len(spawn_points)):
        transform = spawn_points[(spawn_index + offset) % len(spawn_points)]
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(False)
            return vehicle
    raise RuntimeError("failed to spawn ego vehicle at any map spawn point")


def spawn_npc_traffic(
    carla: Any,
    world: Any,
    blueprint_library: Any,
    traffic_manager: Any,
    spawn_points: list[Any],
    *,
    ego_spawn_index: int,
    num_vehicles: int,
    num_walkers: int,
    seed: int,
    rng: random.Random,
    logger: logging.Logger,
) -> tuple[list[Any], list[Any], list[Any]]:
    vehicles: list[Any] = []
    walkers: list[Any] = []
    walker_controllers: list[Any] = []

    if num_vehicles > 0:
        vehicles = spawn_npc_vehicles(
            world,
            blueprint_library,
            traffic_manager,
            spawn_points,
            ego_spawn_index=ego_spawn_index,
            requested=num_vehicles,
            rng=rng,
        )

    if num_walkers > 0:
        walkers, walker_controllers = spawn_npc_walkers(
            carla,
            world,
            blueprint_library,
            requested=num_walkers,
            seed=seed,
            rng=rng,
        )

    logger.info(
        "NPC traffic ready: vehicles requested=%d spawned=%d walkers requested=%d spawned=%d",
        num_vehicles,
        len(vehicles),
        num_walkers,
        len(walkers),
    )
    return vehicles, walkers, walker_controllers


def spawn_npc_vehicles(
    world: Any,
    blueprint_library: Any,
    traffic_manager: Any,
    spawn_points: list[Any],
    *,
    ego_spawn_index: int,
    requested: int,
    rng: random.Random,
) -> list[Any]:
    blueprints = sorted(
        list(blueprint_library.filter("vehicle.*")),
        key=lambda item: item.id,
    )
    if not blueprints:
        return []

    candidates = [
        point for index, point in enumerate(spawn_points) if index != int(ego_spawn_index)
    ]
    rng.shuffle(candidates)
    tm_port = int(traffic_manager.get_port())
    vehicles: list[Any] = []
    for transform in candidates:
        if len(vehicles) >= requested:
            break
        blueprint = rng.choice(blueprints)
        prepare_npc_vehicle_blueprint(blueprint, rng)
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            continue
        try:
            vehicle.set_autopilot(True, tm_port)
        except RuntimeError:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError:
                pass
            continue
        vehicles.append(vehicle)
    return vehicles


def prepare_npc_vehicle_blueprint(blueprint: Any, rng: random.Random) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "autopilot")
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", colors[rng.randrange(len(colors))])
    if blueprint.has_attribute("driver_id"):
        drivers = blueprint.get_attribute("driver_id").recommended_values
        if drivers:
            blueprint.set_attribute("driver_id", drivers[rng.randrange(len(drivers))])


def spawn_npc_walkers(
    carla: Any,
    world: Any,
    blueprint_library: Any,
    *,
    requested: int,
    seed: int,
    rng: random.Random,
) -> tuple[list[Any], list[Any]]:
    if hasattr(world, "set_pedestrians_seed"):
        world.set_pedestrians_seed(int(seed))

    walker_blueprints = sorted(
        list(blueprint_library.filter("walker.pedestrian.*")),
        key=lambda item: item.id,
    )
    if not walker_blueprints:
        return [], []

    controller_blueprint = blueprint_library.find("controller.ai.walker")
    walkers: list[Any] = []
    controllers: list[Any] = []
    speeds_by_walker: list[float] = []

    attempts = max(requested * 4, requested)
    for _ in range(attempts):
        if len(walkers) >= requested:
            break
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        walker_blueprint = rng.choice(walker_blueprints)
        walker_speed = walker_speed_from_blueprint(walker_blueprint)
        walker = world.try_spawn_actor(walker_blueprint, carla.Transform(location))
        if walker is None:
            continue
        controller = world.try_spawn_actor(controller_blueprint, carla.Transform(), walker)
        if controller is None:
            try:
                if walker.is_alive:
                    walker.destroy()
            except RuntimeError:
                pass
            continue
        walkers.append(walker)
        controllers.append(controller)
        speeds_by_walker.append(walker_speed)

    if controllers:
        world.tick()
    for controller, speed in zip(controllers, speeds_by_walker):
        try:
            controller.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(speed)
        except RuntimeError:
            pass

    return walkers, controllers


def walker_speed_from_blueprint(blueprint: Any) -> float:
    if not blueprint.has_attribute("speed"):
        return 1.4
    values = blueprint.get_attribute("speed").recommended_values
    if len(values) > 1:
        return float(values[1])
    if values:
        return float(values[0])
    return 1.4


def route_position(np: Any, route_planner: Any, gps: Any) -> Any:
    return (gps - route_planner.mean) * route_planner.scale


def route_target_point(np: Any, pos: Any, compass: float, next_wp: Any) -> Any:
    theta = compass + np.pi / 2
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
    return rotation.T.dot(local_command_point)


def mark_event(event_frames: set[int], frame_id: int, event_lock: Lock) -> None:
    with event_lock:
        event_frames.add(frame_id)


def event_seen(event_frames: set[int], frame_id: int, event_lock: Lock) -> bool:
    with event_lock:
        return frame_id in event_frames


def ego_measurement(vehicle: Any) -> dict[str, float]:
    transform = vehicle.get_transform()
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw": float(transform.rotation.yaw),
        "speed": ego_speed(vehicle),
    }


def ego_speed(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def build_run_id(
    model_id: str,
    scenario_id: str,
    condition: str,
    stress_type: str | None,
    severity: int,
    seed: int,
) -> str:
    condition_part = "clean" if condition == "clean" else f"{stress_type}_s{severity}"
    return f"{model_id}_{scenario_id}_{condition_part}_seed{seed}"


def print_summary(path: Path, frames: list[dict[str, Any]]) -> None:
    collisions = sum(
        1 for frame in frames if frame["states"]["outcome"].get("collision") is True
    )
    lane_invasions = sum(
        1 for frame in frames if frame["states"]["outcome"].get("lane_invasion") is True
    )
    final_progress = 0.0
    if frames:
        final_progress = float(
            frames[-1]["states"]["outcome"].get("route_progress", 0.0)
        )
    print(
        "Wrote "
        f"{len(frames)} frames to {path} | collisions={collisions} "
        f"lane_invasions={lane_invasions} final_route_progress={final_progress:.3f}"
    )


def cleanup(
    carla: Any | None,
    world: Any | None,
    traffic_manager: Any | None,
    sensors: list[Any],
    ego_vehicle: Any | None,
    npc_vehicles: list[Any] | None = None,
    npc_walkers: list[Any] | None = None,
    npc_walker_controllers: list[Any] | None = None,
) -> None:
    del carla
    try:
        try:
            for sensor in sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.stop()
                        sensor.destroy()
                except RuntimeError:
                    pass

            for controller in npc_walker_controllers or []:
                try:
                    if controller is not None and controller.is_alive:
                        controller.stop()
                        controller.destroy()
                except RuntimeError:
                    pass

            for walker in npc_walkers or []:
                try:
                    if walker is not None and walker.is_alive:
                        walker.destroy()
                except RuntimeError:
                    pass

            _unregister_npc_vehicles_from_traffic_manager(
                world,
                traffic_manager,
                npc_vehicles,
            )

            for vehicle in npc_vehicles or []:
                try:
                    if vehicle is not None and vehicle.is_alive:
                        vehicle.destroy()
                except RuntimeError:
                    pass

            try:
                if ego_vehicle is not None and ego_vehicle.is_alive:
                    ego_vehicle.destroy()
            except RuntimeError:
                pass
        finally:
            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except RuntimeError:
                    pass
    finally:
        _restore_world_async(world)


def _unregister_npc_vehicles_from_traffic_manager(
    world: Any | None,
    traffic_manager: Any | None,
    npc_vehicles: list[Any] | None,
) -> None:
    if traffic_manager is None:
        return

    tm_port: int | None = None
    attempted_unregister = False
    for vehicle in npc_vehicles or []:
        try:
            if vehicle is not None and vehicle.is_alive:
                if tm_port is None:
                    tm_port = int(traffic_manager.get_port())
                attempted_unregister = True
                vehicle.set_autopilot(False, tm_port)
        except RuntimeError:
            attempted_unregister = True

    if attempted_unregister:
        _tick_world_if_synchronous(world)


def _tick_world_if_synchronous(world: Any | None) -> None:
    if world is None:
        return

    try:
        settings = world.get_settings()
        if bool(getattr(settings, "synchronous_mode", False)):
            world.tick()
    except RuntimeError:
        pass


def _restore_world_async(world: Any | None) -> None:
    if world is None:
        return

    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except RuntimeError:
        pass


def distance(left: Any, right: Any) -> float:
    if hasattr(left, "distance"):
        return float(left.distance(right))
    return math.sqrt(
        (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2
    )


def route_length(locations: list[Any]) -> float:
    return RouteProgressTracker._polyline_distance(locations)


def pool_feature(np: Any, feature: Any) -> list[float]:
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim == 0:
        return [float(array)]
    if array.ndim == 1:
        pooled = array
    elif array.ndim == 2 and array.shape[0] == 1:
        pooled = array[0]
    else:
        pooled = array.mean(axis=tuple(range(array.ndim - 1)))
    return [float(value) for value in pooled.reshape(-1)]


def target_speed_from_waypoints(np: Any, waypoints: Any) -> float | None:
    array = np.asarray(waypoints, dtype=np.float32)
    if array.ndim < 2 or array.shape[0] < 2:
        return None
    return float(np.linalg.norm(array[0] - array[1]) * 2.0)


def shape_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: shape_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_summary(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return type(value).__name__


def sensor_shape_summary(packet: Mapping[str, tuple[int, Any]]) -> dict[str, Any]:
    return {
        sensor_id: {"frame": frame_id, "shape": shape_summary(payload)}
        for sensor_id, (frame_id, payload) in packet.items()
    }


def jsonable(value: Any) -> Any:
    value = tolist(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scalar_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "detach"):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
