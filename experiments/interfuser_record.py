"""Record an InterFuser closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.interfuser_adapter``. This
script is the only place that imports CARLA, torch, timm, and InterFuser code.
It mirrors the InterFuser leaderboard agent's sensor spec, preprocessing,
model forward pass, tracker, and controller, but avoids the display/save side
effects in ``team_code.interfuser_agent``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import sys
import time
import types
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Mapping

from sd2.adapters.interfuser_adapter import (
    Sd2JsonlWriter,
    build_interfuser_run_metadata,
    interfuser_record_to_sd2,
)
from sd2.stressors import ImageStressor, build_stressor, validate_severity

try:
    import _carla_e2e_common as e2e
except ImportError:
    from experiments import _carla_e2e_common as e2e

RouteProgressTracker = e2e.RouteProgressTracker


REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_PYTHON_API = (
    REPO_ROOT / "external" / "Carla" / "CARLA_0.9.16" / "PythonAPI" / "carla"
)
_CKPT_ENV = "INTERFUSER_CKPT"
_CKPT_FROM_ENV = os.environ.get(_CKPT_ENV)
DEFAULT_CHECKPOINT = Path(_CKPT_FROM_ENV) if _CKPT_FROM_ENV else None
DEFAULT_TARGET_SPEED_KMH = 25.0
LOGGER = logging.getLogger("interfuser_record")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

INTERFUSER_SENSOR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": 0.0,
        "z": 2.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": 800,
        "height": 600,
        "fov": 100,
        "id": "rgb",
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": 0.0,
        "z": 2.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -60.0,
        "width": 400,
        "height": 300,
        "fov": 100,
        "id": "rgb_left",
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": 0.0,
        "z": 2.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 60.0,
        "width": 400,
        "height": 300,
        "fov": 100,
        "id": "rgb_right",
    },
    {
        "type": "sensor.lidar.ray_cast",
        "x": 1.3,
        "y": 0.0,
        "z": 2.5,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -90.0,
        "id": "lidar",
    },
    {
        "type": "sensor.other.imu",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        # Must fire on every tick. A sensor_tick equal to the world delta (0.05)
        # eventually skips a tick to floating-point accumulation, and the recorder
        # then blocks forever waiting for that frame's reading.
        "sensor_tick": 0.0,
        "id": "imu",
    },
    {
        "type": "sensor.other.gnss",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        # Same trap as the imu: a sensor_tick at or above the world delta
        # eventually skips a tick and the recorder blocks forever.
        "sensor_tick": 0.0,
        "id": "gps",
    },
    {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
)


@dataclass(frozen=True)
class RuntimeModules:
    carla: Any
    torch: Any
    np: Any
    cv2: Any
    Image: Any
    transforms: Any
    create_model: Callable[..., Any]
    BasicAgent: type
    GlobalRoutePlanner: type
    GlobalRoutePlannerDAO: type | None
    GlobalConfig: type
    InterfuserController: type
    RoutePlanner: type
    Tracker: type
    lidar_to_histogram_features: Callable[..., Any]
    transform_2d_points: Callable[..., Any]
    find_peak_box: Callable[..., Any]
    reweight_array: Any


class Resize2FixedSize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, pil_img: Any) -> Any:
        return pil_img.resize(self.size)


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


class InterFuserRuntime:
    def __init__(
        self,
        args: argparse.Namespace,
        modules: RuntimeModules,
        stressor: ImageStressor | None,
        stress_rng: Any | None,
    ) -> None:
        self.args = args
        self.modules = modules
        self.torch = modules.torch
        self.np = modules.np
        self.cv2 = modules.cv2
        self.Image = modules.Image
        self.stressor = stressor
        self.stress_rng = stress_rng
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            LOGGER.warning("CUDA is not available; running InterFuser on %s", self.device)

        self.config = modules.GlobalConfig(
            model="interfuser_baseline",
            model_path=str(args.checkpoint),
        )
        self.net = modules.create_model(self.config.model)
        checkpoint = self.torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_result = self.net.load_state_dict(state_dict)
        LOGGER.info(
            "Loaded InterFuser checkpoint %s | missing=%d unexpected=%d",
            args.checkpoint,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )
        self.net.to(self.device).eval()

        self.rgb_front_transform = _create_carla_rgb_transform(modules, 224)
        self.rgb_left_transform = _create_carla_rgb_transform(modules, 128)
        self.rgb_right_transform = _create_carla_rgb_transform(modules, 128)
        self.rgb_center_transform = _create_carla_rgb_transform(
            modules,
            128,
            need_scale=False,
        )
        self.controller = modules.InterfuserController(self.config)
        self.intervention = e2e.InterventionPolicy.from_args(args, "interfuser")
        self.trackers = {
            "clean_forward": modules.Tracker(),
            "stress_forward": modules.Tracker(),
        }
        self.route_planner = modules.RoutePlanner(4.0, 50.0)
        self.softmax = self.torch.nn.Softmax(dim=1)
        self.step = -1
        self.prev_lidar = None
        self.prev_control = None
        self.prev_extracted: dict[str, Any] | None = None
        self.traffic_meta_moving_avg = {
            "clean_forward": self.np.zeros((400, 7), dtype=self.np.float32),
            "stress_forward": self.np.zeros((400, 7), dtype=self.np.float32),
        }
        self.logged_shapes = False

    def set_global_plan(self, gps_plan: list[tuple[dict[str, float], Any]]) -> None:
        if len(gps_plan) < 2:
            raise RuntimeError("InterFuser requires at least two route points")
        self.route_planner.set_route(gps_plan, True)

    def run_step(
        self,
        sensor_packet: dict[str, tuple[int, Any]],
        timestamp: float,
        frame_id: int,
    ) -> tuple[Any, dict[str, Any]]:
        self.step += 1
        tick_clean = self._tick(sensor_packet)
        tick_stress = self._with_stressed_images(tick_clean)
        stress_input = self._build_model_input(tick_stress)
        input_shapes = _shape_summary(stress_input)
        if not self.logged_shapes:
            LOGGER.info("First InterFuser sensor packet shapes: %s", _sensor_shape_summary(sensor_packet))
            LOGGER.info("First InterFuser model input shapes: %s", input_shapes)

        stress_forward = self._forward_once(tick_stress, "stress_forward", stress_input, input_shapes)
        clean_forward = self._forward_once(
            tick_clean,
            "clean_forward",
            self._build_model_input(tick_clean),
            input_shapes,
        )
        if not self.logged_shapes:
            LOGGER.info("First InterFuser model output shapes: %s", _shape_summary(stress_forward["outputs"]))
            self.logged_shapes = True

        planning_forward = (
            clean_forward
            if self.intervention.source_for_stage("planning") == "clean_forward"
            else stress_forward
        )
        semantic_forward = (
            clean_forward
            if self.intervention.source_for_stage("semantic") == "clean_forward"
            else stress_forward
        )
        control_hybrid_planning_clean = None
        control_hybrid_semantic_clean = None
        if self.intervention.stage == "none":
            control_hybrid_planning_clean, _planning_meta_infos = self._control_from_outputs(
                float(tick_stress["speed"]),
                clean_forward["pred_waypoints"],
                stress_forward,
                preserve_state=True,
            )
            control_hybrid_semantic_clean, _semantic_meta_infos = self._control_from_outputs(
                float(tick_stress["speed"]),
                stress_forward["pred_waypoints"],
                clean_forward,
                preserve_state=True,
            )
        applied_control, meta_infos = self._control_from_outputs(
            float(tick_stress["speed"]),
            planning_forward["pred_waypoints"],
            semantic_forward,
            preserve_state=False,
        )

        control = e2e.vehicle_control_from_dict(self.modules.carla, applied_control)

        extracted = {
            "vision": {
                "image_mean": tick_stress["image_mean"],
                "image_std": tick_stress["image_std"],
                "feature": _pool_feature(self.np, stress_forward["bev_feature"]),
                "feature_source": "mean_pooled_bev_feature",
            },
            "semantic": {
                **_summarize_traffic_meta(self.modules, self.np, stress_forward["traffic_meta"]),
                "junction": stress_forward["is_junction"],
                "traffic_light_state": stress_forward["traffic_light_state"],
                "stop_sign": stress_forward["stop_sign"],
            },
            "planning": {
                "waypoints": stress_forward["pred_waypoints"].astype(float).tolist(),
                "target_speed": _parse_target_speed(stress_forward["meta_infos"][0]),
                "target_point": tick_stress["target_point"].astype(float).tolist(),
                "command": int(tick_stress["next_command"]),
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "interfuser": {
                "carla_frame": int(frame_id),
                "controller_meta": [str(item) for item in meta_infos[:3]],
                "safe_distance": float(meta_infos[3]),
            },
            "intervention": e2e.build_intervention_block(
                self.intervention,
                control_from_stress_forward=stress_forward["control"],
                control_from_clean_forward=clean_forward["control"],
                planning_waypoints_clean_forward=clean_forward["pred_waypoints"].astype(float).tolist(),
                semantic_clean_forward={
                    **_summarize_traffic_meta(self.modules, self.np, clean_forward["traffic_meta"]),
                    "junction": clean_forward["is_junction"],
                    "traffic_light_state": clean_forward["traffic_light_state"],
                    "stop_sign": clean_forward["stop_sign"],
                },
                control_hybrid_planning_clean=control_hybrid_planning_clean,
                control_hybrid_semantic_clean=control_hybrid_semantic_clean,
            ),
        }

        self.prev_control = control
        self.prev_extracted = dict(extracted)
        return control, extracted

    def _forward_once(
        self,
        tick_data: dict[str, Any],
        source: str,
        model_input: dict[str, Any],
        input_shapes: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            with self.torch.no_grad():
                outputs = self.net(model_input)
        except Exception:
            LOGGER.exception("InterFuser forward failed; model input shapes were: %s", input_shapes)
            raise

        if len(outputs) != 6:
            raise RuntimeError(f"InterFuser forward returned {len(outputs)} outputs, expected 6")

        (
            traffic_meta_tensor,
            pred_waypoints_tensor,
            is_junction_tensor,
            traffic_light_tensor,
            stop_sign_tensor,
            bev_feature_tensor,
        ) = outputs

        traffic_meta = traffic_meta_tensor.detach().cpu().numpy()[0]
        bev_feature = bev_feature_tensor.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints_tensor.detach().cpu().numpy()[0].reshape(-1, 2)
        is_junction = float(
            self.softmax(is_junction_tensor).detach().cpu().numpy().reshape(-1)[0]
        )
        traffic_light_state = float(
            self.softmax(traffic_light_tensor).detach().cpu().numpy().reshape(-1)[0]
        )
        stop_sign = float(
            self.softmax(stop_sign_tensor).detach().cpu().numpy().reshape(-1)[0]
        )

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.trackers[source].update_and_predict(
                traffic_meta.reshape(20, 20, -1),
                tick_data["gps"],
                tick_data["compass"],
                self.step // 2,
            ).reshape(400, -1)
            self.traffic_meta_moving_avg[source] = (
                self.config.momentum * self.traffic_meta_moving_avg[source]
                + (1 - self.config.momentum) * traffic_meta
            )
        traffic_meta = self.traffic_meta_moving_avg[source]

        result = {
            "outputs": outputs,
            "traffic_meta": traffic_meta,
            "bev_feature": bev_feature,
            "pred_waypoints": pred_waypoints,
            "is_junction": is_junction,
            "traffic_light_state": traffic_light_state,
            "stop_sign": stop_sign,
        }
        control, meta_infos = self._control_from_outputs(
            float(tick_data["speed"]),
            pred_waypoints,
            result,
            preserve_state=True,
        )
        result["control"] = control
        result["meta_infos"] = meta_infos
        return result

    def _control_from_outputs(
        self,
        velocity: float,
        pred_waypoints: Any,
        semantic: Mapping[str, Any],
        *,
        preserve_state: bool,
    ) -> tuple[dict[str, float], Any]:
        controller = deepcopy(self.controller) if preserve_state else self.controller
        steer, throttle, brake, meta_infos = controller.run_step(
            velocity,
            pred_waypoints,
            float(semantic["is_junction"]),
            float(semantic["traffic_light_state"]),
            float(semantic["stop_sign"]),
            semantic["traffic_meta"],
        )
        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0
        return {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
        }, meta_infos

    def _tick(self, input_data: dict[str, tuple[int, Any]]) -> dict[str, Any]:
        rgb = self.cv2.cvtColor(input_data["rgb"][1][:, :, :3], self.cv2.COLOR_BGR2RGB)
        rgb_left = self.cv2.cvtColor(
            input_data["rgb_left"][1][:, :, :3],
            self.cv2.COLOR_BGR2RGB,
        )
        rgb_right = self.cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3],
            self.cv2.COLOR_BGR2RGB,
        )

        gps = input_data["gps"][1][:2]
        speed = float(input_data["speed"][1]["speed"])
        compass = float(input_data["imu"][1][-1])
        if math.isnan(compass):
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
            "image_mean": float(rgb.astype(self.np.float32).mean() / 255.0),
            "image_std": float(rgb.astype(self.np.float32).std() / 255.0),
        }

        pos = self._get_position(result)
        lidar_data = input_data["lidar"][1]
        lidar_unprocessed = lidar_data[:, :3].copy()
        lidar_unprocessed[:, 1] *= -1
        full_lidar = self.modules.transform_2d_points(
            lidar_unprocessed,
            self.np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            self.np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = self.modules.lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4 or self.prev_lidar is None:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        next_wp, next_cmd = self.route_planner.run_step(pos)
        result["next_command"] = int(next_cmd.value)
        result["measurements"] = [pos[0], pos[1], compass, speed]

        theta = compass + self.np.pi / 2
        rotation = self.np.array(
            [
                [self.np.cos(theta), -self.np.sin(theta)],
                [self.np.sin(theta), self.np.cos(theta)],
            ]
        )
        local_command_point = self.np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = rotation.T.dot(local_command_point)
        result["target_point"] = local_command_point
        return result

    def _with_stressed_images(self, tick_data: dict[str, Any]) -> dict[str, Any]:
        stressed = dict(tick_data)
        stressed["rgb"] = self._apply_visual_stress(tick_data["rgb"])
        stressed["rgb_left"] = self._apply_visual_stress(tick_data["rgb_left"])
        stressed["rgb_right"] = self._apply_visual_stress(tick_data["rgb_right"])
        stressed["image_mean"] = float(stressed["rgb"].astype(self.np.float32).mean() / 255.0)
        stressed["image_std"] = float(stressed["rgb"].astype(self.np.float32).std() / 255.0)
        return stressed

    def _get_position(self, tick_data: dict[str, Any]) -> Any:
        gps = tick_data["gps"]
        return (gps - self.route_planner.mean) * self.route_planner.scale

    def _build_model_input(self, tick_data: dict[str, Any]) -> dict[str, Any]:
        rgb = (
            self.rgb_front_transform(self.Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(self.Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(self.Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(self.Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .to(self.device)
            .float()
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = int(tick_data["next_command"]) - 1
        if cmd < 0 or cmd >= len(cmd_one_hot):
            LOGGER.warning("Invalid route command %s; falling back to lane-follow one-hot", cmd)
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(float(tick_data["speed"]))
        measurements = self.torch.from_numpy(
            self.np.array(cmd_one_hot, dtype=self.np.float32)
        ).float().unsqueeze(0).to(self.device)

        return {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_center": rgb_center,
            "measurements": measurements,
            "target_point": self.torch.from_numpy(tick_data["target_point"])
            .float()
            .to(self.device)
            .view(1, -1),
            "lidar": self.torch.from_numpy(tick_data["lidar"])
            .float()
            .to(self.device)
            .unsqueeze(0),
        }

    def _apply_visual_stress(self, image: Any) -> Any:
        if self.stressor is None:
            return image
        return self.stressor.apply_image(image, self.args.stress_severity, self.stress_rng)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging()
    modules = _import_runtime_modules()
    stressor, stress_rng = _build_image_stressor(args, modules)
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
        "vehicles": {"requested": int(args.num_vehicles), "spawned": 0},
        "walkers": {"requested": int(args.num_walkers), "spawned": 0},
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

        spawn_index = _select_spawn_index(args.spawn_index, len(spawn_points))
        destination_index = e2e.select_destination_index(
            spawn_index,
            len(spawn_points),
            args.dest_index,
        )
        ego_vehicle = _spawn_ego_vehicle(
            world,
            blueprint_library,
            spawn_points,
            spawn_index,
            rng,
        )
        world.tick()

        npc_vehicles, npc_walkers, npc_walker_controllers = e2e.spawn_npc_traffic(
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
            logger=LOGGER,
        )
        scene_counts["vehicles"]["spawned"] = len(npc_vehicles)
        scene_counts["walkers"]["spawned"] = len(npc_walkers)

        event_sensors = _attach_event_sensors(
            modules.carla,
            world,
            blueprint_library,
            ego_vehicle,
            collision_frames,
            lane_invasion_frames,
            event_lock,
        )
        sensor_buffer, interfuser_sensors = _attach_interfuser_sensors(
            modules,
            world,
            blueprint_library,
            ego_vehicle,
        )
        sensors = [*event_sensors, *interfuser_sensors]

        destination = spawn_points[destination_index].location
        basic_agent = modules.BasicAgent(ego_vehicle, target_speed=DEFAULT_TARGET_SPEED_KMH)
        basic_agent.set_destination(destination)
        route = _trace_route(modules, world, ego_vehicle.get_location(), destination)
        if not route:
            route = _route_from_basic_agent(basic_agent)
        route_transforms = [(waypoint.transform, road_option) for waypoint, road_option in route]
        if len(route_transforms) < 2:
            raise RuntimeError("failed to build an InterFuser global route")

        gps_plan = _build_sparse_gps_plan(world, route_transforms)
        route_locations = [transform.location for transform, _road_option in route_transforms]
        route_length_meters = e2e.route_length(route_locations)
        progress_tracker = RouteProgressTracker(route_locations)

        runtime = InterFuserRuntime(args, modules, stressor, stress_rng)
        runtime.set_global_plan(gps_plan)

        LOGGER.info(
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
            packet["speed"] = (frame_id, {"speed": _ego_speed(ego_vehicle)})
            control, _extracted = runtime.run_step(
                packet,
                timestamp=warmup_idx * args.delta,
                frame_id=frame_id,
            )
            ego_vehicle.apply_control(control)

        progress_tracker.reset_initial(ego_vehicle.get_location())

        scenario_id = f"{args.town}_spawn{spawn_index}_dest{destination_index}"
        intervention_policy = e2e.InterventionPolicy.from_args(args, "interfuser")
        condition, stress_type, severity = intervention_policy.metadata_condition
        run_id = _build_run_id(scenario_id, condition, stress_type, severity, args.seed)
        if intervention_policy.enabled:
            run_id = f"{run_id}_iv-{intervention_policy.direction}-{intervention_policy.stage}"
        metadata = build_interfuser_run_metadata(
            run_id=run_id,
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
            town=args.town,
        )

        anti_crawl_nudger = e2e.AntiCrawlNudger(args)

        with Sd2JsonlWriter(args.output, metadata) as jsonl_writer:
            for frame_idx in range(args.frames):
                frame_id = world.tick()
                packet = sensor_buffer.read(frame_id)
                current_speed = _ego_speed(ego_vehicle)
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
                extracted["ego"] = _ego_measurement(ego_vehicle)
                extracted["outcome"] = {
                    "collision": _event_seen(collision_frames, frame_id, event_lock),
                    "lane_invasion": _event_seen(lane_invasion_frames, frame_id, event_lock),
                    "route_progress": route_progress,
                    "off_route": progress_tracker.off_route,
                    "min_ttc": None,
                }
                frame = interfuser_record_to_sd2(extracted, run_id=run_id)
                jsonl_writer.write_frame(frame)
                frames.append(frame)

        e2e.write_intervention_sidecar(args.output, intervention_policy)
        e2e.write_scene_sidecar(
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
        _print_summary(args.output, frames)
        return 0
    finally:
        try:
            e2e.cleanup(
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
            interfuser_sensors = None
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    return e2e.parse_record_args(
        argv,
        description="Record an InterFuser synchronous CARLA run as SD2 JSONL.",
        default_checkpoint=DEFAULT_CHECKPOINT,
        model_id="interfuser",
        checkpoint_required=DEFAULT_CHECKPOINT is None,
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _apply_interfuser_preamble() -> None:
    imgaug = types.ModuleType("imgaug")
    augmenters = types.ModuleType("imgaug.augmenters")
    imgaug.augmenters = augmenters
    sys.modules["imgaug"] = imgaug
    sys.modules["imgaug.augmenters"] = augmenters

    paths = [
        REPO_ROOT / "models" / "InterFuser" / "interfuser",
        REPO_ROOT / "models" / "InterFuser" / "leaderboard",
        REPO_ROOT / "models" / "InterFuser" / "scenario_runner",
    ]
    for path in paths:
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    carla_api_path = str(CARLA_PYTHON_API)
    if carla_api_path not in sys.path:
        sys.path.insert(0, carla_api_path)


def _import_runtime_modules() -> RuntimeModules:
    _apply_interfuser_preamble()

    import carla
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    if "int" not in np.__dict__:
        np.int = int

    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    except ImportError:
        GlobalRoutePlannerDAO = None

    from team_code.interfuser_config import GlobalConfig
    from team_code.interfuser_controller import InterfuserController
    from team_code.planner import RoutePlanner
    from team_code.render import find_peak_box, reweight_array
    from team_code.tracker import Tracker
    from team_code.utils import lidar_to_histogram_features, transform_2d_points
    from timm.models import create_model

    return RuntimeModules(
        carla=carla,
        torch=torch,
        np=np,
        cv2=cv2,
        Image=Image,
        transforms=transforms,
        create_model=create_model,
        BasicAgent=BasicAgent,
        GlobalRoutePlanner=GlobalRoutePlanner,
        GlobalRoutePlannerDAO=GlobalRoutePlannerDAO,
        GlobalConfig=GlobalConfig,
        InterfuserController=InterfuserController,
        RoutePlanner=RoutePlanner,
        Tracker=Tracker,
        lidar_to_histogram_features=lidar_to_histogram_features,
        transform_2d_points=transform_2d_points,
        find_peak_box=find_peak_box,
        reweight_array=reweight_array,
    )


def _create_carla_rgb_transform(
    modules: RuntimeModules,
    input_size: int | tuple[int, int],
    need_scale: bool = True,
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> Any:
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
        input_size_num = input_size[-1]
    else:
        img_size = input_size
        input_size_num = input_size

    tfl: list[Any] = []
    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("cannot find InterFuser crop size")
    tfl.append(modules.transforms.CenterCrop(img_size))
    tfl.append(modules.transforms.ToTensor())
    tfl.append(
        modules.transforms.Normalize(
            mean=modules.torch.tensor(mean),
            std=modules.torch.tensor(std),
        )
    )
    return modules.transforms.Compose(tfl)


def _build_image_stressor(
    args: argparse.Namespace,
    modules: RuntimeModules,
) -> tuple[ImageStressor | None, Any | None]:
    if args.stress == "none":
        return None, None
    stressor_type = "brightness_shift" if args.stress == "brightness" else args.stress
    stressor = build_stressor(stressor_type)
    if not isinstance(stressor, ImageStressor):
        raise RuntimeError(f"stress {args.stress!r} is not an image stressor")
    rng = modules.np.random.default_rng(args.seed)
    LOGGER.info("Using visual stressor: %s", stressor.describe(args.stress_severity))
    return stressor, rng


def _select_spawn_index(requested: int, count: int) -> int:
    return int(requested) % count


def _select_destination_index(spawn_index: int, count: int) -> int:
    return (spawn_index + max(1, count // 2)) % count


def _spawn_ego_vehicle(
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


def _attach_event_sensors(
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
        lambda event: _mark_event(collision_frames, event.frame, event_lock)
    )
    sensors.append(collision_sensor)

    lane_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego_vehicle,
    )
    lane_sensor.listen(
        lambda event: _mark_event(lane_invasion_frames, event.frame, event_lock)
    )
    sensors.append(lane_sensor)
    return sensors


def _attach_interfuser_sensors(
    modules: RuntimeModules,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
) -> tuple[SensorBuffer, list[Any]]:
    active_specs = [spec for spec in INTERFUSER_SENSOR_SPECS if spec["type"] != "sensor.speedometer"]
    buffer = SensorBuffer([str(spec["id"]) for spec in active_specs])
    sensors: list[Any] = []
    for spec in active_specs:
        blueprint = blueprint_library.find(spec["type"])
        for attr in ("width", "height", "fov", "sensor_tick", "reading_frequency"):
            if attr in spec and blueprint.has_attribute(attr):
                blueprint.set_attribute(attr, str(spec[attr]))
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
        sensor.listen(buffer.callback(sensor_id, _sensor_converter(modules, spec)))
        sensors.append(sensor)
    LOGGER.info(
        "Attached InterFuser sensor rig: %s",
        ", ".join(str(spec["id"]) for spec in INTERFUSER_SENSOR_SPECS),
    )
    return buffer, sensors


def _sensor_converter(modules: RuntimeModules, spec: Mapping[str, Any]) -> Callable[[Any], Any]:
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
    raise ValueError(f"unsupported InterFuser sensor type {sensor_type!r}")


def _trace_route(modules: RuntimeModules, world: Any, start: Any, destination: Any) -> list[Any]:
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


def _route_from_basic_agent(agent: Any) -> list[Any]:
    local_planner = getattr(agent, "_local_planner", None)
    queue = getattr(local_planner, "_waypoints_queue", None)
    if queue is None:
        return []
    return list(queue)


def _build_sparse_gps_plan(
    world: Any,
    route_transforms: list[tuple[Any, Any]],
    sample_factor: float = 50.0,
) -> list[tuple[dict[str, float], Any]]:
    sampled_indices = _downsample_route(route_transforms, sample_factor)
    sparse_route = [route_transforms[index] for index in sampled_indices]
    lat_ref, lon_ref = _get_latlon_ref(world)
    return [
        (_location_to_gps(lat_ref, lon_ref, transform.location), road_option)
        for transform, road_option in sparse_route
    ]


def _downsample_route(route: list[tuple[Any, Any]], sample_factor: float) -> list[int]:
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
            distance_since_sample += _distance(current_location, previous_location)
        prev_option = curr_option
    return sorted(set(ids_to_sample))


def _location_to_gps(lat_ref: float, lon_ref: float, location: Any) -> dict[str, float]:
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


def _get_latlon_ref(world: Any) -> tuple[float, float]:
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


def _summarize_traffic_meta(modules: RuntimeModules, np: Any, traffic_meta: Any) -> dict[str, Any]:
    meta = traffic_meta.reshape(20, 20, 7)
    weighted = meta * modules.reweight_array
    box_ids, box_info = modules.find_peak_box(weighted)
    vehicle_count = len(box_info["car"])
    bike_count = len(box_info["bike"])
    pedestrian_count = len(box_info["pedestrian"])
    density = meta[:, :, 0]
    summary = {
        "vehicle": vehicle_count,
        "bike": bike_count,
        "pedestrian": pedestrian_count,
        "occupied_cells": int(np.sum(density > 0.4)),
        "max_density": float(np.max(density)),
        "mean_density": float(np.mean(density)),
    }
    objects = []
    if vehicle_count:
        objects.append("vehicle")
    if bike_count:
        objects.append("bike")
    if pedestrian_count:
        objects.append("pedestrian")
    return {
        "object_density_summary": summary,
        "num_objects": int(len(box_ids)),
        "objects": objects,
    }


def _pool_feature(np: Any, feature: Any) -> list[float]:
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim == 0:
        return [float(array)]
    if array.ndim == 1:
        pooled = array
    else:
        pooled = array.mean(axis=tuple(range(array.ndim - 1)))
    return [float(value) for value in pooled.reshape(-1)]


def _parse_target_speed(meta_info: Any) -> float | None:
    match = re.search(r"target_speed:\s*([-+]?\d+(?:\.\d+)?)", str(meta_info))
    if match is None:
        return None
    return float(match.group(1))


def _shape_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _shape_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return type(value).__name__


def _sensor_shape_summary(packet: Mapping[str, tuple[int, Any]]) -> dict[str, Any]:
    return {
        sensor_id: {"frame": frame_id, "shape": _shape_summary(payload)}
        for sensor_id, (frame_id, payload) in packet.items()
    }


def _mark_event(event_frames: set[int], frame_id: int, event_lock: Lock) -> None:
    with event_lock:
        event_frames.add(frame_id)


def _event_seen(event_frames: set[int], frame_id: int, event_lock: Lock) -> bool:
    with event_lock:
        return frame_id in event_frames


def _ego_measurement(vehicle: Any) -> dict[str, float]:
    transform = vehicle.get_transform()
    return {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw": float(transform.rotation.yaw),
        "speed": _ego_speed(vehicle),
    }


def _ego_speed(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _build_run_id(
    scenario_id: str,
    condition: str,
    stress_type: str | None,
    severity: int,
    seed: int,
) -> str:
    condition_part = "clean" if condition == "clean" else f"{stress_type}_s{severity}"
    return f"interfuser_{scenario_id}_{condition_part}_seed{seed}"


def _print_summary(path: Path, frames: list[dict[str, Any]]) -> None:
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


def _distance(left: Any, right: Any) -> float:
    if hasattr(left, "distance"):
        return float(left.distance(right))
    return math.sqrt(
        (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
