"""Record a TransFuser closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.transfuser_adapter``. This
script is the only place that imports CARLA, torch, timm, and TransFuser code.
It mirrors ``team_code_transfuser.submission_agent`` for the sensor spec,
preprocessing, ``forward_ego`` data flow, PID control, action repeat, and GPS
route target-point logic, while writing SD2 frame logs for offline analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Mapping

from sd2.adapters.transfuser_adapter import (
    Sd2JsonlWriter,
    build_transfuser_run_metadata,
    transfuser_record_to_sd2,
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
TRANSFUSER_TEAM_CODE = (
    REPO_ROOT
    / "models"
    / "TransFuser"
    / "TransFuser_UI_V2"
    / "transfuser"
    / "team_code_transfuser"
)
DEFAULT_CHECKPOINT = Path("models/TransFuser/checkpoints/models_2022/transfuser")
DEFAULT_TARGET_SPEED_KMH = 25.0
SAFETY_BOX_MIN_POINTS = 30
LOGGER = logging.getLogger("transfuser_record")


@dataclass(frozen=True)
class RuntimeModules:
    carla: Any
    torch: Any
    np: Any
    cv2: Any
    Image: Any
    BasicAgent: type
    GlobalRoutePlanner: type
    GlobalRoutePlannerDAO: type | None
    GlobalConfig: type
    LidarCenterNet: type
    lidar_to_histogram_features: Callable[..., Any]
    draw_target_point: Callable[..., Any]


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


class TransFuserRoutePlanner:
    """RoutePlanner copy from TransFuser's submission agent, without CARLA imports."""

    def __init__(self, np: Any, min_distance: float, max_distance: float) -> None:
        self.np = np
        self.saved_route: deque[Any] = deque()
        self.route: deque[Any] = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])

    def set_route(self, global_plan: list[tuple[Any, Any]], gps: bool = False) -> None:
        self.route.clear()
        for pos, cmd in global_plan:
            if gps:
                pos = self.np.array([pos["lat"], pos["lon"]])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = self.np.array([pos.location.x, pos.location.y])
                pos -= self.mean
            self.route.append((pos, cmd))

    def run_step(self, gps: Any) -> deque[Any]:
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -self.np.inf
        cumulative_distance = 0.0
        for idx in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += self.np.linalg.norm(self.route[idx][0] - self.route[idx - 1][0])
            distance = self.np.linalg.norm(self.route[idx][0] - gps)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = idx

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route


class EgoModel:
    """Bicycle model copy from TransFuser's submission agent for GPS denoising."""

    def __init__(self, np: Any, dt: float = 1.0 / 4.0) -> None:
        self.np = np
        self.dt = dt
        self.front_wb = -0.090769015
        self.rear_wb = 1.4178275
        self.steer_gain = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs: Any, yaws: Any, spds: Any, acts: Any) -> tuple[Any, Any, Any]:
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(self.np.uint8)
        accel = self.brake_accel if brake else self.throt_accel * throt
        wheel = self.steer_gain * steer
        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaw = yaws.item()
        speed = spds.item()
        next_locs_0 = locs[0].item() + speed * math.cos(yaw + beta) * self.dt
        next_locs_1 = locs[1].item() + speed * math.sin(yaw + beta) * self.dt
        next_yaws = yaw + speed / self.rear_wb * math.sin(beta) * self.dt
        next_spds = speed + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)
        return (
            self.np.array([next_locs_0, next_locs_1]),
            self.np.array(next_yaws),
            self.np.array(next_spds),
        )


class TransFuserRuntime:
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
            LOGGER.warning("CUDA is not available; running TransFuser on %s", self.device)

        self.checkpoint_dir = args.checkpoint
        self.model_args = _load_model_args(self.checkpoint_dir)
        self.config = _build_transfuser_config(modules.GlobalConfig, self.model_args)
        self.backbone = str(self.model_args.get("backbone", "transFuser"))
        self.image_architecture = str(self.model_args.get("image_architecture", "resnet34"))
        self.lidar_architecture = str(self.model_args.get("lidar_architecture", "resnet18"))
        self.use_velocity = _as_bool(self.model_args.get("use_velocity", True))

        checkpoint_file = _select_checkpoint_file(self.checkpoint_dir)
        self.net = modules.LidarCenterNet(
            self.config,
            self.device.type,
            self.backbone,
            self.image_architecture,
            self.lidar_architecture,
            self.use_velocity,
        )
        if _as_bool(self.model_args.get("sync_batch_norm", False)):
            self.net = self.torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        state_dict = self.torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            (key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        load_result = self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.device).eval()
        param_count = sum(param.numel() for param in self.net.parameters())
        LOGGER.info(
            "Loaded TransFuser checkpoint %s | params=%.1fM missing=%d unexpected=%d",
            checkpoint_file,
            param_count / 1_000_000,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )

        self.route_planner = TransFuserRoutePlanner(
            self.np,
            self.config.route_planner_min_distance,
            self.config.route_planner_max_distance,
        )
        self.gps_buffer: deque[Any] = deque(maxlen=self.config.gps_buffer_max_len)
        self.ego_model = EgoModel(self.np, dt=self.config.carla_frame_rate)
        self.aug_degrees = [0]
        self.steer_damping = self.config.steer_damping
        self.intervention = e2e.InterventionPolicy.from_args(args, "transfuser")
        # Driving-diagnosis toggles (see README "TransFuser driving debugging").
        # The reviewer runs these live to isolate why TransFuser holds station.
        self.use_lidar_safe_check = not bool(getattr(args, "no_lidar_safe_check", False))
        self.debug_driving = bool(getattr(args, "debug_driving", False))
        creep_threshold = getattr(args, "creep_threshold", None)
        if creep_threshold is not None:
            self.config.stuck_threshold = int(creep_threshold)
        creep_duration = getattr(args, "creep_duration", None)
        if creep_duration is not None:
            self.config.creep_duration = int(creep_duration)
        # Anti-crawl: count a frame toward the stuck detector when speed is below
        # this threshold, and only reset when the ego is clearly moving above it.
        # Default 0.1 m/s keeps the original "only truly stopped counts" behavior;
        # raising it (e.g. 2.0) lets the creep controller break the cold-start
        # limit-cycle where short predicted waypoints keep the ego crawling.
        self.crawl_speed = float(getattr(args, "creep_speed", None) or 0.1)
        self.step = -1
        self.stuck_detector = 0
        self.forced_move = 0
        self.logged_shapes = False
        self.logged_detection_failure = False
        self.logged_bev_failure = False
        self.prev_control = modules.carla.VehicleControl()
        self.prev_control.steer = 0.0
        self.prev_control.throttle = 0.0
        self.prev_control.brake = 1.0
        self.prev_extracted: dict[str, Any] | None = None

    def sensor_specs(self) -> tuple[dict[str, Any], ...]:
        specs: list[dict[str, Any]] = [
            {
                "type": "sensor.camera.rgb",
                "x": self.config.camera_pos[0],
                "y": self.config.camera_pos[1],
                "z": self.config.camera_pos[2],
                "roll": self.config.camera_rot_0[0],
                "pitch": self.config.camera_rot_0[1],
                "yaw": self.config.camera_rot_0[2],
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fov": self.config.camera_fov,
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.config.camera_pos[0],
                "y": self.config.camera_pos[1],
                "z": self.config.camera_pos[2],
                "roll": self.config.camera_rot_1[0],
                "pitch": self.config.camera_rot_1[1],
                "yaw": self.config.camera_rot_1[2],
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fov": self.config.camera_fov,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.config.camera_pos[0],
                "y": self.config.camera_pos[1],
                "z": self.config.camera_pos[2],
                "roll": self.config.camera_rot_2[0],
                "pitch": self.config.camera_rot_2[1],
                "yaw": self.config.camera_rot_2[2],
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fov": self.config.camera_fov,
                "id": "rgb_right",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                # Must fire on every tick. A sensor_tick equal to the world delta eventually
                # skips a tick to floating-point accumulation, and the recorder then blocks
                # forever waiting for that frame's reading.
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
            {
                "type": "sensor.speedometer",
                "reading_frequency": self.config.carla_fps,
                "id": "speed",
            },
        ]
        if self.backbone != "latentTF":
            specs.append(
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": self.config.lidar_pos[0],
                    "y": self.config.lidar_pos[1],
                    "z": self.config.lidar_pos[2],
                    "roll": self.config.lidar_rot[0],
                    "pitch": self.config.lidar_rot[1],
                    "yaw": self.config.lidar_rot[2],
                    "id": "lidar",
                }
            )
        return tuple(specs)

    def set_global_plan(self, gps_plan: list[tuple[dict[str, float], Any]]) -> None:
        if len(gps_plan) < 2:
            raise RuntimeError("TransFuser requires at least two route points")
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
        stress_input = self._build_model_input(sensor_packet, tick_stress)
        clean_input = self._build_model_input(sensor_packet, tick_clean)
        input_shapes = _shape_summary(stress_input)
        if not self.logged_shapes:
            LOGGER.info("First TransFuser sensor packet shapes: %s", _sensor_shape_summary(sensor_packet))
            LOGGER.info("First TransFuser model input shapes: %s", input_shapes)

        is_stuck = (
            self.stuck_detector > self.config.stuck_threshold
            and self.forced_move < self.config.creep_duration
        )
        forced_move_after = self.forced_move + 1 if is_stuck else self.forced_move
        first_forced_move = is_stuck and forced_move_after == 1
        if is_stuck:
            LOGGER.info("Detected TransFuser stuck state; forced_move=%s", self.forced_move)
        safety_box = self._safety_box(tick_stress)

        stress_forward = self._forward_once(
            stress_input,
            input_shapes,
            is_stuck=is_stuck,
            first_forced_move=first_forced_move,
            safety_box=safety_box,
        )
        clean_forward = self._forward_once(
            clean_input,
            input_shapes,
            is_stuck=is_stuck,
            first_forced_move=first_forced_move,
            safety_box=safety_box,
        )

        control_hybrid_planning_clean = (
            clean_forward["control"] if self.intervention.stage == "none" else None
        )
        applied_forward = clean_forward if self.intervention.applied_source == "clean_forward" else stress_forward
        applied_control = self._control_from_waypoints(
            applied_forward["pred_wp"],
            applied_forward["gt_velocity"],
            is_stuck=is_stuck,
            first_forced_move=first_forced_move,
            safety_box=safety_box,
            preserve_state=False,
        )
        applied_forward = {**applied_forward, "control": applied_control}
        self.forced_move = forced_move_after

        speed_value = _scalar_float(stress_input["gt_velocity"])
        if speed_value < self.crawl_speed:
            self.stuck_detector += 1
        elif speed_value > self.crawl_speed and not is_stuck:
            self.stuck_detector = 0
            self.forced_move = 0

        control = self.modules.carla.VehicleControl()
        control.steer = float(applied_control["steer"])
        control.throttle = float(applied_control["throttle"])
        control.brake = float(applied_control["brake"])

        if self.debug_driving:
            target_point_np = stress_input["target_point"].detach().cpu().numpy().reshape(-1)
            debug_wp = stress_forward["pred_wp"].detach().cpu().numpy()[0]
            LOGGER.info(
                "DRIVE step=%d speed=%.3f is_stuck=%s stuck_detector=%d forced_move=%d "
                "emergency_stop=%s safety_pts=%d target_point=%s wp0=%s wp_last=%s "
                "steer=%.3f throttle=%.3f brake=%.3f",
                self.step,
                speed_value,
                is_stuck,
                self.stuck_detector,
                self.forced_move,
                applied_control["emergency_stop"],
                0 if safety_box is None else len(safety_box),
                target_point_np.round(2).tolist(),
                debug_wp[0].round(2).tolist() if len(debug_wp) else None,
                debug_wp[-1].round(2).tolist() if len(debug_wp) else None,
                control.steer,
                control.throttle,
                control.brake,
            )

        self.update_gps_buffer(control, tick_stress["compass"], tick_stress["speed"])

        pred_wp_np = stress_forward["pred_wp"].detach().cpu().numpy()[0]
        fused_feature_np = stress_forward["fused_feature"].detach().cpu().numpy()
        bev_seg_summary = self._try_bev_seg_summary(stress_forward["features"])
        rotated_records = _rotated_bb_to_records(stress_forward["rotated_bb"])

        if not self.logged_shapes:
            output_shapes = {
                "pred_wp": _shape_summary(stress_forward["pred_wp"]),
                "rotated_bb": _shape_summary(stress_forward["rotated_bb"]),
                "rotated_bb_count": len(rotated_records),
                "fused_feature": _shape_summary(stress_forward["fused_feature"]),
                "feature_source": "fused_features_before_waypoint_gru",
            }
            LOGGER.info("First TransFuser model output shapes: %s", output_shapes)
            self.logged_shapes = True

        extracted = {
            "vision": {
                "image_mean": tick_stress["image_mean"],
                "image_std": tick_stress["image_std"],
                "feature": _pool_feature(self.np, fused_feature_np),
                "feature_source": "mean_pooled_fused_features",
            },
            "semantic": {
                "rotated_bb": rotated_records,
                **({"bev_seg_summary": bev_seg_summary} if bev_seg_summary else {}),
            },
            "planning": {
                "waypoints": pred_wp_np.astype(float).tolist(),
                "target_speed": _target_speed_from_waypoints(self.np, pred_wp_np),
                "target_point": [float(value) for value in tick_stress["target_point"]],
                "command": int(tick_stress["next_command"]),
                "is_stuck": bool(is_stuck),
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "transfuser": {
                "carla_frame": int(frame_id),
                "action_repeated": False,
                "backbone": self.backbone,
                "image_architecture": self.image_architecture,
                "lidar_architecture": self.lidar_architecture,
                "use_target_point_image": bool(self.config.use_target_point_image),
                "use_point_pillars": bool(self.config.use_point_pillars),
                "safety_box_points": 0 if safety_box is None else int(len(safety_box)),
                "emergency_stop": bool(applied_control["emergency_stop"]),
            },
            "intervention": e2e.build_intervention_block(
                self.intervention,
                control_from_stress_forward=stress_forward["control"],
                control_from_clean_forward=clean_forward["control"],
                planning_waypoints_clean_forward=clean_forward["pred_wp"].detach().cpu().numpy()[0].astype(float).tolist(),
                control_hybrid_planning_clean=control_hybrid_planning_clean,
            ),
        }

        self.prev_control = control
        self.prev_extracted = dict(extracted)
        return control, extracted

    def _tick(self, input_data: dict[str, tuple[int, Any]]) -> dict[str, Any]:
        rgb_parts = []
        raw_rgb_by_pos = {}
        for pos in ("left", "front", "right"):
            sensor_id = f"rgb_{pos}"
            rgb_pos = self.cv2.cvtColor(input_data[sensor_id][1][:, :, :3], self.cv2.COLOR_BGR2RGB)
            raw_rgb_by_pos[pos] = rgb_pos
            rgb_cropped = self.scale_crop(
                self.Image.fromarray(rgb_pos),
                self.config.scale,
                self.config.img_width,
                self.config.img_width,
                self.config.img_resolution[0],
                self.config.img_resolution[0],
            )
            rgb_parts.append(rgb_cropped)
        rgb = self.np.concatenate(rgb_parts, axis=1)

        gps = input_data["gps"][1][:2]
        speed = float(input_data["speed"][1]["speed"])
        compass = float(input_data["imu"][1][-1])
        if math.isnan(compass):
            compass = 0.0

        result: dict[str, Any] = {
            "rgb": rgb,
            "gps": gps,
            "speed": speed,
            "compass": compass,
            "image_mean": float(rgb.astype(self.np.float32).mean() / 255.0),
            "image_std": float(rgb.astype(self.np.float32).std() / 255.0),
            "_raw_rgb_by_pos": raw_rgb_by_pos,
        }
        result["front_image_mean"] = float(raw_rgb_by_pos["front"].astype(self.np.float32).mean() / 255.0)
        result["front_image_std"] = float(raw_rgb_by_pos["front"].astype(self.np.float32).std() / 255.0)

        if self.backbone != "latentTF":
            result["lidar"] = input_data["lidar"][1][:, :3]

        pos = self._get_position(result)
        result["gps"] = pos
        self.gps_buffer.append(pos)
        denoised_pos = self.np.average(self.gps_buffer, axis=0)

        waypoint_route = self.route_planner.run_step(denoised_pos)
        next_wp, next_cmd = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        result["next_command"] = int(next_cmd.value)

        theta = compass + self.np.pi / 2
        rotation = self.np.array(
            [
                [self.np.cos(theta), -self.np.sin(theta)],
                [self.np.sin(theta), self.np.cos(theta)],
            ]
        )
        local_command_point = self.np.array([next_wp[0] - denoised_pos[0], next_wp[1] - denoised_pos[1]])
        local_command_point = rotation.T.dot(local_command_point)
        result["target_point"] = tuple(local_command_point)
        return result

    def _with_stressed_images(self, tick_data: dict[str, Any]) -> dict[str, Any]:
        stressed = dict(tick_data)
        rgb_parts = []
        raw_rgb_by_pos = tick_data.get("_raw_rgb_by_pos", {})
        stressed_raw_by_pos = {}
        for pos in ("left", "front", "right"):
            raw_rgb = raw_rgb_by_pos[pos]
            stressed_raw = self._apply_visual_stress(raw_rgb)
            stressed_raw_by_pos[pos] = stressed_raw
            rgb_cropped = self.scale_crop(
                self.Image.fromarray(stressed_raw),
                self.config.scale,
                self.config.img_width,
                self.config.img_width,
                self.config.img_resolution[0],
                self.config.img_resolution[0],
            )
            rgb_parts.append(rgb_cropped)
        rgb = self.np.concatenate(rgb_parts, axis=1)
        stressed["rgb"] = rgb
        stressed["_raw_rgb_by_pos"] = stressed_raw_by_pos
        stressed["image_mean"] = float(rgb.astype(self.np.float32).mean() / 255.0)
        stressed["image_std"] = float(rgb.astype(self.np.float32).std() / 255.0)
        stressed["front_image_mean"] = float(stressed_raw_by_pos["front"].astype(self.np.float32).mean() / 255.0)
        stressed["front_image_std"] = float(stressed_raw_by_pos["front"].astype(self.np.float32).std() / 255.0)
        return stressed

    def _get_position(self, tick_data: dict[str, Any]) -> Any:
        gps = tick_data["gps"]
        return (gps - self.route_planner.mean) * self.route_planner.scale

    def _build_model_input(
        self,
        sensor_packet: dict[str, tuple[int, Any]],
        tick_data: dict[str, Any],
    ) -> dict[str, Any]:
        image = self.prepare_image(tick_data)
        num_points = None
        if self.backbone == "latentTF":
            lidar_bev = self.torch.zeros(
                (1, 2, self.config.lidar_resolution_width, self.config.lidar_resolution_height),
                device=self.device,
                dtype=self.torch.float32,
            )
        elif self.config.use_point_pillars:
            lidar_cloud = deepcopy(sensor_packet["lidar"][1])
            lidar_cloud[:, 1] *= -1
            lidar_bev = [self.torch.tensor(lidar_cloud, device=self.device, dtype=self.torch.float32)]
            num_points = [
                self.torch.tensor(len(lidar_cloud), device=self.device, dtype=self.torch.int32)
            ]
        else:
            lidar_bev = self.prepare_lidar(tick_data)

        target_point_image, target_point = self.prepare_goal_location(tick_data)
        gt_velocity = self.torch.FloatTensor([tick_data["speed"]]).to(
            self.device,
            dtype=self.torch.float32,
        )
        velocity = gt_velocity.reshape(1, 1)
        return {
            "image": image,
            "lidar_bev": lidar_bev,
            "target_point_image": target_point_image,
            "target_point": target_point,
            "gt_velocity": gt_velocity,
            "velocity": velocity,
            "num_points": num_points,
        }

    def _forward_ego_with_feature(
        self,
        image: Any,
        lidar_bev: Any,
        target_point: Any,
        target_point_image: Any,
        velocity: Any,
        *,
        num_points: Any = None,
    ) -> tuple[Any, list[Any], Any, Any]:
        model_lidar = lidar_bev
        if self.config.use_point_pillars:
            model_lidar = self.net.point_pillar_net(model_lidar, num_points)
            model_lidar = self.torch.rot90(model_lidar, -1, dims=(2, 3))
        if self.config.use_target_point_image:
            model_lidar = self.torch.cat((model_lidar, target_point_image), dim=1)

        if self.backbone in {"transFuser", "late_fusion", "latentTF"}:
            features, _image_features_grid, fused_features = self.net._model(
                image,
                model_lidar,
                velocity,
            )
        else:
            raise RuntimeError(
                "experiments/transfuser_record.py supports transFuser, late_fusion, "
                f"and latentTF forward signatures; got backbone={self.backbone!r}"
            )

        pred_wp, _pred_brake, _steer, _throttle, _brake = self.net.forward_gru(
            fused_features,
            target_point,
        )
        rotated_bboxes: list[Any] = []
        try:
            preds = self.net.head([features[0]])
            results = self.net.head.get_bboxes(
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
                preds[5],
                preds[6],
            )
            bboxes, _labels = results[0]
            bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
            for bbox in bboxes.detach().cpu().numpy():
                rotated_bboxes.append(self.net.get_bbox_local_metric(bbox))
        except Exception as exc:
            if not self.logged_detection_failure:
                LOGGER.warning(
                    "TransFuser detection branch unavailable; semantic boxes will be empty: %s",
                    exc,
                )
                self.logged_detection_failure = True
            rotated_bboxes = []
        return pred_wp, rotated_bboxes, fused_features, features

    def _forward_once(
        self,
        model_input: dict[str, Any],
        input_shapes: dict[str, Any],
        *,
        is_stuck: bool,
        first_forced_move: bool,
        safety_box: Any | None,
    ) -> dict[str, Any]:
        try:
            with self.torch.no_grad():
                raw_pred_wp, rotated_bb, fused_feature, features = self._forward_ego_with_feature(
                    model_input["image"],
                    model_input["lidar_bev"],
                    model_input["target_point"],
                    model_input["target_point_image"],
                    model_input["velocity"],
                    num_points=model_input.get("num_points"),
                )
        except Exception:
            LOGGER.exception("TransFuser forward failed; model input shapes were: %s", input_shapes)
            raise

        pred_wp = self._postprocess_waypoints([raw_pred_wp])
        control = self._control_from_waypoints(
            pred_wp,
            model_input["gt_velocity"],
            is_stuck=is_stuck,
            first_forced_move=first_forced_move,
            safety_box=safety_box,
            preserve_state=True,
        )
        return {
            "pred_wp": pred_wp,
            "rotated_bb": rotated_bb,
            "fused_feature": fused_feature,
            "features": features,
            "gt_velocity": model_input["gt_velocity"],
            "control": control,
        }

    def _control_from_waypoints(
        self,
        pred_wp: Any,
        gt_velocity: Any,
        *,
        is_stuck: bool,
        first_forced_move: bool,
        safety_box: Any | None,
        preserve_state: bool,
    ) -> dict[str, Any]:
        pid_state = e2e.snapshot_pid_controllers(self.net) if preserve_state else {}
        steer, throttle, brake = self.net.control_pid(pred_wp, gt_velocity, is_stuck)
        if preserve_state:
            e2e.restore_pid_controllers(self.net, pid_state)
        if first_forced_move:
            steer = 0.0
        if _as_bool(brake) or is_stuck:
            steer = float(steer) * self.steer_damping

        emergency_stop = False
        if self.use_lidar_safe_check and safety_box is not None:
            emergency_stop = len(safety_box) >= SAFETY_BOX_MIN_POINTS
            if emergency_stop:
                throttle = 0.0
                brake = True

        return {
            "steer": _scalar_float(steer),
            "throttle": _scalar_float(throttle),
            "brake": _scalar_float(brake),
            "emergency_stop": bool(emergency_stop),
        }

    def _postprocess_waypoints(self, pred_wps: list[Any]) -> Any:
        pred_wp = self.torch.stack(pred_wps, dim=0).mean(dim=0)
        transformed = []
        for idx, degree in enumerate(self.aug_degrees):
            rad = self.np.deg2rad(degree)
            degree_matrix = self.np.array(
                [
                    [self.np.cos(rad), self.np.sin(rad)],
                    [-self.np.sin(rad), self.np.cos(rad)],
                ]
            ).T
            cur_pred_wp = pred_wp[idx].detach().cpu().numpy()
            transformed.append((degree_matrix @ cur_pred_wp.T).T)
        stacked = self.np.stack(transformed, axis=0)
        return self.torch.median(
            self.torch.from_numpy(stacked).to(self.device, dtype=self.torch.float32),
            dim=0,
            keepdims=True,
        )[0]

    def _safety_box(self, tick_data: dict[str, Any]) -> Any | None:
        if self.backbone == "latentTF" or "lidar" not in tick_data:
            return None
        safety_box = deepcopy(tick_data["lidar"])
        safety_box[:, 1] *= -1
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        return safety_box

    def _try_bev_seg_summary(self, features: Any) -> dict[str, Any] | None:
        try:
            pred_bev = self.net.pred_bev(features[0])
            pred_bev = self.torch.nn.functional.interpolate(
                pred_bev,
                (self.config.bev_resolution_height, self.config.bev_resolution_width),
                mode="bilinear",
                align_corners=True,
            )
            prediction = self.torch.argmax(pred_bev, dim=1).detach().cpu().numpy()[0]
            counts = self.np.bincount(prediction.reshape(-1), minlength=3)
            total = int(prediction.size)
            return {
                "class_0": int(counts[0]),
                "class_1": int(counts[1]),
                "class_2": int(counts[2]),
                "nonzero_fraction": float((total - int(counts[0])) / max(1, total)),
                "dominant_class": int(self.np.argmax(counts)),
            }
        except Exception as exc:
            if not self.logged_bev_failure:
                LOGGER.warning("TransFuser BEV summary unavailable: %s", exc)
                self.logged_bev_failure = True
            return None

    def prepare_image(self, tick_data: dict[str, Any]) -> Any:
        image = self.Image.fromarray(tick_data["rgb"])
        image_degrees = []
        for degree in self.aug_degrees:
            crop_shift = degree / 60 * self.config.img_width
            rgb = self.torch.from_numpy(
                self.shift_x_scale_crop(
                    image,
                    scale=self.config.scale,
                    crop=self.config.img_resolution,
                    crop_shift=crop_shift,
                )
            ).unsqueeze(0)
            image_degrees.append(rgb.to(self.device, dtype=self.torch.float32))
        return self.torch.cat(image_degrees, dim=0)

    def prepare_lidar(self, tick_data: dict[str, Any]) -> Any:
        lidar_transformed = deepcopy(tick_data["lidar"])
        lidar_transformed[:, 1] *= -1
        lidar_transformed = self.torch.from_numpy(
            self.modules.lidar_to_histogram_features(lidar_transformed)
        ).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to(self.device, dtype=self.torch.float32)]
        return self.torch.cat(lidar_transformed_degrees[::-1], dim=1)

    def prepare_goal_location(self, tick_data: dict[str, Any]) -> tuple[Any, Any]:
        target_point_values = [
            self.torch.FloatTensor([tick_data["target_point"][0]]),
            self.torch.FloatTensor([tick_data["target_point"][1]]),
        ]
        target_point = self.torch.stack(target_point_values, dim=1).to(
            self.device,
            dtype=self.torch.float32,
        )

        target_point_image_degrees = []
        target_point_degrees = []
        for degree in self.aug_degrees:
            rad = self.np.deg2rad(degree)
            degree_matrix = self.np.array(
                [
                    [self.np.cos(rad), self.np.sin(rad)],
                    [-self.np.sin(rad), self.np.cos(rad)],
                ]
            )
            current_target_point = (degree_matrix @ target_point[0].cpu().numpy().reshape(2, 1)).T
            target_point_image = self.modules.draw_target_point(current_target_point[0])
            target_point_image = self.torch.from_numpy(target_point_image)[None].to(
                self.device,
                dtype=self.torch.float32,
            )
            target_point_image_degrees.append(target_point_image)
            target_point_degrees.append(self.torch.from_numpy(current_target_point))
        target_point_image = self.torch.cat(target_point_image_degrees, dim=0)
        target_point = self.torch.cat(target_point_degrees, dim=0).to(
            self.device,
            dtype=self.torch.float32,
        )
        return target_point_image, target_point

    def update_gps_buffer(self, control: Any, theta: float, speed: float) -> None:
        yaw = self.np.array([(theta - self.np.pi / 2.0)])
        speed_array = self.np.array([speed])
        action = self.np.array(
            self.np.stack([control.steer, control.throttle, control.brake], axis=-1)
        )
        for idx in range(len(self.gps_buffer)):
            loc = self.gps_buffer[idx]
            loc_temp = self.np.array([loc[1], -loc[0]])
            next_loc_tmp, _next_yaw, _next_speed = self.ego_model.forward(
                loc_temp,
                yaw,
                speed_array,
                action,
            )
            self.gps_buffer[idx] = self.np.array([-next_loc_tmp[1], next_loc_tmp[0]])

    def scale_crop(
        self,
        image: Any,
        scale: float = 1,
        start_x: int = 0,
        crop_x: int | None = None,
        start_y: int = 0,
        crop_y: int | None = None,
    ) -> Any:
        width, height = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((int(width), int(height)))
        if crop_x is None:
            crop_x = int(width)
        if crop_y is None:
            crop_y = int(height)
        image_array = self.np.asarray(image)
        return image_array[start_y : start_y + crop_y, start_x : start_x + crop_x]

    def shift_x_scale_crop(
        self,
        image: Any,
        scale: float,
        crop: tuple[int, int],
        crop_shift: float = 0,
    ) -> Any:
        crop_h, crop_w = crop
        width, height = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image_array = self.np.array(im_resized)
        start_y = height // 2 - crop_h // 2
        start_x = width // 2 - crop_w // 2
        start_x += int(crop_shift // scale)
        cropped_image = image_array[start_y : start_y + crop_h, start_x : start_x + crop_w]
        return self.np.transpose(cropped_image, (2, 0, 1))

    def _apply_visual_stress(self, image: Any) -> Any:
        if self.stressor is None:
            return image
        return self.stressor.apply_image(image, self.args.stress_severity, self.stress_rng)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging()
    modules = _import_runtime_modules()
    stressor, stress_rng = _build_image_stressor(args, modules)
    rng = random.Random(args.seed)

    client = modules.carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = None
    traffic_manager = None
    ego_vehicle = None
    sensors: list[Any] = []
    frames: list[dict[str, Any]] = []
    collision_frames: set[int] = set()
    lane_invasion_frames: set[int] = set()
    event_lock = Lock()

    try:
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
        destination_index = _select_destination_index(spawn_index, len(spawn_points))
        ego_vehicle = _spawn_ego_vehicle(
            world,
            blueprint_library,
            spawn_points,
            spawn_index,
            rng,
        )
        world.tick()

        runtime = TransFuserRuntime(args, modules, stressor, stress_rng)
        event_sensors = _attach_event_sensors(
            modules.carla,
            world,
            blueprint_library,
            ego_vehicle,
            collision_frames,
            lane_invasion_frames,
            event_lock,
        )
        sensor_buffer, transfuser_sensors = _attach_transfuser_sensors(
            modules,
            world,
            blueprint_library,
            ego_vehicle,
            runtime.sensor_specs(),
        )
        sensors = [*event_sensors, *transfuser_sensors]

        destination = spawn_points[destination_index].location
        basic_agent = modules.BasicAgent(ego_vehicle, target_speed=DEFAULT_TARGET_SPEED_KMH)
        basic_agent.set_destination(destination)
        route = _trace_route(modules, world, ego_vehicle.get_location(), destination)
        if not route:
            route = _route_from_basic_agent(basic_agent)
        route_transforms = [(waypoint.transform, road_option) for waypoint, road_option in route]
        if len(route_transforms) < 2:
            raise RuntimeError("failed to build a TransFuser global route")

        gps_plan = _build_sparse_gps_plan(world, route_transforms)
        route_locations = [transform.location for transform, _road_option in route_transforms]
        progress_tracker = RouteProgressTracker(route_locations)
        runtime.set_global_plan(gps_plan)

        LOGGER.info(
            "Route ready: town=%s spawn=%d dest=%d dense_points=%d sparse_points=%d",
            args.town,
            spawn_index,
            destination_index,
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
        intervention_policy = e2e.InterventionPolicy.from_args(args, "transfuser")
        condition, stress_type, severity = intervention_policy.metadata_condition
        run_id = _build_run_id(scenario_id, condition, stress_type, severity, args.seed)
        if intervention_policy.enabled:
            run_id = f"{run_id}_iv-{intervention_policy.direction}-{intervention_policy.stage}"
        metadata = build_transfuser_run_metadata(
            run_id=run_id,
            scenario_id=scenario_id,
            condition=condition,
            stress_type=stress_type,
            severity=severity,
            seed=args.seed,
            town=args.town,
        )

        with Sd2JsonlWriter(args.output, metadata) as jsonl_writer:
            for frame_idx in range(args.frames):
                frame_id = world.tick()
                packet = sensor_buffer.read(frame_id)
                packet["speed"] = (frame_id, {"speed": _ego_speed(ego_vehicle)})
                control, extracted = runtime.run_step(
                    packet,
                    timestamp=frame_idx * args.delta,
                    frame_id=frame_id,
                )
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
                frame = transfuser_record_to_sd2(extracted, run_id=run_id)
                jsonl_writer.write_frame(frame)
                frames.append(frame)

        e2e.write_intervention_sidecar(args.output, intervention_policy)
        _print_summary(args.output, frames)
        return 0
    finally:
        _cleanup(modules.carla if "modules" in locals() else None, world, traffic_manager, sensors, ego_vehicle)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a TransFuser synchronous CARLA run as SD2 JSONL.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--stress",
        choices=["none", "gaussian_noise", "motion_blur", "brightness", "fog"],
        default="none",
    )
    parser.add_argument("--stress-severity", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=0)
    e2e.add_intervention_args(parser)
    # Driving-diagnosis toggles for the "TransFuser holds station" investigation.
    parser.add_argument(
        "--debug-driving",
        action="store_true",
        help="log per-tick driving state (speed, is_stuck, emergency_stop, waypoints, control)",
    )
    parser.add_argument(
        "--no-lidar-safe-check",
        action="store_true",
        help="disable the LiDAR safety-box emergency brake to test if it causes station-holding",
    )
    parser.add_argument(
        "--creep-threshold",
        type=int,
        default=None,
        help="override config.stuck_threshold (frames of ~zero speed before creep engages)",
    )
    parser.add_argument(
        "--creep-duration",
        type=int,
        default=None,
        help="override config.creep_duration (frames the forced-move creep is applied)",
    )
    parser.add_argument(
        "--creep-speed",
        type=float,
        default=None,
        help="speed (m/s) below which a frame counts as crawling toward the creep "
        "trigger; default 0.1 (only truly stopped). Set e.g. 2.0 with a low "
        "--creep-threshold to break the cold-start crawl limit-cycle.",
    )
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.delta <= 0:
        parser.error("--delta must be positive")
    if args.stress != "none":
        try:
            validate_severity(args.stress_severity)
        except ValueError as exc:
            parser.error(str(exc))
    e2e.validate_intervention_args(parser, args, "transfuser")
    return args


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _apply_transfuser_preamble() -> None:
    for path in (CARLA_PYTHON_API, TRANSFUSER_TEAM_CODE):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _import_runtime_modules() -> RuntimeModules:
    _apply_transfuser_preamble()

    import carla
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    if "int" not in np.__dict__:
        np.int = int

    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    except ImportError:
        GlobalRoutePlannerDAO = None

    from config import GlobalConfig
    from data import draw_target_point, lidar_to_histogram_features
    from model import LidarCenterNet

    return RuntimeModules(
        carla=carla,
        torch=torch,
        np=np,
        cv2=cv2,
        Image=Image,
        BasicAgent=BasicAgent,
        GlobalRoutePlanner=GlobalRoutePlanner,
        GlobalRoutePlannerDAO=GlobalRoutePlannerDAO,
        GlobalConfig=GlobalConfig,
        LidarCenterNet=LidarCenterNet,
        lidar_to_histogram_features=lidar_to_histogram_features,
        draw_target_point=draw_target_point,
    )


def _load_model_args(checkpoint_dir: Path) -> dict[str, Any]:
    args_path = checkpoint_dir / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"TransFuser checkpoint args file not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_transfuser_config(config_cls: type, model_args: Mapping[str, Any]) -> Any:
    config = config_cls(setting="eval")
    if "sync_batch_norm" in model_args:
        config.sync_batch_norm = _as_bool(model_args["sync_batch_norm"])
    if "use_point_pillars" in model_args:
        config.use_point_pillars = _as_bool(model_args["use_point_pillars"])
    if "n_layer" in model_args:
        config.n_layer = int(model_args["n_layer"])
    if "use_target_point_image" in model_args:
        config.use_target_point_image = _as_bool(model_args["use_target_point_image"])
    return config


def _select_checkpoint_file(checkpoint_dir: Path) -> Path:
    preferred = checkpoint_dir / "model_seed1_39.pth"
    if preferred.is_file():
        return preferred
    candidates = sorted(checkpoint_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no .pth TransFuser checkpoints found in {checkpoint_dir}")
    LOGGER.warning(
        "Preferred TransFuser checkpoint %s not found; using %s",
        preferred.name,
        candidates[0].name,
    )
    return candidates[0]


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


def _attach_transfuser_sensors(
    modules: RuntimeModules,
    world: Any,
    blueprint_library: Any,
    ego_vehicle: Any,
    sensor_specs: tuple[dict[str, Any], ...],
) -> tuple[SensorBuffer, list[Any]]:
    active_specs = [spec for spec in sensor_specs if spec["type"] != "sensor.speedometer"]
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
        "Attached TransFuser sensor rig: %s",
        ", ".join(str(spec["id"]) for spec in sensor_specs),
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
    raise ValueError(f"unsupported TransFuser sensor type {sensor_type!r}")


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


def _rotated_bb_to_records(rotated_bb: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _tolist(rotated_bb) or []:
        item = _tolist(item)
        bbox = item
        brake = None
        confidence = None
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            bbox = item[0]
            brake = _optional_float(item[1])
            confidence = _optional_float(item[2])
        record: dict[str, Any] = {
            "class": "vehicle",
            "bbox": _jsonable(bbox),
        }
        if brake is not None:
            record["brake"] = brake
        if confidence is not None:
            record["confidence"] = confidence
        records.append(record)
    return records


def _pool_feature(np: Any, feature: Any) -> list[float]:
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


def _target_speed_from_waypoints(np: Any, waypoints: Any) -> float | None:
    array = np.asarray(waypoints, dtype=np.float32)
    if array.ndim < 2 or array.shape[0] < 2:
        return None
    return float(np.linalg.norm(array[0] - array[1]) * 2.0)


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
    return f"transfuser_{scenario_id}_{condition_part}_seed{seed}"


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


def _cleanup(
    carla: Any | None,
    world: Any | None,
    traffic_manager: Any | None,
    sensors: list[Any],
    ego_vehicle: Any | None,
) -> None:
    del carla
    for sensor in sensors:
        try:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        except RuntimeError:
            pass

    try:
        if ego_vehicle is not None and ego_vehicle.is_alive:
            ego_vehicle.destroy()
    except RuntimeError:
        pass

    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except RuntimeError:
            pass


def _distance(left: Any, right: Any) -> float:
    if hasattr(left, "distance"):
        return float(left.distance(right))
    return math.sqrt(
        (left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2
    )


def _jsonable(value: Any) -> Any:
    value = _tolist(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "detach"):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
