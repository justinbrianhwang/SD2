"""Record a TCP Bench2Drive-variant closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.tcp_adapter``. This script is
the only place that imports CARLA, torch, torchvision, and TCP code. It follows
the TCP Bench2Drive load recipe while reusing SD2's CARLA route, target-point,
warmup, event-sensor, stressor, route-progress, and JSONL scaffolding.

TCP's Bench2Drive agent builds a three-front-camera nuScenes-style mosaic. This
recorder intentionally uses one front RGB camera already sized to the model
input, then normalizes it to a ``(1, 3, 256, 900)`` tensor. That keeps clean and
stress SD2 runs comparable to the other local E2E integrations while preserving
the model's expected tensor shape.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd2.adapters.tcp_adapter import (
    TCP_MODEL_ID,
    Sd2JsonlWriter,
    build_tcp_run_metadata,
    tcp_record_to_sd2,
)
from sd2.stressors import ImageStressor, validate_severity

try:
    import _carla_e2e_common as e2e
except ImportError:
    from experiments import _carla_e2e_common as e2e


MODEL_SRC = e2e.REPO_ROOT / "models" / "TCP" / "Bench2DriveZoo_tcp_admlp"
DEFAULT_CHECKPOINT = Path("models/TCP/checkpoints/tcp_b2d.ckpt")
PLANNER_TYPES = ("only_traj", "only_ctrl", "merge_ctrl_traj")
LOGGER = logging.getLogger("tcp_record")


@dataclass(frozen=True)
class RuntimeModules:
    carla: Any
    torch: Any
    np: Any
    cv2: Any
    T: Any
    BasicAgent: type
    GlobalRoutePlanner: type
    GlobalRoutePlannerDAO: type | None
    TCP: type
    GlobalConfig: type
    RoutePlanner: type
    tcp_model_module: Any


class SD2RoutePlanner:
    """RoutePlanner-compatible helper using the existing SD2 GPS target-point path."""

    def __init__(self, np: Any, min_distance: float, max_distance: float) -> None:
        self.np = np
        self.route: deque[Any] = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])

    def set_route(self, global_plan: list[tuple[dict[str, float], Any]], gps: bool = False) -> None:
        self.route.clear()
        for pos, cmd in global_plan:
            if gps:
                route_pos = self.np.array([pos["lat"], pos["lon"]])
                route_pos -= self.mean
                route_pos *= self.scale
            else:
                route_pos = self.np.array([pos.location.x, pos.location.y])
                route_pos -= self.mean
            self.route.append((route_pos, cmd))

    def run_step(self, gps: Any) -> tuple[Any, Any]:
        if not self.route:
            raise RuntimeError("TCP route planner has no route")
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -self.np.inf
        cumulative_distance = 0.0
        for idx in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += self.np.linalg.norm(
                self.route[idx][0] - self.route[idx - 1][0]
            )
            distance = self.np.linalg.norm(self.route[idx][0] - gps)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = idx

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route[1]


class TCPRuntime:
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
        self.stressor = stressor
        self.stress_rng = stress_rng
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            LOGGER.warning("CUDA is not available; running TCP on %s", self.device)

        self.config = modules.GlobalConfig()
        self.config.seq_len = 1
        self.config.pred_len = 4
        self.net = self._construct_tcp_model()
        checkpoint = self._load_checkpoint_state(args.checkpoint)
        load_result = self.net.load_state_dict(checkpoint, strict=False)
        self.net.to(self.device).eval()
        param_count = sum(param.numel() for param in self.net.parameters())
        LOGGER.info(
            "Loaded TCP checkpoint %s | params=%.1fM missing=%d unexpected=%d",
            args.checkpoint,
            param_count / 1_000_000,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )
        if getattr(load_result, "missing_keys", None):
            LOGGER.warning("TCP checkpoint missing keys: %s", load_result.missing_keys[:20])
        if getattr(load_result, "unexpected_keys", None):
            LOGGER.warning("TCP checkpoint unexpected keys: %s", load_result.unexpected_keys[:20])

        self.im_transform = modules.T.Compose(
            [
                modules.T.ToTensor(),
                modules.T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.route_planner = SD2RoutePlanner(self.np, 4.0, 50.0)
        self.intervention = e2e.InterventionPolicy.from_args(args, TCP_MODEL_ID)
        self.step = -1
        self.logged_shapes = False
        self.logged_bad_command = False
        self.perception_feature: Any | None = None
        self.perception_hook = self.net.perception.register_forward_hook(
            self._capture_perception_feature
        )

    def sensor_specs(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 900,
                "height": 256,
                "fov": 70,
                "id": "rgb_front",
            },
            {
                "type": "sensor.other.imu",
                "x": -1.4,
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
                "x": -1.4,
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

    def set_global_plan(self, gps_plan: list[tuple[dict[str, float], Any]]) -> None:
        if len(gps_plan) < 2:
            raise RuntimeError("TCP requires at least two route points")
        self.route_planner.set_route(gps_plan, True)

    def run_step(
        self,
        sensor_packet: dict[str, tuple[int, Any]],
        timestamp: float,
        frame_id: int,
    ) -> tuple[Any, dict[str, Any]]:
        del timestamp
        self.step += 1
        tick_clean = self._tick(sensor_packet)
        tick_stress = self._with_stressed_image(tick_clean)

        input_shapes = {
            "rgb": "clean/stress dual tensors",
            "target_point": e2e.shape_summary(tick_stress["target_point"]),
            "gt_velocity": e2e.shape_summary(tick_stress["speed"]),
        }
        if not self.logged_shapes:
            LOGGER.info("First TCP sensor packet shapes: %s", e2e.sensor_shape_summary(sensor_packet))
            LOGGER.info("First TCP model input shapes: %s", input_shapes)

        if self.step < self.config.seq_len:
            control = self.modules.carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control, self._warmup_record(tick_stress, control, frame_id)

        stress_forward = self._forward_once(tick_stress, input_shapes)
        clean_forward = self._forward_once(tick_clean, input_shapes)
        control_hybrid_planning_clean = (
            clean_forward["control"] if self.intervention.stage == "none" else None
        )
        applied_forward = clean_forward if self.intervention.applied_source == "clean_forward" else stress_forward
        applied_control, applied_details = self._control_from_prediction(
            applied_forward["pred"],
            applied_forward["pred_wp"],
            applied_forward["gt_velocity"],
            applied_forward["target_point"],
            applied_forward["next_command"],
            applied_forward["speed"],
            preserve_state=False,
        )
        applied_forward = {
            **applied_forward,
            "control": applied_control,
            "control_details": applied_details,
        }

        control = self.modules.carla.VehicleControl()
        control.steer = float(applied_control["steer"])
        control.throttle = float(applied_control["throttle"])
        control.brake = float(applied_control["brake"])

        pred_wp_np = stress_forward["pred_wp_np"]
        feature_np = stress_forward["feature_np"]
        if not self.logged_shapes:
            LOGGER.info(
                "First TCP model output shapes: %s",
                {
                    "perception_feature": e2e.shape_summary(stress_forward["feature"]),
                    "pred": e2e.shape_summary(stress_forward["pred"]),
                    "feature_source": "mean_pooled_perception_feature",
                    "planner_type": self.args.planner_type,
                },
            )
            self.logged_shapes = True

        target_speed = e2e.optional_float(stress_forward["control_details"]["traj_metadata"].get("desired_speed"))
        if target_speed is None:
            target_speed = e2e.target_speed_from_waypoints(self.np, pred_wp_np)

        extracted = {
            "vision": {
                "image_mean": tick_stress["image_mean"],
                "image_std": tick_stress["image_std"],
                "feature": (
                    e2e.pool_feature(self.np, feature_np)
                    if feature_np is not None
                    else [tick_data["image_mean"], tick_data["image_std"]]
                ),
                "feature_source": (
                    "mean_pooled_perception_feature"
                    if feature_np is not None
                    else "image_stats"
                ),
                "input_simplification": "single_front_rgb_256x900",
            },
            "planning": {
                "waypoints": pred_wp_np.astype(float).tolist(),
                "target_speed": target_speed,
                "target_point": tick_stress["target_point"].astype(float).tolist(),
                "command": int(tick_stress["next_command"]),
                "planning_source": "pred_wp",
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "planner_type": self.args.planner_type,
                "details": {
                    "selected_branch": self.args.planner_type,
                    "traj_branch": applied_details["traj_branch"],
                    "ctrl_branch": applied_details["ctrl_branch"],
                    "pre_clamp": applied_details["pre_clamp"],
                    "post_clamp": applied_details["post_clamp"],
                    "traj_metadata": e2e.jsonable(applied_details["traj_metadata"]),
                    "ctrl_metadata": e2e.jsonable(applied_details["ctrl_metadata"]),
                },
            },
            "tcp": {
                "carla_frame": int(frame_id),
                "input_simplification": "single_front_rgb_256x900",
            },
            "intervention": e2e.build_intervention_block(
                self.intervention,
                control_from_stress_forward=stress_forward["control"],
                control_from_clean_forward=clean_forward["control"],
                planning_waypoints_clean_forward=clean_forward["pred_wp_np"].astype(float).tolist(),
                control_hybrid_planning_clean=control_hybrid_planning_clean,
            ),
        }
        return control, extracted

    def _construct_tcp_model(self) -> Any:
        try:
            return self.modules.TCP(self.config)
        except Exception as exc:
            LOGGER.warning(
                "TCP constructor failed with pretrained ResNet path; retrying with "
                "pretrained=False before checkpoint load: %s",
                exc,
            )
            original_resnet34 = self.modules.tcp_model_module.resnet34

            def _resnet34_no_pretrained(*args: Any, **kwargs: Any) -> Any:
                kwargs["pretrained"] = False
                return original_resnet34(*args, **kwargs)

            self.modules.tcp_model_module.resnet34 = _resnet34_no_pretrained
            try:
                return self.modules.TCP(self.config)
            finally:
                self.modules.tcp_model_module.resnet34 = original_resnet34

    def _load_checkpoint_state(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = self.torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, Mapping) else checkpoint
        return {str(key).replace("model.", ""): value for key, value in state_dict.items()}

    def _capture_perception_feature(self, _module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)) and output:
            self.perception_feature = output[0]
        else:
            self.perception_feature = output

    def _tick(self, input_data: dict[str, tuple[int, Any]]) -> dict[str, Any]:
        rgb = self.cv2.cvtColor(
            input_data["rgb_front"][1][:, :, :3],
            self.cv2.COLOR_BGR2RGB,
        )
        if rgb.shape[:2] != (256, 900):
            rgb = self.cv2.resize(rgb, (900, 256), interpolation=self.cv2.INTER_LINEAR)

        gps = input_data["gps"][1][:2]
        speed = float(input_data["speed"][1]["speed"])
        compass = float(input_data["imu"][1][-1])
        if math.isnan(compass):
            compass = 0.0

        pos = e2e.route_position(self.np, self.route_planner, gps)
        next_wp, next_cmd = self.route_planner.run_step(pos)
        target_point = e2e.route_target_point(self.np, pos, compass, next_wp)
        rgb_float = rgb.astype(self.np.float32)
        return {
            "rgb": rgb,
            "gps": pos,
            "speed": speed,
            "compass": compass,
            "next_command": int(getattr(next_cmd, "value", next_cmd)),
            "target_point": target_point,
            "image_mean": float(rgb_float.mean() / 255.0),
            "image_std": float(rgb_float.std() / 255.0),
        }

    def _with_stressed_image(self, tick_data: dict[str, Any]) -> dict[str, Any]:
        stressed = dict(tick_data)
        rgb = e2e.apply_visual_stress(
            tick_data["rgb"],
            self.stressor,
            self.args.stress_severity,
            self.stress_rng,
        )
        if rgb.shape[:2] != (256, 900):
            rgb = self.cv2.resize(rgb, (900, 256), interpolation=self.cv2.INTER_LINEAR)
        rgb_float = rgb.astype(self.np.float32)
        stressed["rgb"] = rgb
        stressed["image_mean"] = float(rgb_float.mean() / 255.0)
        stressed["image_std"] = float(rgb_float.std() / 255.0)
        return stressed

    def _forward_once(
        self,
        tick_data: dict[str, Any],
        input_shapes: dict[str, Any],
    ) -> dict[str, Any]:
        rgb_tensor = self._rgb_tensor(tick_data["rgb"])
        gt_velocity = self.torch.tensor(
            [tick_data["speed"]],
            device=self.device,
            dtype=self.torch.float32,
        )
        target_point = self.torch.tensor(
            [[tick_data["target_point"][0], tick_data["target_point"][1]]],
            device=self.device,
            dtype=self.torch.float32,
        )
        command_index = self._command_index(tick_data["next_command"])
        cmd_one_hot = self.torch.zeros((1, 6), device=self.device, dtype=self.torch.float32)
        cmd_one_hot[0, command_index] = 1.0
        speed_state = self.torch.tensor(
            [[float(tick_data["speed"]) / 12.0]],
            device=self.device,
            dtype=self.torch.float32,
        )
        state = self.torch.cat([speed_state, target_point, cmd_one_hot], dim=1)

        self.perception_feature = None
        try:
            with self.torch.no_grad():
                pred = self.net(rgb_tensor, state, target_point)
        except Exception:
            LOGGER.exception("TCP forward failed; model input shapes were: %s", input_shapes)
            raise

        pred_wp = pred["pred_wp"]
        pred_wp_np = pred_wp.detach().cpu().numpy()[0].copy()
        feature = self.perception_feature
        feature_np = feature.detach().cpu().numpy() if feature is not None else None
        control, details = self._control_from_prediction(
            pred,
            pred_wp,
            gt_velocity,
            target_point,
            int(tick_data["next_command"]),
            float(tick_data["speed"]),
            preserve_state=True,
        )
        return {
            "pred": pred,
            "pred_wp": pred_wp,
            "pred_wp_np": pred_wp_np,
            "gt_velocity": gt_velocity,
            "target_point": target_point,
            "next_command": int(tick_data["next_command"]),
            "speed": float(tick_data["speed"]),
            "feature": feature,
            "feature_np": feature_np,
            "control": control,
            "control_details": details,
        }

    def _control_from_prediction(
        self,
        pred: Mapping[str, Any],
        pred_wp: Any,
        gt_velocity: Any,
        target_point: Any,
        next_command: int,
        speed: float,
        *,
        preserve_state: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        pid_state = e2e.snapshot_pid_controllers(self.net) if preserve_state else {}
        # process_action does float(speed.cpu().numpy()); numpy>=2 rejects that on
        # a shape-(1,) array, so hand it a 0-dim scalar tensor. control_pid below
        # still needs the shape-(1,) tensor because it indexes velocity[0].
        steer_ctrl, throttle_ctrl, brake_ctrl, ctrl_metadata = self.net.process_action(
            pred,
            int(next_command),
            gt_velocity.reshape(()),
            target_point,
        )
        steer_traj, throttle_traj, brake_traj, traj_metadata = self.net.control_pid(
            pred_wp.detach().clone(),
            gt_velocity.detach().clone(),
            target_point.detach().clone(),
        )
        if preserve_state:
            e2e.restore_pid_controllers(self.net, pid_state)

        traj_branch = {
            "steer": e2e.scalar_float(steer_traj),
            "throttle": e2e.scalar_float(throttle_traj),
            "brake": e2e.scalar_float(brake_traj),
        }
        ctrl_branch = {
            "steer": e2e.scalar_float(steer_ctrl),
            "throttle": e2e.scalar_float(throttle_ctrl),
            "brake": e2e.scalar_float(brake_ctrl),
        }
        selected = self._select_control(traj_branch, ctrl_branch)
        pre_clamp = dict(selected)
        post_clamp = self._apply_tcp_clamps(selected, speed)
        control = {
            "steer": float(post_clamp["steer"]),
            "throttle": float(post_clamp["throttle"]),
            "brake": float(post_clamp["brake"]),
        }
        return control, {
            "traj_branch": traj_branch,
            "ctrl_branch": ctrl_branch,
            "pre_clamp": pre_clamp,
            "post_clamp": post_clamp,
            "traj_metadata": traj_metadata,
            "ctrl_metadata": ctrl_metadata,
        }

    def _rgb_tensor(self, rgb: Any) -> Any:
        rgb = self.np.ascontiguousarray(rgb)
        return self.im_transform(rgb).unsqueeze(0).to(self.device, dtype=self.torch.float32)

    def _command_index(self, raw_command: Any) -> int:
        command = int(getattr(raw_command, "value", raw_command))
        if command < 0:
            command = 4
        command -= 1
        if 0 <= command <= 5:
            return command
        if not self.logged_bad_command:
            LOGGER.warning("TCP received out-of-range route command %r; using lane-follow", raw_command)
            self.logged_bad_command = True
        return 3

    def _select_control(
        self,
        traj_branch: dict[str, float],
        ctrl_branch: dict[str, float],
    ) -> dict[str, float]:
        if self.args.planner_type == "only_traj":
            return {
                "steer": e2e.clamp(traj_branch["steer"], -1.0, 1.0),
                "throttle": e2e.clamp(traj_branch["throttle"], 0.0, 0.75),
                "brake": e2e.clamp(traj_branch["brake"], 0.0, 1.0),
            }
        if self.args.planner_type == "only_ctrl":
            return {
                "steer": e2e.clamp(ctrl_branch["steer"], -1.0, 1.0),
                "throttle": e2e.clamp(ctrl_branch["throttle"], 0.0, 0.75),
                "brake": e2e.clamp(ctrl_branch["brake"], 0.0, 1.0),
            }

        alpha = 0.5
        return {
            "steer": e2e.clamp(
                alpha * traj_branch["steer"] + (1.0 - alpha) * ctrl_branch["steer"],
                -1.0,
                1.0,
            ),
            "throttle": e2e.clamp(
                alpha * traj_branch["throttle"] + (1.0 - alpha) * ctrl_branch["throttle"],
                0.0,
                0.75,
            ),
            "brake": max(
                e2e.clamp(ctrl_branch["brake"], 0.0, 1.0),
                e2e.clamp(traj_branch["brake"], 0.0, 1.0),
            ),
        }

    def _apply_tcp_clamps(self, selected: dict[str, float], speed: float) -> dict[str, float]:
        steer = float(selected["steer"])
        throttle = float(selected["throttle"])
        brake = float(selected["brake"])
        speed_threshold = 1.0 if abs(steer) > 0.07 else 1.5
        max_throttle = 0.05 if speed > speed_threshold else 0.5
        throttle = e2e.clamp(throttle, 0.0, max_throttle)
        if brake > 0.0:
            brake = 1.0
        if brake > 0.5:
            throttle = 0.0
        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "speed_threshold": speed_threshold,
            "max_throttle": max_throttle,
        }

    def _warmup_record(self, tick_data: dict[str, Any], control: Any, frame_id: int) -> dict[str, Any]:
        neutral = e2e.control_to_dict(control)
        return {
            "vision": {
                "image_mean": tick_data["image_mean"],
                "image_std": tick_data["image_std"],
                "input_simplification": "single_front_rgb_256x900",
            },
            "planning": {
                "target_point": tick_data["target_point"].astype(float).tolist(),
                "command": int(tick_data["next_command"]),
                "planning_source": "warmup_buffer",
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "planner_type": self.args.planner_type,
                "details": {
                    "selected_branch": self.args.planner_type,
                    "warmup": True,
                },
            },
            "tcp": {
                "carla_frame": int(frame_id),
                "warmup": True,
                "input_simplification": "single_front_rgb_256x900",
            },
            "intervention": e2e.build_intervention_block(
                self.intervention,
                control_from_stress_forward=neutral,
                control_from_clean_forward=neutral,
                planning_waypoints_clean_forward=None,
            ),
        }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["PLANNER_TYPE"] = args.planner_type
    e2e.configure_logging()
    modules = _import_runtime_modules()
    return e2e.run_recording(
        args,
        modules,
        TCPRuntime,
        model_id=TCP_MODEL_ID,
        model_label="TCP",
        record_to_sd2=tcp_record_to_sd2,
        build_run_metadata=build_tcp_run_metadata,
        jsonl_writer_cls=Sd2JsonlWriter,
        logger=LOGGER,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a TCP Bench2Drive synchronous CARLA run as SD2 JSONL.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--planner-type", choices=PLANNER_TYPES, default="only_traj")
    parser.add_argument("--stress", choices=e2e.STRESS_CHOICES, default="none")
    parser.add_argument("--stress-severity", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-index", type=int, default=0)
    e2e.add_intervention_args(parser)
    # Anti-crawl driving aid (see experiments/_carla_e2e_common.run_recording).
    parser.add_argument("--anti-crawl", action="store_true")
    parser.add_argument("--creep-speed", type=float, default=2.0)
    parser.add_argument("--creep-frames", type=int, default=5)
    parser.add_argument("--creep-throttle", type=float, default=0.6)
    parser.add_argument("--creep-duration", type=int, default=40)
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
    e2e.validate_intervention_args(parser, args, TCP_MODEL_ID)
    return args


def _apply_model_preamble() -> None:
    for path in (e2e.CARLA_PYTHON_API, MODEL_SRC):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _import_runtime_modules() -> RuntimeModules:
    _apply_model_preamble()

    import carla
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms as T

    if "int" not in np.__dict__:
        np.int = int

    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    except ImportError:
        GlobalRoutePlannerDAO = None

    import TCP.model as tcp_model_module
    from TCP.config import GlobalConfig
    from TCP.model import TCP
    from team_code.planner import RoutePlanner

    return RuntimeModules(
        carla=carla,
        torch=torch,
        np=np,
        cv2=cv2,
        T=T,
        BasicAgent=BasicAgent,
        GlobalRoutePlanner=GlobalRoutePlanner,
        GlobalRoutePlannerDAO=GlobalRoutePlannerDAO,
        TCP=TCP,
        GlobalConfig=GlobalConfig,
        RoutePlanner=RoutePlanner,
        tcp_model_module=tcp_model_module,
    )


if __name__ == "__main__":
    raise SystemExit(main())
