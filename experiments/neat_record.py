"""Record a NEAT closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.neat_adapter``. This script is
the only place that imports CARLA, torch, and NEAT code. It mirrors
``leaderboard/team_code/neat_agent.py`` for sensor style, preprocessing,
RoutePlanner target-point logic, attention-field planning, PID control, and BEV
semantic occupancy decoding.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd2.adapters.neat_adapter import (
    NEAT_MODEL_ID,
    build_neat_run_metadata,
    neat_record_to_sd2,
    write_sd2_jsonl,
)
from sd2.stressors import ImageStressor

try:
    import _carla_e2e_common as e2e
except ImportError:
    from experiments import _carla_e2e_common as e2e


MODEL_SRC = e2e.REPO_ROOT / "models" / "NEAT" / "src"
DEFAULT_CHECKPOINT = Path("models/NEAT/neat")
LOGGER = logging.getLogger("neat_record")


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
    AttentionField: type
    GlobalConfig: type
    RoutePlanner: type


class NEATRuntime:
    def __init__(
        self,
        args: Any,
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
        self.device_name = self.device.type
        if self.device.type != "cuda":
            LOGGER.warning("CUDA is not available; running NEAT on %s", self.device)

        self.model_args = _load_neat_args(args.checkpoint)
        self.model_args["out_res"] = 100
        self.config = modules.GlobalConfig(num_camera=3, seq_len=1)
        self.net = modules.AttentionField(self.config, self.device_name)
        encoder_path = args.checkpoint / "best_encoder.pth"
        decoder_path = args.checkpoint / "best_decoder.pth"
        encoder_state = self.torch.load(encoder_path, map_location=self.device, weights_only=False)
        decoder_state = self.torch.load(decoder_path, map_location=self.device, weights_only=False)
        encoder_result = self.net.encoder.load_state_dict(encoder_state)
        decoder_result = self.net.decoder.load_state_dict(decoder_state)
        self.plan_grid = self.net.create_plan_grid(
            self.config.plan_scale,
            self.config.plan_points,
            1,
        )
        self.light_grid = self.net.create_light_grid(
            self.config.light_x_steps,
            self.config.light_y_steps,
            1,
        )
        self.net.to(self.device).eval()
        param_count = sum(param.numel() for param in self.net.parameters())
        LOGGER.info(
            "Loaded NEAT checkpoint %s | params=%.1fM encoder_missing=%d "
            "encoder_unexpected=%d decoder_missing=%d decoder_unexpected=%d",
            args.checkpoint,
            param_count / 1_000_000,
            len(getattr(encoder_result, "missing_keys", [])),
            len(getattr(encoder_result, "unexpected_keys", [])),
            len(getattr(decoder_result, "missing_keys", [])),
            len(getattr(decoder_result, "unexpected_keys", [])),
        )

        self.route_planner = modules.RoutePlanner(4.0, 50.0)
        self.input_buffer: dict[str, deque[Any]] = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
        }
        self.step = -1
        self.logged_shapes = False
        self.logged_bev_failure = False

    def sensor_specs(self) -> tuple[dict[str, Any], ...]:
        camera_base = {
            "type": "sensor.camera.rgb",
            "x": 1.3,
            "z": 2.3,
            "roll": 0.0,
            "pitch": 0.0,
            "width": 400,
            "height": 300,
            "fov": 100,
        }
        return (
            {**camera_base, "y": 0.0, "yaw": 0.0, "id": "rgb"},
            {**camera_base, "y": 0.0, "yaw": -60.0, "id": "rgb_left"},
            {**camera_base, "y": 0.0, "yaw": 60.0, "id": "rgb_right"},
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
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
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        )

    def set_global_plan(self, gps_plan: list[tuple[dict[str, float], Any]]) -> None:
        if len(gps_plan) < 2:
            raise RuntimeError("NEAT requires at least two route points")
        self.route_planner.set_route(gps_plan, True)

    def run_step(
        self,
        sensor_packet: dict[str, tuple[int, Any]],
        timestamp: float,
        frame_id: int,
    ) -> tuple[Any, dict[str, Any]]:
        del timestamp
        self.step += 1
        tick_data = self._tick(sensor_packet)
        tensors = {
            "rgb": self._rgb_tensor(tick_data["rgb"]),
            "rgb_left": self._rgb_tensor(tick_data["rgb_left"]),
            "rgb_right": self._rgb_tensor(tick_data["rgb_right"]),
        }
        input_shapes = {
            "images": e2e.shape_summary(list(tensors.values())),
            "target_point": e2e.shape_summary(tick_data["target_point"]),
            "gt_velocity": e2e.shape_summary(tick_data["speed"]),
        }
        if not self.logged_shapes:
            LOGGER.info("First NEAT sensor packet shapes: %s", e2e.sensor_shape_summary(sensor_packet))
            LOGGER.info("First NEAT model input shapes: %s", input_shapes)

        for key, tensor in tensors.items():
            if len(self.input_buffer[key]) >= self.config.seq_len:
                self.input_buffer[key].popleft()
            self.input_buffer[key].append(tensor)

        if (
            any(len(buffer) < self.config.seq_len for buffer in self.input_buffer.values())
            or self.step < self.config.seq_len
        ):
            control = self.modules.carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control, self._warmup_record(tick_data, control, frame_id)

        target_point = self.torch.tensor(
            [[tick_data["target_point"][0]], [tick_data["target_point"][1]]],
            device=self.device,
            dtype=self.torch.float32,
        )
        gt_velocity = self.torch.tensor(
            [tick_data["speed"]],
            device=self.device,
            dtype=self.torch.float32,
        )
        images = []
        for index in range(self.config.seq_len):
            images.append(self.input_buffer["rgb"][index])
            if self.config.num_camera == 3:
                images.append(self.input_buffer["rgb_left"][index])
                images.append(self.input_buffer["rgb_right"][index])

        try:
            with self.torch.no_grad():
                encoding = self.net.encoder(images, gt_velocity)
                pred_waypoint_mean, red_light_occ = self.net.plan(
                    target_point,
                    encoding,
                    self.plan_grid,
                    self.light_grid,
                    self.config.plan_points,
                    self.config.plan_iters,
                )
        except Exception:
            LOGGER.exception("NEAT forward failed; model input shapes were: %s", input_shapes)
            raise

        future_waypoints = pred_waypoint_mean[:, self.config.seq_len :]
        future_waypoints_np = future_waypoints.detach().cpu().numpy()[0].copy()
        steer, throttle, brake, metadata = self.net.control_pid(
            future_waypoints,
            gt_velocity,
            target_point,
            red_light_occ,
        )
        steer_value = e2e.scalar_float(steer)
        throttle_value = e2e.scalar_float(throttle)
        brake_value = e2e.scalar_float(brake)
        if brake_value < 0.05:
            brake_value = 0.0
        if throttle_value > brake_value:
            brake_value = 0.0

        control = self.modules.carla.VehicleControl()
        control.steer = steer_value
        control.throttle = throttle_value
        control.brake = brake_value

        encoding_np = encoding.detach().cpu().numpy()
        bev_seg_summary = self._try_bev_seg_summary(encoding, target_point)
        red_light_occ_value = e2e.scalar_float(red_light_occ)
        if not self.logged_shapes:
            LOGGER.info(
                "First NEAT model output shapes: %s",
                {
                    "encoding": e2e.shape_summary(encoding),
                    "pred_waypoint_mean": e2e.shape_summary(pred_waypoint_mean),
                    "red_light_occ": e2e.shape_summary(red_light_occ),
                    "bev_seg_summary": e2e.shape_summary(bev_seg_summary),
                    "feature_source": "mean_pooled_encoder_tokens",
                },
            )
            self.logged_shapes = True

        target_speed = e2e.optional_float(metadata.get("desired_speed"))
        if target_speed is None:
            target_speed = e2e.target_speed_from_waypoints(self.np, future_waypoints_np)

        extracted = {
            "vision": {
                "image_mean": tick_data["image_mean"],
                "image_std": tick_data["image_std"],
                "feature": e2e.pool_feature(self.np, encoding_np),
                "feature_source": "mean_pooled_encoder_tokens",
            },
            "semantic": {
                "bev_seg_summary": bev_seg_summary,
                "objects": None,
                "red_light_occ": red_light_occ_value,
                "semantic_source": "bev_occupancy_decode",
            },
            "planning": {
                "waypoints": future_waypoints_np.astype(float).tolist(),
                "target_speed": target_speed,
                "target_point": tick_data["target_point"].astype(float).tolist(),
                "command": int(tick_data["next_command"]),
                "planning_source": "attention_field_waypoints",
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "neat": {
                "carla_frame": int(frame_id),
                "pid_metadata": e2e.jsonable(metadata),
            },
        }
        return control, extracted

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
        rgb = e2e.apply_visual_stress(rgb, self.stressor, self.args.stress_severity, self.stress_rng)
        rgb_left = e2e.apply_visual_stress(
            rgb_left,
            self.stressor,
            self.args.stress_severity,
            self.stress_rng,
        )
        rgb_right = e2e.apply_visual_stress(
            rgb_right,
            self.stressor,
            self.args.stress_severity,
            self.stress_rng,
        )
        gps = input_data["gps"][1][:2]
        speed = float(input_data["speed"][1]["speed"])
        compass = float(input_data["imu"][1][-1])
        if math.isnan(compass):
            compass = 0.0

        pos = e2e.route_position(self.np, self.route_planner, gps)
        next_wp, next_cmd = self.route_planner.run_step(pos)
        target_point = e2e.route_target_point(self.np, pos, compass, next_wp)
        image_stack = self.np.concatenate(
            [
                rgb.astype(self.np.float32).reshape(-1, 3),
                rgb_left.astype(self.np.float32).reshape(-1, 3),
                rgb_right.astype(self.np.float32).reshape(-1, 3),
            ],
            axis=0,
        )
        return {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": pos,
            "speed": speed,
            "compass": compass,
            "next_command": int(getattr(next_cmd, "value", next_cmd)),
            "target_point": target_point,
            "image_mean": float(image_stack.mean() / 255.0),
            "image_std": float(image_stack.std() / 255.0),
        }

    def _rgb_tensor(self, rgb: Any) -> Any:
        return self.torch.from_numpy(
            _scale_and_crop_image(
                self.modules,
                self.Image.fromarray(rgb),
                scale=self.config.scale,
                crop=self.config.crop,
            )
        ).unsqueeze(0).to(self.device, dtype=self.torch.float32)

    def _try_bev_seg_summary(self, encoding: Any, target_point: Any) -> dict[str, Any] | None:
        try:
            out_res = min(int(self.model_args.get("out_res", 100)), 100)
            linspace_x = self.torch.linspace(
                -self.config.axis / 2,
                self.config.axis / 2,
                steps=out_res,
                device=self.device,
            )
            linspace_y = self.torch.linspace(
                -self.config.axis / 2,
                self.config.axis / 2,
                steps=out_res,
                device=self.device,
            )
            grid_x, grid_y = self.torch.meshgrid(linspace_x, linspace_y)
            grid_t = self.torch.zeros_like(grid_x)
            grid_points = self.torch.stack((grid_x, grid_y, grid_t), dim=2)
            grid_points = grid_points.reshape(1, -1, 3).to(self.device, dtype=self.torch.float32)
            with self.torch.no_grad():
                pred_img_pts, _pred_img_offsets, _attn = self.net.decode(
                    grid_points,
                    target_point,
                    encoding,
                )
                prediction = self.torch.argmax(pred_img_pts[-1], dim=1)
            prediction_np = prediction.detach().cpu().numpy()[0]
            counts = self.np.bincount(prediction_np.reshape(-1), minlength=self.config.num_class)
            total = int(prediction_np.size)
            summary = {
                f"class_{idx}": int(counts[idx])
                for idx in range(int(self.config.num_class))
            }
            summary["nonzero_fraction"] = float((total - int(counts[0])) / max(1, total))
            summary["dominant_class"] = int(self.np.argmax(counts))
            return summary
        except Exception as exc:
            if not self.logged_bev_failure:
                LOGGER.warning("NEAT BEV summary unavailable: %s", exc)
                self.logged_bev_failure = True
            return None

    def _warmup_record(self, tick_data: dict[str, Any], control: Any, frame_id: int) -> dict[str, Any]:
        return {
            "vision": {
                "image_mean": tick_data["image_mean"],
                "image_std": tick_data["image_std"],
            },
            "semantic": {
                "bev_seg_summary": None,
                "objects": None,
                "semantic_source": "warmup_buffer",
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
            },
            "neat": {"carla_frame": int(frame_id), "warmup": True},
        }


def main(argv: list[str] | None = None) -> int:
    args = e2e.parse_record_args(
        argv,
        description="Record a NEAT synchronous CARLA run as SD2 JSONL.",
        default_checkpoint=DEFAULT_CHECKPOINT,
    )
    e2e.configure_logging()
    modules = _import_runtime_modules()
    return e2e.run_recording(
        args,
        modules,
        NEATRuntime,
        model_id=NEAT_MODEL_ID,
        model_label="NEAT",
        record_to_sd2=neat_record_to_sd2,
        build_run_metadata=build_neat_run_metadata,
        write_sd2_jsonl=write_sd2_jsonl,
        logger=LOGGER,
    )


def _apply_model_preamble() -> None:
    for path in (
        e2e.CARLA_PYTHON_API,
        MODEL_SRC,
        MODEL_SRC / "leaderboard",
        MODEL_SRC / "scenario_runner",
    ):
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
    from PIL import Image

    if "int" not in np.__dict__:
        np.int = int

    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    try:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    except ImportError:
        GlobalRoutePlannerDAO = None

    from neat.architectures import AttentionField
    from neat.config import GlobalConfig
    from team_code.planner import RoutePlanner

    return RuntimeModules(
        carla=carla,
        torch=torch,
        np=np,
        cv2=cv2,
        Image=Image,
        BasicAgent=BasicAgent,
        GlobalRoutePlanner=GlobalRoutePlanner,
        GlobalRoutePlannerDAO=GlobalRoutePlannerDAO,
        AttentionField=AttentionField,
        GlobalConfig=GlobalConfig,
        RoutePlanner=RoutePlanner,
    )


def _load_neat_args(checkpoint_dir: Path) -> dict[str, Any]:
    args_path = checkpoint_dir / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"NEAT checkpoint args file not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scale_and_crop_image(modules: RuntimeModules, image: Any, scale: int = 1, crop: int = 256) -> Any:
    width, height = image.width // scale, image.height // scale
    resized = image.resize((width, height))
    image_array = modules.np.asarray(resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image_array[start_x : start_x + crop, start_y : start_y + crop]
    return modules.np.transpose(cropped_image, (2, 0, 1))


if __name__ == "__main__":
    raise SystemExit(main())
