"""Record a CILRS closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.cilrs_adapter``. This script is
the only place that imports CARLA, torch, and CILRS code. It mirrors
``leaderboard/team_code/cilrs_agent.py`` for sensor style, preprocessing,
RoutePlanner command logic, direct control regression, and brake gating.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd2.adapters.cilrs_adapter import (
    CILRS_MODEL_ID,
    Sd2JsonlWriter,
    build_cilrs_run_metadata,
    cilrs_record_to_sd2,
)
from sd2.stressors import ImageStressor

try:
    import _carla_e2e_common as e2e
except ImportError:
    from experiments import _carla_e2e_common as e2e


MODEL_SRC = e2e.REPO_ROOT / "models" / "CILRS" / "src"
DEFAULT_CHECKPOINT = Path("models/CILRS/cilrs/best_model.pth")
LOGGER = logging.getLogger("cilrs_record")


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
    CILRS: type
    GlobalConfig: type
    RoutePlanner: type
    scale_and_crop_image: Any


class CILRSRuntime:
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
            LOGGER.warning("CUDA is not available; running CILRS on %s", self.device)

        self.config = modules.GlobalConfig(
            seq_len=1,
            ignore_sides=True,
            ignore_rear=True,
        )
        self.net = modules.CILRS(self.config, self.device_name)
        state_dict = self.torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        load_result = self.net.load_state_dict(state_dict)
        self.net.to(self.device).eval()
        param_count = sum(param.numel() for param in self.net.parameters())
        LOGGER.info(
            "Loaded CILRS checkpoint %s | params=%.1fM missing=%d unexpected=%d",
            args.checkpoint,
            param_count / 1_000_000,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )

        self.route_planner = modules.RoutePlanner(4.0, 50.0)
        self.input_buffer: dict[str, deque[Any]] = {"rgb": deque()}
        self.step = -1
        self.logged_shapes = False

    def sensor_specs(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb",
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
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        )

    def set_global_plan(self, gps_plan: list[tuple[dict[str, float], Any]]) -> None:
        if len(gps_plan) < 2:
            raise RuntimeError("CILRS requires at least two route points")
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
        rgb_tensor = self._rgb_tensor(tick_data["rgb"])
        input_shapes = {
            "rgb": e2e.shape_summary(rgb_tensor),
            "gt_velocity": e2e.shape_summary(tick_data["speed"]),
            "command": e2e.shape_summary(tick_data["next_command"]),
        }
        if not self.logged_shapes:
            LOGGER.info("First CILRS sensor packet shapes: %s", e2e.sensor_shape_summary(sensor_packet))
            LOGGER.info("First CILRS model input shapes: %s", input_shapes)

        if len(self.input_buffer["rgb"]) >= self.config.seq_len:
            self.input_buffer["rgb"].popleft()
        self.input_buffer["rgb"].append(rgb_tensor)

        if len(self.input_buffer["rgb"]) < self.config.seq_len or self.step < self.config.seq_len:
            control = self.modules.carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control, self._warmup_record(tick_data, control, frame_id)

        gt_velocity = self.torch.tensor(
            [tick_data["speed"]],
            device=self.device,
            dtype=self.torch.float32,
        )
        command = self.torch.tensor(
            [tick_data["next_command"]],
            device=self.device,
            dtype=self.torch.float32,
        )

        try:
            with self.torch.no_grad():
                encoding = [self.net.encoder(list(self.input_buffer["rgb"]))]
                steer, throttle, brake, velocity_pred = self.net(
                    encoding,
                    gt_velocity,
                    command,
                )
        except Exception:
            LOGGER.exception("CILRS forward failed; model input shapes were: %s", input_shapes)
            raise

        steer_value = e2e.scalar_float(steer.squeeze(0))
        throttle_value = e2e.scalar_float(throttle.squeeze(0))
        brake_value = e2e.scalar_float(brake.squeeze(0))
        if brake_value < 0.05:
            brake_value = 0.0
        if throttle_value > brake_value:
            brake_value = 0.0

        control = self.modules.carla.VehicleControl()
        control.steer = steer_value
        control.throttle = throttle_value
        control.brake = brake_value

        feature_np = encoding[0].detach().cpu().numpy()
        velocity_pred_value = e2e.scalar_float(velocity_pred)
        if not self.logged_shapes:
            LOGGER.info(
                "First CILRS model output shapes: %s",
                {
                    "encoding": e2e.shape_summary(encoding),
                    "steer": e2e.shape_summary(steer),
                    "throttle": e2e.shape_summary(throttle),
                    "brake": e2e.shape_summary(brake),
                    "velocity_pred": e2e.shape_summary(velocity_pred),
                    "feature_source": "mean_pooled_encoder",
                },
            )
            self.logged_shapes = True

        extracted = {
            "vision": {
                "image_mean": tick_data["image_mean"],
                "image_std": tick_data["image_std"],
                "feature": e2e.pool_feature(self.np, feature_np),
                "feature_source": "mean_pooled_encoder",
            },
            "planning": {
                "target_speed": velocity_pred_value,
                "velocity_pred": velocity_pred_value,
                "target_point": tick_data["target_point"].astype(float).tolist(),
                "command": int(tick_data["next_command"]),
                "planning_source": "predicted_velocity",
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "cilrs": {"carla_frame": int(frame_id)},
        }
        return control, extracted

    def _tick(self, input_data: dict[str, tuple[int, Any]]) -> dict[str, Any]:
        rgb = self.cv2.cvtColor(input_data["rgb"][1][:, :, :3], self.cv2.COLOR_BGR2RGB)
        rgb = e2e.apply_visual_stress(
            rgb,
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
        return {
            "rgb": rgb,
            "gps": pos,
            "speed": speed,
            "compass": compass,
            "next_command": int(getattr(next_cmd, "value", next_cmd)),
            "target_point": target_point,
            "image_mean": float(rgb.astype(self.np.float32).mean() / 255.0),
            "image_std": float(rgb.astype(self.np.float32).std() / 255.0),
        }

    def _rgb_tensor(self, rgb: Any) -> Any:
        return self.torch.from_numpy(
            self.modules.scale_and_crop_image(self.Image.fromarray(rgb))
        ).unsqueeze(0).to(self.device, dtype=self.torch.float32)

    def _warmup_record(self, tick_data: dict[str, Any], control: Any, frame_id: int) -> dict[str, Any]:
        return {
            "vision": {
                "image_mean": tick_data["image_mean"],
                "image_std": tick_data["image_std"],
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
            "cilrs": {"carla_frame": int(frame_id), "warmup": True},
        }


def main(argv: list[str] | None = None) -> int:
    args = e2e.parse_record_args(
        argv,
        description="Record a CILRS synchronous CARLA run as SD2 JSONL.",
        default_checkpoint=DEFAULT_CHECKPOINT,
        model_id=CILRS_MODEL_ID,
    )
    e2e.configure_logging()
    modules = _import_runtime_modules()
    return e2e.run_recording(
        args,
        modules,
        CILRSRuntime,
        model_id=CILRS_MODEL_ID,
        model_label="CILRS",
        record_to_sd2=cilrs_record_to_sd2,
        build_run_metadata=build_cilrs_run_metadata,
        jsonl_writer_cls=Sd2JsonlWriter,
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

    from cilrs.config import GlobalConfig
    from cilrs.data import scale_and_crop_image
    from cilrs.model import CILRS
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
        CILRS=CILRS,
        GlobalConfig=GlobalConfig,
        RoutePlanner=RoutePlanner,
        scale_and_crop_image=scale_and_crop_image,
    )


if __name__ == "__main__":
    raise SystemExit(main())
