"""Record an AIM closed-loop CARLA drive as SD2 JSONL.

The pure SD2 conversion lives in ``sd2.adapters.aim_adapter``. This script is
the only place that imports CARLA, torch, and AIM code. It mirrors
``leaderboard/team_code/aim_agent.py`` for sensor style, preprocessing,
RoutePlanner target-point logic, waypoint prediction, and PID control.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd2.adapters.aim_adapter import (
    AIM_MODEL_ID,
    Sd2JsonlWriter,
    aim_record_to_sd2,
    build_aim_run_metadata,
)
from sd2.stressors import ImageStressor

try:
    import _carla_e2e_common as e2e
except ImportError:
    from experiments import _carla_e2e_common as e2e


MODEL_SRC = e2e.REPO_ROOT / "models" / "AIM" / "src"
DEFAULT_CHECKPOINT = Path("models/AIM/aim/best_model.pth")
LOGGER = logging.getLogger("aim_record")


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
    AIM: type
    GlobalConfig: type
    RoutePlanner: type
    scale_and_crop_image: Any


class AIMRuntime:
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
            LOGGER.warning("CUDA is not available; running AIM on %s", self.device)

        self.config = modules.GlobalConfig(
            seq_len=1,
            ignore_sides=True,
            ignore_rear=True,
            input_resolution=256,
            scale=1,
        )
        self.net = modules.AIM(self.config, self.device_name)
        state_dict = self.torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        load_result = self.net.load_state_dict(state_dict)
        self.net.to(self.device).eval()
        param_count = sum(param.numel() for param in self.net.parameters())
        LOGGER.info(
            "Loaded AIM checkpoint %s | params=%.1fM missing=%d unexpected=%d",
            args.checkpoint,
            param_count / 1_000_000,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )

        self.route_planner = modules.RoutePlanner(4.0, 50.0)
        self.intervention = e2e.InterventionPolicy.from_args(args, AIM_MODEL_ID)
        self.input_buffer: dict[str, dict[str, deque[Any]]] = {
            "clean_forward": {"rgb": deque()},
            "stress_forward": {"rgb": deque()},
        }
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
            raise RuntimeError("AIM requires at least two route points")
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
        rgb_tensor = self._rgb_tensor(tick_stress["rgb"])
        input_shapes = {
            "rgb": e2e.shape_summary(rgb_tensor),
            "target_point": e2e.shape_summary(tick_stress["target_point"]),
            "gt_velocity": e2e.shape_summary(tick_stress["speed"]),
        }
        if not self.logged_shapes:
            LOGGER.info("First AIM sensor packet shapes: %s", e2e.sensor_shape_summary(sensor_packet))
            LOGGER.info("First AIM model input shapes: %s", input_shapes)

        stress_forward = self._forward_once(tick_stress, "stress_forward", input_shapes)
        clean_forward = self._forward_once(tick_clean, "clean_forward", input_shapes)
        if stress_forward.get("warmup") or clean_forward.get("warmup"):
            control = self.modules.carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control, self._warmup_record(tick_stress, control, frame_id)

        control_hybrid_planning_clean = (
            clean_forward["control"] if self.intervention.stage == "none" else None
        )
        applied_forward = clean_forward if self.intervention.applied_source == "clean_forward" else stress_forward
        applied_control, applied_metadata = self._control_from_waypoints(
            applied_forward["pred_wp"],
            applied_forward["gt_velocity"],
            preserve_state=False,
        )
        applied_forward = {
            **applied_forward,
            "control": applied_control,
            "metadata": applied_metadata,
        }
        control = e2e.vehicle_control_from_dict(
            self.modules.carla,
            applied_forward["control"],
        )

        if not self.logged_shapes:
            LOGGER.info(
                "First AIM model output shapes: %s",
                {
                    "encoding": e2e.shape_summary(stress_forward["encoding"]),
                    "pred_wp": e2e.shape_summary(stress_forward["pred_wp"]),
                    "feature_source": "mean_pooled_image_encoder",
                },
            )
            self.logged_shapes = True

        pred_wp_np = stress_forward["pred_wp_np"]
        feature_np = stress_forward["feature_np"]
        target_speed = e2e.optional_float(stress_forward["metadata"].get("desired_speed"))
        if target_speed is None:
            target_speed = e2e.target_speed_from_waypoints(self.np, pred_wp_np)

        extracted = {
            "vision": {
                "image_mean": tick_stress["image_mean"],
                "image_std": tick_stress["image_std"],
                "feature": e2e.pool_feature(self.np, feature_np),
                "feature_source": "mean_pooled_image_encoder",
            },
            "planning": {
                "waypoints": pred_wp_np.astype(float).tolist(),
                "target_speed": target_speed,
                "target_point": tick_stress["target_point"].astype(float).tolist(),
                "command": int(tick_stress["next_command"]),
                "planning_source": "predicted_waypoints",
            },
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "aim": {
                "carla_frame": int(frame_id),
                "pid_metadata": e2e.jsonable(applied_forward["metadata"]),
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

    def _tick(self, input_data: dict[str, tuple[int, Any]]) -> dict[str, Any]:
        rgb = self.cv2.cvtColor(input_data["rgb"][1][:, :, :3], self.cv2.COLOR_BGR2RGB)
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

    def _with_stressed_image(self, tick_data: dict[str, Any]) -> dict[str, Any]:
        stressed = dict(tick_data)
        rgb = e2e.apply_visual_stress(
            tick_data["rgb"],
            self.stressor,
            self.args.stress_severity,
            self.stress_rng,
        )
        stressed["rgb"] = rgb
        stressed["image_mean"] = float(rgb.astype(self.np.float32).mean() / 255.0)
        stressed["image_std"] = float(rgb.astype(self.np.float32).std() / 255.0)
        return stressed

    def _forward_once(
        self,
        tick_data: dict[str, Any],
        source: str,
        input_shapes: dict[str, Any],
    ) -> dict[str, Any]:
        rgb_tensor = self._rgb_tensor(tick_data["rgb"])
        buffer = self.input_buffer[source]["rgb"]
        if len(buffer) >= self.config.seq_len:
            buffer.popleft()
        buffer.append(rgb_tensor)

        if len(buffer) < self.config.seq_len or self.step < self.config.seq_len:
            return {"warmup": True}

        target_point = self.torch.tensor(
            [[tick_data["target_point"][0], tick_data["target_point"][1]]],
            device=self.device,
            dtype=self.torch.float32,
        )
        gt_velocity = self.torch.tensor(
            [tick_data["speed"]],
            device=self.device,
            dtype=self.torch.float32,
        )

        try:
            with self.torch.no_grad():
                encoding = [self.net.image_encoder(list(buffer))]
                pred_wp = self.net(encoding, target_point)
        except Exception:
            LOGGER.exception("AIM forward failed; model input shapes were: %s", input_shapes)
            raise

        pred_wp_np = pred_wp.detach().cpu().numpy()[0].copy()
        feature_np = encoding[0].detach().cpu().numpy()
        control, metadata = self._control_from_waypoints(
            pred_wp,
            gt_velocity,
            preserve_state=True,
        )
        return {
            "warmup": False,
            "encoding": encoding,
            "pred_wp": pred_wp,
            "pred_wp_np": pred_wp_np,
            "feature_np": feature_np,
            "gt_velocity": gt_velocity,
            "control": control,
            "metadata": metadata,
        }

    def _control_from_waypoints(
        self,
        pred_wp: Any,
        gt_velocity: Any,
        *,
        preserve_state: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        pid_state = e2e.snapshot_pid_controllers(self.net) if preserve_state else {}
        steer, throttle, brake, metadata = self.net.control_pid(pred_wp, gt_velocity)
        if preserve_state:
            e2e.restore_pid_controllers(self.net, pid_state)
        brake_value = e2e.scalar_float(brake)
        throttle_value = e2e.scalar_float(throttle)
        if brake_value < 0.05:
            brake_value = 0.0
        if throttle_value > brake_value:
            brake_value = 0.0
        return {
            "steer": e2e.scalar_float(steer),
            "throttle": throttle_value,
            "brake": brake_value,
        }, metadata

    def _rgb_tensor(self, rgb: Any) -> Any:
        return self.torch.from_numpy(
            self.modules.scale_and_crop_image(
                self.Image.fromarray(rgb),
                scale=self.config.scale,
                crop=self.config.input_resolution,
            )
        ).unsqueeze(0).to(self.device, dtype=self.torch.float32)

    def _warmup_record(self, tick_data: dict[str, Any], control: Any, frame_id: int) -> dict[str, Any]:
        neutral = e2e.control_to_dict(control)
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
            "aim": {"carla_frame": int(frame_id), "warmup": True},
            "intervention": e2e.build_intervention_block(
                self.intervention,
                control_from_stress_forward=neutral,
                control_from_clean_forward=neutral,
                planning_waypoints_clean_forward=None,
            ),
        }


def main(argv: list[str] | None = None) -> int:
    args = e2e.parse_record_args(
        argv,
        description="Record an AIM synchronous CARLA run as SD2 JSONL.",
        default_checkpoint=DEFAULT_CHECKPOINT,
        model_id=AIM_MODEL_ID,
    )
    e2e.configure_logging()
    modules = _import_runtime_modules()
    return e2e.run_recording(
        args,
        modules,
        AIMRuntime,
        model_id=AIM_MODEL_ID,
        model_label="AIM",
        record_to_sd2=aim_record_to_sd2,
        build_run_metadata=build_aim_run_metadata,
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

    from aim.config import GlobalConfig
    from aim.data import scale_and_crop_image
    from aim.model import AIM
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
        AIM=AIM,
        GlobalConfig=GlobalConfig,
        RoutePlanner=RoutePlanner,
        scale_and_crop_image=scale_and_crop_image,
    )


if __name__ == "__main__":
    raise SystemExit(main())
