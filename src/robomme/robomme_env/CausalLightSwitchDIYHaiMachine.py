"""Two-button causal light-switch scene for demos and dataset collection."""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from .utils import reset_panda
from .utils.object_generation import build_button


@register_env("CausalLightSwitchDIYHaiMachine")
class CausalLightSwitchDIYHaiMachine(BaseEnv):
    """A red and a blue physical button with one hidden causal controller."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]

    BUTTON_POSITIONS = {
        "red": (-0.12, -0.09),
        "blue": (-0.12, 0.09),
    }
    LAMP_POSITION = np.asarray([0.14, 0.0, 0.135], dtype=np.float32)
    HIDDEN_POSITION = np.asarray([0.0, 0.0, -2.0], dtype=np.float32)
    PRESS_THRESHOLD_M = 0.005
    RELEASE_THRESHOLD_M = 0.002

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0,
        control_button_color="blue",
        control_button_colors=None,
        initial_lamp_on=False,
        red_button_xy=None,
        blue_button_xy=None,
        **kwargs,
    ):
        if control_button_colors is None:
            controls = {str(control_button_color).lower()}
        else:
            if isinstance(control_button_colors, str):
                control_button_colors = [control_button_colors]
            controls = {str(color).lower() for color in control_button_colors}
        invalid_controls = controls.difference(self.BUTTON_POSITIONS)
        if invalid_controls:
            raise ValueError(
                f"control button colors must be red/blue, got {sorted(invalid_controls)}"
            )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.control_button_colors = frozenset(controls)
        self.control_button_color = (
            next(iter(controls)) if len(controls) == 1 else None
        )
        self.initial_lamp_on = bool(initial_lamp_on)
        self.button_positions = {
            "red": np.asarray(
                self.BUTTON_POSITIONS["red"] if red_button_xy is None else red_button_xy,
                dtype=np.float32,
            ),
            "blue": np.asarray(
                self.BUTTON_POSITIONS["blue"] if blue_button_xy is None else blue_button_xy,
                dtype=np.float32,
            ),
        }
        self.lamp_on = self.initial_lamp_on
        self.button_press_counts = {"red": 0, "blue": 0}
        self.press_history = []
        self._button_latched = {"red": False, "blue": False}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.38, -0.02, 0.44],
            target=[-0.02, 0.0, 0.035],
        )
        return [CameraConfig("base_camera", pose, 384, 384, 1.0, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.42, -0.02, 0.46],
            target=[-0.02, 0.0, 0.04],
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    @staticmethod
    def _material(color):
        material = sapien.render.RenderMaterial()
        material.set_base_color(list(color))
        return material

    def _build_visual_box(self, name, position, half_size, color):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=list(half_size), material=self._material(color)
        )
        builder.set_initial_pose(sapien.Pose(p=list(position)))
        return builder.build_kinematic(name=name)

    def _build_visual_sphere(self, name, position, radius, color):
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(radius=float(radius), material=self._material(color))
        builder.set_initial_pose(sapien.Pose(p=list(position)))
        return builder.build_kinematic(name=name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        build_button(
            self,
            center_xy=self.button_positions["red"],
            scale=1.25,
            travel=0.012,
            name="causal_red_button",
            randomize=False,
            cap_color=[0.9, 0.03, 0.03, 1.0],
        )
        self.button_red = self.button
        self.button_joint_red = self.button_joint

        build_button(
            self,
            center_xy=self.button_positions["blue"],
            scale=1.25,
            travel=0.012,
            name="causal_blue_button",
            randomize=False,
            cap_color=[0.03, 0.12, 0.95, 1.0],
        )
        self.button_blue = self.button
        self.button_joint_blue = self.button_joint
        self.buttons = {"red": self.button_red, "blue": self.button_blue}
        self._button_root_z = {
            color: self._pose_vector(button.pose.p)[2]
            for color, button in self.buttons.items()
        }
        self._button_root_q = {
            color: self._pose_vector(button.pose.q)[:4]
            for color, button in self.buttons.items()
        }

        self.lamp_base = self._build_visual_box(
            "causal_lamp_base",
            [0.14, 0.0, 0.018],
            [0.05, 0.05, 0.018],
            [0.12, 0.12, 0.12, 1.0],
        )
        self.lamp_post = self._build_visual_box(
            "causal_lamp_post",
            [0.14, 0.0, 0.075],
            [0.012, 0.012, 0.06],
            [0.18, 0.18, 0.18, 1.0],
        )
        self.lamp_off_actor = self._build_visual_sphere(
            "causal_lamp_off",
            self.LAMP_POSITION,
            0.037,
            [0.045, 0.045, 0.045, 1.0],
        )
        self.lamp_on_actor = self._build_visual_sphere(
            "causal_lamp_on",
            self.HIDDEN_POSITION,
            0.037,
            [1.0, 0.88, 0.04, 1.0],
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.agent.reset(reset_panda.get_reset_panda_param("qpos"))
            for button in self.buttons.values():
                button.set_qpos(torch.zeros_like(button.get_qpos()))
                button.set_qvel(torch.zeros_like(button.get_qvel()))
        self._apply_button_positions()
        self.lamp_on = self.initial_lamp_on
        self.button_press_counts = {"red": 0, "blue": 0}
        self.press_history = []
        self._button_latched = {"red": False, "blue": False}
        self._apply_lamp_visual()

    @staticmethod
    def _pose_vector(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def configure_episode(
        self,
        *,
        red_button_xy,
        blue_button_xy,
        control_button_color: str | None = None,
        control_button_colors=None,
        initial_lamp_on: bool = False,
    ) -> None:
        """Configure the hidden cause and button layout used by the next reset."""
        if control_button_colors is None:
            if control_button_color is None:
                raise ValueError("a control button color or color set is required")
            controls = {str(control_button_color).lower()}
        else:
            if isinstance(control_button_colors, str):
                control_button_colors = [control_button_colors]
            controls = {str(color).lower() for color in control_button_colors}
        invalid_controls = controls.difference(self.BUTTON_POSITIONS)
        if invalid_controls:
            raise ValueError(
                f"control button colors must be red/blue, got {sorted(invalid_controls)}"
            )
        positions = {
            "red": np.asarray(red_button_xy, dtype=np.float32).reshape(2),
            "blue": np.asarray(blue_button_xy, dtype=np.float32).reshape(2),
        }
        if not all(np.isfinite(position).all() for position in positions.values()):
            raise ValueError("button positions must be finite XY coordinates")
        self.control_button_colors = frozenset(controls)
        self.control_button_color = (
            next(iter(controls)) if len(controls) == 1 else None
        )
        self.initial_lamp_on = bool(initial_lamp_on)
        self.button_positions = positions

    def _apply_button_positions(self) -> None:
        for color, button in self.buttons.items():
            xy = self.button_positions[color]
            button.set_pose(
                sapien.Pose(
                    p=[float(xy[0]), float(xy[1]), self._button_root_z[color]],
                    q=self._button_root_q[color].tolist(),
                )
            )

    @staticmethod
    def _scalar(value) -> float:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return float(np.asarray(value).reshape(-1)[0])

    def _button_depth(self, color: str) -> float:
        return max(0.0, -self._scalar(self.buttons[color].get_qpos()))

    def _apply_lamp_visual(self):
        visible = sapien.Pose(p=self.LAMP_POSITION)
        hidden = sapien.Pose(p=self.HIDDEN_POSITION)
        self.lamp_on_actor.set_pose(visible if self.lamp_on else hidden)
        self.lamp_off_actor.set_pose(hidden if self.lamp_on else visible)

    def _process_button_events(self):
        for color in ("red", "blue"):
            depth = self._button_depth(color)
            if depth >= self.PRESS_THRESHOLD_M and not self._button_latched[color]:
                self._button_latched[color] = True
                before = bool(self.lamp_on)
                self.button_press_counts[color] += 1
                controls_lamp = color in self.control_button_colors
                if controls_lamp:
                    self.lamp_on = not self.lamp_on
                    self._apply_lamp_visual()
                self.press_history.append(
                    {
                        "step": int(self.elapsed_steps.item()),
                        "button_color": color,
                        "button_depth_m": depth,
                        "controls_lamp": controls_lamp,
                        "lamp_before": before,
                        "lamp_after": bool(self.lamp_on),
                    }
                )
            elif depth <= self.RELEASE_THRESHOLD_M:
                self._button_latched[color] = False

    def _get_obs_extra(self, info: Dict):
        return {
            "lamp_on": torch.tensor([float(self.lamp_on)], device=self.device),
            "red_button_depth": torch.tensor(
                [self._button_depth("red")], device=self.device
            ),
            "blue_button_depth": torch.tensor(
                [self._button_depth("blue")], device=self.device
            ),
        }

    def evaluate(self):
        success = len(self.press_history) >= 4
        return {
            "success": torch.tensor([success], device=self.device),
            "fail": torch.tensor([False], device=self.device),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros((self.num_envs,), device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = BaseEnv.step(self, action)
        self._process_button_events()
        info["causal_light_switch"] = {
            "lamp_on": bool(self.lamp_on),
            "press_counts": dict(self.button_press_counts),
            "event_count": len(self.press_history),
        }
        return obs, reward, terminated, truncated, info
