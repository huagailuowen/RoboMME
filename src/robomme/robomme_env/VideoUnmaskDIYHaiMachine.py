"""Deterministic VideoUnmask variant for local occlusion-memory demos.

This environment keeps the original RoboMME ``VideoUnmask`` task semantics,
but removes random object layout for reproducible DIY data:

- fixed bin/container positions
- fixed cube colors and target order
- smooth top-down covering motion instead of teleporting bins away/back

The original ``VideoUnmask`` environment is intentionally left unchanged.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from .VideoUnmask import VideoUnmask
from .utils import *
from .utils.object_generation import build_bin, spawn_fixed_cube
from .utils.subgoal_evaluate_func import static_check


@register_env("VideoUnmaskDIYHaiMachine")
class VideoUnmaskDIYHaiMachine(VideoUnmask):
    """Fixed-layout unmasking task with smooth cover descent."""

    # Fixed from an oracle-solvable seed-0 layout.  bin_0 is the green target
    # container; the other two bins are deterministic distractors.
    DIY_BIN_POSITIONS = [
        [-0.0013, 0.0925],
        [-0.1269, -0.0664],
        [0.1562, -0.1600],
    ]
    DIY_COLOR_NAMES = ["green", "red", "blue"]
    DIY_CUBE_COLORS = [
        (0, 1, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 1, 1),
    ]

    COVER_START_STEP = 24
    COVER_DURATION_STEPS = 48
    COVER_HEIGHT = 0.18
    COVER_SETTLE_STEPS = 12

    def __init__(self, *args, **kwargs):
        self.diy_bin_positions = kwargs.pop("diy_bin_positions", None) or self.DIY_BIN_POSITIONS
        self.diy_color_names = kwargs.pop("diy_color_names", None) or self.DIY_COLOR_NAMES
        self.diy_cube_colors = kwargs.pop("diy_cube_colors", None) or self.DIY_CUBE_COLORS
        present_cube_indices = kwargs.pop("diy_present_cube_indices", None)
        if present_cube_indices is None:
            present_cube_indices = range(len(self.diy_bin_positions))
        self.diy_present_cube_indices = set(int(i) for i in present_cube_indices)
        self.diy_hidden_cube_z = float(kwargs.pop("diy_hidden_cube_z", -1.0))
        self.cover_start_step = int(kwargs.pop("cover_start_step", self.COVER_START_STEP))
        self.cover_duration_steps = int(kwargs.pop("cover_duration_steps", self.COVER_DURATION_STEPS))
        self.cover_height = float(kwargs.pop("cover_height", self.COVER_HEIGHT))
        self.cover_settle_steps = int(kwargs.pop("cover_settle_steps", self.COVER_SETTLE_STEPS))
        kwargs.setdefault("difficulty", "easy")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _to_np(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _actor_pose_np(self, actor):
        pose = actor.pose
        p = self._to_np(pose.p)[:3]
        q = self._to_np(pose.q)[:4]
        return p, q

    def _set_actor_pose_np(self, actor, p: np.ndarray, q: np.ndarray):
        actor.set_pose(sapien.Pose(p=np.asarray(p, dtype=np.float32), q=np.asarray(q, dtype=np.float32)))
        for setter in ("set_linear_velocity", "set_angular_velocity"):
            if hasattr(actor, setter):
                try:
                    getattr(actor, setter)([0.0, 0.0, 0.0])
                except Exception:
                    pass

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.spawned_bins = []
        self._diy_bin_final_poses = []
        self._diy_bin_high_poses = []

        for i, xy in enumerate(self.diy_bin_positions):
            bin_actor = build_bin(
                self,
                callsign=f"diy_bin_{i}",
                position=[float(xy[0]), float(xy[1]), 0.002],
                z_rotation_deg=0.0,
            )
            self.spawned_bins.append(bin_actor)
            setattr(self, f"bin_{i}", bin_actor)

            final_p, q = self._actor_pose_np(bin_actor)
            high_p = final_p.copy()
            high_p[2] += self.cover_height
            self._diy_bin_final_poses.append((final_p, q))
            self._diy_bin_high_poses.append((high_p, q))
            self._set_actor_pose_np(bin_actor, high_p, q)

        spawned_dynamic_cubes = []
        self._diy_cube_nominal_poses = []
        self.color_names = list(self.diy_color_names)
        for i, (xy, color, color_name) in enumerate(
            zip(self.diy_bin_positions, self.diy_cube_colors, self.color_names)
        ):
            cube_actor = spawn_fixed_cube(
                self,
                position=[float(xy[0]), float(xy[1])],
                half_size=self.cube_half_size / 1.2,
                color=tuple(color),
                name_prefix=f"target_cube_{color_name}",
                yaw=0.0,
                dynamic=True,
            )
            spawned_dynamic_cubes.append(cube_actor)
            setattr(self, f"target_cube_{color_name}", cube_actor)
            setattr(self, f"target_cube_{i}", cube_actor)
            self._diy_cube_nominal_poses.append(self._actor_pose_np(cube_actor))

        tasks = [
            {
                "func": lambda: static_check(self, timestep=int(self.elapsed_steps), static_steps=64),
                "name": "static",
                "subgoal_segment": "static",
                "choice_label": "static",
                "demonstration": True,
                "failure_func": None,
                "solve": lambda env, planner: solve_hold_obj(env, planner, static_steps=64),
            },
            {
                "func": (lambda: is_bin_pickup(self, obj=self.bin_0)),
                "name": f"pick up the container that hides the {self.color_names[0]} cube",
                "subgoal_segment": f"pick up the container at <> that hides the {self.color_names[0]} cube",
                "choice_label": "pick up the container",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, [bin for bin in self.spawned_bins if bin != self.bin_0]),
                "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.bin_0),
                "segment": self.bin_0,
            },
        ]

        self.task_list = tasks
        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        self.fail_grasp_task_index = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        for actor, (_, q), (high_p, _) in zip(
            self.spawned_bins, self._diy_bin_final_poses, self._diy_bin_high_poses
        ):
            self._set_actor_pose_np(actor, high_p, q)
        self._apply_cube_presence()
        self._apply_cover_animation(cur_step=0)

    def set_present_cube_indices(self, present_cube_indices):
        """Select logically present cubes without rebuilding the renderer."""
        indices = {int(i) for i in present_cube_indices}
        if not indices.issubset({0, 1, 2}):
            raise ValueError(f"Cube indices must be in [0, 2], got {sorted(indices)}")
        self.diy_present_cube_indices = indices
        if hasattr(self, "_diy_cube_nominal_poses"):
            self._apply_cube_presence()

    def _apply_cube_presence(self):
        for i, (nominal_p, nominal_q) in enumerate(self._diy_cube_nominal_poses):
            actor = getattr(self, f"target_cube_{i}")
            if i in self.diy_present_cube_indices:
                p = nominal_p
            else:
                p = nominal_p.copy()
                p[2] = self.diy_hidden_cube_z
            self._set_actor_pose_np(actor, p, nominal_q)

    def _apply_cover_animation(self, cur_step: int | None = None):
        if not hasattr(self, "spawned_bins"):
            return
        if cur_step is None:
            elapsed = getattr(self, "elapsed_steps", 0)
            if isinstance(elapsed, torch.Tensor):
                cur_step = int(elapsed.detach().cpu().reshape(-1)[0].item())
            else:
                cur_step = int(elapsed)

        start = self.cover_start_step
        duration = max(1, self.cover_duration_steps)
        release_step = start + duration + max(0, self.cover_settle_steps)
        if cur_step > release_step:
            return

        alpha = self._smoothstep((cur_step - start) / duration)
        if cur_step < start:
            alpha = 0.0

        for actor, (final_p, q), (high_p, _) in zip(
            self.spawned_bins, self._diy_bin_final_poses, self._diy_bin_high_poses
        ):
            p = high_p * (1.0 - alpha) + final_p * alpha
            self._set_actor_pose_np(actor, p, q)

    def step(self, action):
        # Override VideoUnmask.step so we do not call the original
        # lift_and_drop_objects_back_to_original teleport logic.
        self._apply_cover_animation()
        obs, reward, terminated, truncated, info = BaseEnv.step(self, action)
        return obs, reward, terminated, truncated, info
