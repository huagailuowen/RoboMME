"""Collect RoboMME DIY occlusion demos as LeRobot datasets.

This hai-machine collector writes three local datasets by default:

- sequence: visible cubes -> covers descend -> reveal each cover
- swap: visible cubes -> covers descend -> swap cover 0/1 -> reveal each cover
- reveal: fully covered -> reveal all -> swap cover 0/1 -> reveal all again

Each episode has a 0.5 probability of missing exactly one random cube while
all three covers are still present.  Per-frame object poses are stored in the
LeRobot data under ``observation.object_poses`` so the full scene can be
reconstructed alongside the episode metadata JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = REPO_ROOT.parent / "FastWAM-TTT" / "src"
if str(FASTWAM_SRC) not in sys.path:
    sys.path.insert(0, str(FASTWAM_SRC))

from fastwam.datasets.lerobot.lerobot import lerobot_dataset as lerobot_dataset_module
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)
from robomme.robomme_env.utils.subgoal_planner_func import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT/data/robomme-occlusion"
)
CASE_TYPES = ("sequence", "swap", "reveal")
OBJECT_POSE_ORDER = ("bin_0", "bin_1", "bin_2", "cube_0", "cube_1", "cube_2")
COLOR_NAMES = ("green", "red", "blue")
CUBE_COLORS = ((0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1))
VIDEO_CRF = 18
JPEG_QUALITY = 98
_VIDEO_ENCODER_PATCHED = False


def _to_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vec(value) -> np.ndarray:
    return _to_np(value).reshape(-1).astype(np.float32)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def _rgb(obs: dict[str, Any], preferred: str, resize_to: tuple[int, int] | None):
    sensor = obs.get("sensor_data", {})
    if preferred in sensor and "rgb" in sensor[preferred]:
        arr = _to_np(sensor[preferred]["rgb"])
    else:
        keys = list(sensor.keys())
        if not keys:
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            arr = _to_np(sensor[keys[0]]["rgb"])
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    arr = arr.astype(np.uint8)
    if resize_to is not None and arr.shape[:2] != resize_to:
        arr = cv2.resize(arr, (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA)
    return arr


def _patch_lerobot_video_crf(crf: int) -> None:
    global _VIDEO_ENCODER_PATCHED
    if _VIDEO_ENCODER_PATCHED:
        return
    original = lerobot_dataset_module.encode_video_frames

    def encode_with_crf(*args: Any, **kwargs: Any) -> None:
        kwargs["crf"] = int(crf)
        return original(*args, **kwargs)

    lerobot_dataset_module.encode_video_frames = encode_with_crf
    _VIDEO_ENCODER_PATCHED = True


def _write_episode_image(
    dataset: LeRobotDataset,
    key: str,
    frame_index: int,
    image: np.ndarray,
) -> None:
    path = dataset._get_image_file_path(
        episode_index=dataset.episode_buffer["episode_index"],
        image_key=key,
        frame_index=frame_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


def _robot_state(env) -> np.ndarray:
    qpos = _vec(env.unwrapped.agent.robot.get_qpos())
    gripper = float(np.mean(qpos[7:])) if qpos.size > 7 else 1.0
    return np.concatenate([qpos[:7], np.asarray([gripper], dtype=np.float32)]).astype(np.float32)


def _eef_state(env) -> np.ndarray:
    tcp_pose = env.unwrapped.agent.tcp.pose
    p = _vec(tcp_pose.p)[:3]
    q = _vec(tcp_pose.q)[:4]
    return np.concatenate([p, q]).astype(np.float32)


def _hold_action(env, gripper: float = 1.0) -> np.ndarray:
    qpos = _vec(env.unwrapped.agent.robot.get_qpos())
    return np.concatenate([qpos[:7], [float(gripper)]]).astype(np.float32)


def _normalize_action(action, env) -> np.ndarray:
    if action is None:
        return _hold_action(env)
    arr = _vec(action)
    if arr.size >= 8:
        return arr[:8].astype(np.float32)
    return np.pad(arr, (0, 8 - arr.size), constant_values=0.0).astype(np.float32)


def _pose_from_actor(actor) -> tuple[np.ndarray, np.ndarray]:
    return _vec(actor.pose.p)[:3], _vec(actor.pose.q)[:4]


def _object_pose_features(env) -> tuple[np.ndarray, np.ndarray]:
    poses: list[np.ndarray] = []
    exists: list[float] = []
    for i in range(3):
        actor = getattr(env.unwrapped, f"bin_{i}", None)
        if actor is None:
            poses.append(np.zeros(7, dtype=np.float32))
            exists.append(0.0)
        else:
            p, q = _pose_from_actor(actor)
            poses.append(np.concatenate([p, q]).astype(np.float32))
            exists.append(1.0)
    present_cube_indices = getattr(
        env.unwrapped, "diy_present_cube_indices", {0, 1, 2}
    )
    for i in range(3):
        actor = getattr(env.unwrapped, f"target_cube_{i}", None)
        if actor is None or i not in present_cube_indices:
            poses.append(np.zeros(7, dtype=np.float32))
            exists.append(0.0)
        else:
            p, q = _pose_from_actor(actor)
            poses.append(np.concatenate([p, q]).astype(np.float32))
            exists.append(1.0)
    return np.concatenate(poses).astype(np.float32), np.asarray(exists, dtype=np.float32)


def _set_actor_pose(env, actor, p, q) -> None:
    env.unwrapped._set_actor_pose_np(
        actor,
        np.asarray(p, dtype=np.float32),
        np.asarray(q, dtype=np.float32),
    )


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _patch_planner(planner) -> None:
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def move_screw_then_rrt(*args, **kwargs):
        for _ in range(3):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        for _ in range(3):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1

    planner.move_to_pose_with_screw = move_screw_then_rrt


def _make_bin_grasp_poses(env, obj):
    env_u = env.unwrapped
    obb = get_actor_obb(obj)
    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    target_closing = (
        env_u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        .detach()
        .cpu()
        .numpy()
    )
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,
    )
    grasp_pose = env_u.agent.build_grasp_pose(
        approaching,
        grasp_info["closing"],
        obj.pose.sp.p,
    )

    reach_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.15])
    reach_p = _vec(reach_pose.p)[:3]
    reach_p[2] = 0.20
    reach_pose = sapien.Pose(p=reach_p, q=_vec(reach_pose.q)[:4])

    grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.01])
    lift_p = _vec(obj.pose.p)[:3]
    lift_p[2] = 0.22
    lift_pose = sapien.Pose(p=lift_p, q=_vec(grasp_pose.q)[:4])
    return reach_pose, grasp_pose, lift_pose


@dataclass
class EpisodeRecorder:
    dataset: LeRobotDataset
    env: Any
    case_type: str
    episode_index: int
    image_size: tuple[int, int] | None

    def __post_init__(self):
        self.phase_label = "init"
        self.phase_ranges: list[dict[str, Any]] = []
        self._phase_start = 0
        self.frame_count = 0

    def set_phase(self, label: str) -> None:
        if label == self.phase_label:
            return
        if self.frame_count > self._phase_start:
            self.phase_ranges.append(
                {
                    "label": self.phase_label,
                    "start_frame": self._phase_start,
                    "end_frame_exclusive": self.frame_count,
                }
            )
        self.phase_label = label
        self._phase_start = self.frame_count

    def add(self, obs: dict[str, Any], action=None) -> None:
        object_poses, object_exists = _object_pose_features(self.env)
        base_image = _rgb(obs, "base_camera", self.image_size)
        wrist_image = _rgb(obs, "hand_camera", self.image_size)
        frame = {
            "observation.images.image": base_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": _robot_state(self.env),
            "observation.eef_state": _eef_state(self.env),
            "observation.object_poses": object_poses,
            "observation.object_exists": object_exists,
            "action": _normalize_action(action, self.env),
        }
        task = [
            f"RoboMME occlusion-memory {self.case_type}",
            self.phase_label,
            "successful scripted physical-gripper rollout",
            "success",
        ]
        self.dataset.add_frame(frame, task=task)
        _write_episode_image(
            self.dataset, "observation.images.image", self.frame_count, base_image
        )
        _write_episode_image(
            self.dataset,
            "observation.images.wrist_image",
            self.frame_count,
            wrist_image,
        )
        self.frame_count += 1

    def finish(self) -> None:
        if self.frame_count > self._phase_start:
            self.phase_ranges.append(
                {
                    "label": self.phase_label,
                    "start_frame": self._phase_start,
                    "end_frame_exclusive": self.frame_count,
                }
            )


def _wait(env, steps: int, gripper: float = 1.0) -> None:
    for _ in range(int(steps)):
        env.step(_hold_action(env, gripper=gripper))


def scripted_swap_bins(
    env,
    bin_a,
    bin_b,
    *,
    steps: int = 80,
    lane_offset: float = 0.075,
) -> None:
    pa0, qa = _pose_from_actor(bin_a)
    pb0, qb = _pose_from_actor(bin_b)
    delta = pb0[:2] - pa0[:2]
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        lateral = np.array([0.0, lane_offset], dtype=np.float32)
    else:
        direction = delta / norm
        lateral = np.array([-direction[1], direction[0]], dtype=np.float32) * lane_offset

    for i in range(int(steps)):
        alpha = _smoothstep(i / max(1, steps - 1))
        lane = np.sin(np.pi * alpha) * lateral
        pa = pa0 * (1.0 - alpha) + pb0 * alpha
        pb = pb0 * (1.0 - alpha) + pa0 * alpha
        pa[:2] += lane
        pb[:2] -= lane
        _set_actor_pose(env, bin_a, pa, qa)
        _set_actor_pose(env, bin_b, pb, qb)
        env.step(_hold_action(env, gripper=1.0))

    _set_actor_pose(env, bin_a, pb0, qa)
    _set_actor_pose(env, bin_b, pa0, qb)
    _wait(env, 8, gripper=1.0)


def pick_show_put_back(env, planner, obj, show_offset_xy) -> None:
    original_p, _ = _pose_from_actor(obj)
    reach_pose, grasp_pose, lift_pose = _make_bin_grasp_poses(env, obj)
    grasp_q = _vec(grasp_pose.q)[:4]

    planner.open_gripper()
    planner.move_to_pose_with_screw(reach_pose)
    planner.open_gripper()
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(lift_pose)

    show_p = original_p.copy()
    show_p[:2] += np.asarray(show_offset_xy, dtype=np.float32)
    show_p[2] = 0.22
    planner.move_to_pose_with_screw(sapien.Pose(p=show_p, q=grasp_q))
    _wait(env, 8, gripper=-1.0)

    back_high_p = original_p.copy()
    back_high_p[2] = 0.22
    planner.move_to_pose_with_screw(sapien.Pose(p=back_high_p, q=grasp_q))
    planner.move_to_pose_with_screw(grasp_pose)
    _wait(env, 2, gripper=-1.0)
    planner.open_gripper()
    _wait(env, 6, gripper=1.0)
    planner.move_to_pose_with_screw(reach_pose)
    _wait(env, 2, gripper=1.0)


def create_lerobot_dataset(root: Path, repo_id: str, fps: int, image_size: tuple[int, int]):
    image_shape = (3, image_size[0], image_size[1])
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["channel", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "observation.eef_state": {"dtype": "float32", "shape": (7,)},
        "observation.object_poses": {"dtype": "float32", "shape": (42,)},
        "observation.object_exists": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (8,)},
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type="panda",
        features=features,
        use_videos=True,
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )


def _save_episode_compat(dataset: LeRobotDataset) -> None:
    dataset.save_episode()


def _finalize_dataset(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "finalize"):
        dataset.finalize()


def _episode_setup(rng: np.random.Generator, missing_probability: float) -> dict[str, Any]:
    missing_enabled = bool(rng.random() < missing_probability)
    missing_cube_index = int(rng.integers(0, 3)) if missing_enabled else None
    present_cube_indices = [i for i in range(3) if i != missing_cube_index]
    return {
        "missing_enabled": missing_enabled,
        "missing_cube_index": missing_cube_index,
        "present_cube_indices": present_cube_indices,
    }


def _env_kwargs(case_type: str, present_cube_indices: list[int], seed: int) -> dict[str, Any]:
    kwargs = {
        "obs_mode": "rgb+depth+segmentation",
        "control_mode": "pd_joint_pos",
        "render_mode": "rgb_array",
        "reward_mode": "dense",
        "seed": seed,
        "difficulty": "easy",
        "diy_present_cube_indices": present_cube_indices,
    }
    if case_type == "reveal":
        kwargs.update(
            {
                "cover_start_step": -1,
                "cover_duration_steps": 1,
                "cover_settle_steps": 0,
            }
        )
    return kwargs


def _initial_scene_metadata(env) -> dict[str, Any]:
    bins = []
    cubes = []
    for i in range(3):
        actor = getattr(env.unwrapped, f"bin_{i}", None)
        p, q = _pose_from_actor(actor)
        bins.append({"index": i, "pose": np.concatenate([p, q])})
    present_cube_indices = getattr(
        env.unwrapped, "diy_present_cube_indices", {0, 1, 2}
    )
    for i in range(3):
        actor = getattr(env.unwrapped, f"target_cube_{i}", None)
        if actor is None or i not in present_cube_indices:
            cubes.append({"index": i, "present": False, "pose": None})
        else:
            p, q = _pose_from_actor(actor)
            cubes.append({"index": i, "present": True, "pose": np.concatenate([p, q])})
    return {"bins": bins, "cubes": cubes}


def _install_recording_step(env, recorder: EpisodeRecorder):
    original_step = env.step

    def recording_step(action):
        result = original_step(action)
        recorder.add(result[0], action)
        return result

    env.step = recording_step
    return original_step


def _make_planner(env):
    planner = FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    _patch_planner(planner)
    return planner


def _run_reveal_sequence(env, planner, recorder: EpisodeRecorder, prefix: str) -> None:
    sequence = [
        (0, env.unwrapped.bin_0, [0.00, 0.12]),
        (1, env.unwrapped.bin_1, [-0.10, 0.00]),
        (2, env.unwrapped.bin_2, [0.00, 0.12]),
    ]
    for idx, obj, offset in sequence:
        recorder.set_phase(f"{prefix}: reveal cover {idx} and put back")
        pick_show_put_back(env, planner, obj, offset)


def collect_episode(
    dataset: LeRobotDataset,
    env,
    planner,
    case_type: str,
    episode_index: int,
    seed: int,
    present_cube_indices: list[int],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    env_kwargs = _env_kwargs(case_type, present_cube_indices, seed)
    env.unwrapped.set_present_cube_indices(present_cube_indices)
    obs, _ = env.reset(seed=seed)

    recorder = EpisodeRecorder(
        dataset=dataset,
        env=env,
        case_type=case_type,
        episode_index=episode_index,
        image_size=image_size,
    )
    original_step = _install_recording_step(env, recorder)

    try:
        if case_type == "reveal":
            recorder.set_phase("start: fully covered")
            recorder.add(obs, _hold_action(env, gripper=1.0))
            cover_steps = 0
        else:
            recorder.set_phase("start: cubes visible, covers above")
            recorder.add(obs, _hold_action(env, gripper=1.0))
            cover_steps = (
                env.unwrapped.cover_start_step
                + env.unwrapped.cover_duration_steps
                + env.unwrapped.cover_settle_steps
            )
            recorder.set_phase("smooth cover descent")
            for _ in range(cover_steps):
                env.step(_hold_action(env, gripper=1.0))

        scene_meta = _initial_scene_metadata(env)
        initial_robot_state = _robot_state(env)
        initial_eef_state = _eef_state(env)

        if case_type == "sequence":
            _run_reveal_sequence(env, planner, recorder, "sequence")
        elif case_type == "swap":
            recorder.set_phase("scripted swap: cover 0 <-> cover 1")
            scripted_swap_bins(env, env.unwrapped.bin_0, env.unwrapped.bin_1)
            _run_reveal_sequence(env, planner, recorder, "after swap")
        elif case_type == "reveal":
            _run_reveal_sequence(env, planner, recorder, "before swap")
            recorder.set_phase("scripted swap: cover 0 <-> cover 1")
            scripted_swap_bins(env, env.unwrapped.bin_0, env.unwrapped.bin_1)
            _run_reveal_sequence(env, planner, recorder, "after swap")
        else:
            raise ValueError(f"Unknown case_type: {case_type}")

        recorder.set_phase("done: all covers restored")
        _wait(env, 30, gripper=1.0)
        recorder.finish()
        final_meta = _initial_scene_metadata(env)
    finally:
        env.step = original_step

    return {
        "episode_index": episode_index,
        "case_type": case_type,
        "seed": seed,
        "present_cube_indices": present_cube_indices,
        "missing_enabled": len(present_cube_indices) < 3,
        "missing_cube_index": next((i for i in range(3) if i not in present_cube_indices), None),
        "color_names": list(COLOR_NAMES),
        "cube_colors_rgba": [list(c) for c in CUBE_COLORS],
        "env_kwargs": env_kwargs,
        "swap_pair": [0, 1] if case_type in {"swap", "reveal"} else None,
        "swap_steps": 80 if case_type in {"swap", "reveal"} else 0,
        "swap_lane_offset": 0.075 if case_type in {"swap", "reveal"} else 0.0,
        "object_pose_order": list(OBJECT_POSE_ORDER),
        "object_exists_order": list(OBJECT_POSE_ORDER),
        "initial_scene_after_covering": scene_meta,
        "initial_robot_state_after_covering": initial_robot_state,
        "initial_eef_state_after_covering": initial_eef_state,
        "final_scene": final_meta,
        "cover_animation_steps": cover_steps,
        "phase_ranges": recorder.phase_ranges,
        "num_frames": recorder.frame_count,
        "per_frame_object_pose_storage": "LeRobot key observation.object_poses, flattened as 3 bins then 3 cubes, each [x,y,z,qw,qx,qy,qz]",
        "per_frame_object_exists_storage": "LeRobot key observation.object_exists, order matches object_pose_order",
    }


def collect_case(
    case_type: str,
    output_root: Path,
    episodes_per_case: int,
    seed: int,
    missing_probability: float,
    fps: int,
    image_size: tuple[int, int],
    overwrite: bool,
) -> None:
    dataset_name = f"robomme_occlusion_{case_type}_missing05_hai-machine_lerobot"
    dataset_root = output_root / dataset_name
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} already exists; pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_lerobot_dataset(
        root=dataset_root,
        repo_id=f"local/{dataset_name}",
        fps=fps,
        image_size=image_size,
    )

    rng = np.random.default_rng(seed)
    metadata_path = dataset_root / "robomme_occlusion_generation_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(),
        "generation_status": "running",
        "completed_episodes": 0,
        "dataset_type": "robomme_occlusion_diy_hai-machine_lerobot",
        "lerobot_codebase_version": "v2.1",
        "lerobot_storage_layout": "per_episode_files",
        "case_type": case_type,
        "episodes_per_case": episodes_per_case,
        "missing_probability": missing_probability,
        "seed": seed,
        "fps": fps,
        "image_size_hw": list(image_size),
        "output_root": str(dataset_root),
        "env_id": "VideoUnmaskDIYHaiMachine",
        "control_mode": "pd_joint_pos",
        "action_dim": 8,
        "state_dim": 8,
        "object_pose_dim": 42,
        "object_pose_order": list(OBJECT_POSE_ORDER),
        "scene_definition": {
            "bin_positions_xy_m": VideoUnmaskDIYHaiMachine.DIY_BIN_POSITIONS,
            "cube_color_names": list(COLOR_NAMES),
            "cube_colors_rgba": [list(c) for c in CUBE_COLORS],
            "cover_height_m": VideoUnmaskDIYHaiMachine.COVER_HEIGHT,
            "cover_start_step": VideoUnmaskDIYHaiMachine.COVER_START_STEP,
            "cover_duration_steps": VideoUnmaskDIYHaiMachine.COVER_DURATION_STEPS,
            "cover_settle_steps": VideoUnmaskDIYHaiMachine.COVER_SETTLE_STEPS,
        },
        "notes": [
            "All three covers are always present.",
            "With probability 0.5, exactly one randomly selected cube is not spawned.",
            "Per-frame object poses and existence masks are stored as LeRobot observation fields.",
        ],
        "episodes": [],
    }
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")

    bootstrap_kwargs = _env_kwargs(case_type, [0, 1, 2], seed)
    env = gym.make("VideoUnmaskDIYHaiMachine", **bootstrap_kwargs)
    env.reset(seed=seed)
    planner = _make_planner(env)
    try:
        for ep in range(episodes_per_case):
            ep_seed = int(seed + ep)
            setup = _episode_setup(rng, missing_probability)
            print(
                f"[{case_type}] episode {ep + 1}/{episodes_per_case} "
                f"seed={ep_seed} present={setup['present_cube_indices']}",
                flush=True,
            )
            episode_meta = collect_episode(
                dataset=dataset,
                env=env,
                planner=planner,
                case_type=case_type,
                episode_index=ep,
                seed=ep_seed,
                present_cube_indices=setup["present_cube_indices"],
                image_size=image_size,
            )
            episode_meta["missing_enabled"] = setup["missing_enabled"]
            episode_meta["missing_cube_index"] = setup["missing_cube_index"]
            metadata["episodes"].append(episode_meta)
            _save_episode_compat(dataset)
            metadata["completed_episodes"] = ep + 1
            metadata_path.write_text(
                json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
            )
            gc.collect()

        _finalize_dataset(dataset)
        metadata["generation_status"] = "completed"
        metadata["completed_at"] = datetime.now().isoformat()
        metadata_path.write_text(
            json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        metadata["generation_status"] = "failed"
        metadata["error"] = repr(exc)
        metadata_path.write_text(
            json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
        )
        raise
    finally:
        env.close()
        del planner, env
        gc.collect()

    print(f"[{case_type}] wrote {dataset_root}")
    print(f"[{case_type}] metadata {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes-per-case", type=int, default=100)
    parser.add_argument("--case-types", nargs="+", choices=CASE_TYPES, default=list(CASE_TYPES))
    parser.add_argument("--seed", type=int, default=2026070900)
    parser.add_argument("--missing-probability", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _patch_lerobot_video_crf(VIDEO_CRF)
    image_size = (int(args.image_size), int(args.image_size))
    for offset, case_type in enumerate(args.case_types):
        collect_case(
            case_type=case_type,
            output_root=args.output_root,
            episodes_per_case=args.episodes_per_case,
            seed=int(args.seed + offset * 100000),
            missing_probability=float(args.missing_probability),
            fps=int(args.fps),
            image_size=image_size,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
