"""Collect small RoboMME occlusion demos into HDF5.

This collector is intentionally independent from ``RobommeRecordWrapper``.
Some planner paths call the underlying ManiSkill env directly, so wrapper
step hooks may not see low-level steps.  Here we collect the public
BenchmarkEnvBuilder observation batches and write them to the documented
``record_dataset_<EnvID>.h5`` layout directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import tyro

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env.utils import generate_sample_actions


TaskID = Literal[
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
]
DatasetType = Literal["train", "test", "val"]
ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]


def _np(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _list_get(obs: dict, key: str, idx: int, default):
    values = obs.get(key)
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return default


def _str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.reshape(-1)[0])
    return str(value)


def _bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0].item()) if value.size else False
    if isinstance(value, (list, tuple)):
        return _bool(value[-1]) if value else False
    return bool(value)


def _normalize_action(action, action_space: str):
    if action is None:
        return None, np.full(7, np.nan, dtype=np.float32), np.full(7, np.nan, dtype=np.float32), "{}"

    if action_space == "multi_choice":
        payload = json.dumps(action, ensure_ascii=False)
        return None, np.full(7, np.nan, dtype=np.float32), np.full(7, np.nan, dtype=np.float32), payload

    arr = _np(action, np.float32).reshape(-1)
    joint = None
    eef = np.full(7, np.nan, dtype=np.float32)
    waypoint = np.full(7, np.nan, dtype=np.float32)

    if action_space == "joint_angle":
        joint = arr[:8] if arr.size >= 8 else np.pad(arr, (0, max(0, 8 - arr.size)), constant_values=-1.0)[:8]
    elif action_space in {"ee_pose", "waypoint"}:
        eef = arr[:7] if arr.size >= 7 else np.pad(arr, (0, max(0, 7 - arr.size)), constant_values=np.nan)[:7]
        if action_space == "waypoint":
            waypoint = eef.copy()
    return joint, eef, waypoint, "{}"


def _write_string(ds_group, name: str, value: str):
    ds_group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


def _write_step(
    episode_group,
    timestep: int,
    obs: dict,
    obs_idx: int,
    info: dict,
    action,
    action_space: str,
    is_video_demo: bool,
    is_subgoal_boundary: bool,
):
    front = _np(_list_get(obs, "front_rgb_list", obs_idx, np.zeros((256, 256, 3), dtype=np.uint8)), np.uint8)
    wrist = _np(_list_get(obs, "wrist_rgb_list", obs_idx, np.zeros_like(front)), np.uint8)
    h, w = front.shape[:2]

    front_depth = _np(_list_get(obs, "front_depth_list", obs_idx, np.zeros((h, w, 1), dtype=np.int16)), np.int16)
    wrist_depth = _np(_list_get(obs, "wrist_depth_list", obs_idx, np.zeros((h, w, 1), dtype=np.int16)), np.int16)
    joint_state = _np(_list_get(obs, "joint_state_list", obs_idx, np.zeros(7, dtype=np.float32)), np.float32).reshape(-1)[:7]
    eef_state = _np(_list_get(obs, "eef_state_list", obs_idx, np.zeros(6, dtype=np.float32)), np.float32).reshape(-1)[:6]
    gripper_state = _np(_list_get(obs, "gripper_state_list", obs_idx, np.zeros(2, dtype=np.float32)), np.float32).reshape(-1)[:2]
    front_ext = _np(_list_get(obs, "front_camera_extrinsic_list", obs_idx, np.zeros((3, 4), dtype=np.float32)), np.float32).reshape(3, 4)
    wrist_ext = _np(_list_get(obs, "wrist_camera_extrinsic_list", obs_idx, np.zeros((3, 4), dtype=np.float32)), np.float32).reshape(3, 4)

    joint_action, eef_action, waypoint_action, choice_action = _normalize_action(action, action_space)
    if not np.all(np.isfinite(eef_action)):
        gripper_cmd = float(joint_action[-1]) if joint_action is not None and np.asarray(joint_action).size else 1.0
        eef_action = np.concatenate([eef_state.astype(np.float32), np.asarray([gripper_cmd], dtype=np.float32)])

    ts_group = episode_group.create_group(f"timestep_{timestep}")
    obs_group = ts_group.create_group("obs")
    obs_group.create_dataset("front_rgb", data=front)
    obs_group.create_dataset("wrist_rgb", data=wrist)
    obs_group.create_dataset("front_depth", data=front_depth)
    obs_group.create_dataset("wrist_depth", data=wrist_depth)
    obs_group.create_dataset("joint_state", data=joint_state)
    obs_group.create_dataset("eef_state", data=eef_state)
    obs_group.create_dataset("gripper_state", data=gripper_state)
    obs_group.create_dataset("is_gripper_close", data=bool(np.any(gripper_state < 0.03)))
    obs_group.create_dataset("front_camera_extrinsic", data=front_ext)
    obs_group.create_dataset("wrist_camera_extrinsic", data=wrist_ext)

    action_group = ts_group.create_group("action")
    if joint_action is None:
        _write_string(action_group, "joint_action", "None")
    else:
        action_group.create_dataset("joint_action", data=joint_action.astype(np.float32))
    action_group.create_dataset("eef_action", data=eef_action.astype(np.float32))
    action_group.create_dataset("waypoint_action", data=waypoint_action.astype(np.float32))
    _write_string(action_group, "choice_action", choice_action)

    subgoal = _str(info.get("simple_subgoal") or info.get("task_goal") or "")
    grounded = _str(info.get("grounded_subgoal") or subgoal)
    info_group = ts_group.create_group("info")
    info_group.create_dataset("simple_subgoal", data=subgoal.encode("utf-8"))
    info_group.create_dataset("simple_subgoal_online", data=subgoal.encode("utf-8"))
    info_group.create_dataset("grounded_subgoal", data=grounded.encode("utf-8"))
    info_group.create_dataset("grounded_subgoal_online", data=grounded.encode("utf-8"))
    info_group.create_dataset("is_video_demo", data=bool(is_video_demo))
    info_group.create_dataset("is_subgoal_boundary", data=bool(is_subgoal_boundary))
    info_group.create_dataset("is_completed", data=_bool(info.get("is_completed", False)))


def _write_setup(episode_group, info: dict, episode_idx: int):
    setup = episode_group.create_group("setup")
    setup.create_dataset("seed", data=int(info.get("seed", episode_idx) or episode_idx))
    _write_string(setup, "difficulty", _str(info.get("difficulty", "")))
    goals = info.get("task_goal", [""])
    if not isinstance(goals, (list, tuple)):
        goals = [goals]
    setup.create_dataset(
        "task_goal",
        data=np.asarray([str(g) for g in goals], dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    choices = info.get("available_multi_choices", "")
    _write_string(setup, "available_multi_choices", json.dumps(choices, ensure_ascii=False) if not isinstance(choices, str) else choices)

    front_k = info.get("front_camera_intrinsic", np.eye(3, dtype=np.float32))
    wrist_k = info.get("wrist_camera_intrinsic", np.eye(3, dtype=np.float32))
    setup.create_dataset("front_camera_intrinsic", data=_np(front_k, np.float32).reshape(3, 3))
    setup.create_dataset("wrist_camera_intrinsic", data=_np(wrist_k, np.float32).reshape(3, 3))


def collect(
    task_id: TaskID = "VideoUnmask",
    dataset: DatasetType = "test",
    action_space_type: ActionSpaceType = "joint_angle",
    output_dir: str = "runs/occlusion_collector_hai_machine",
    episode_idx: int = 0,
    num_episodes: int = 1,
    max_steps: int = 300,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h5_path = output / f"record_dataset_{task_id}.h5"

    builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=action_space_type,
        gui_render=False,
        max_steps=max_steps,
    )

    with h5py.File(h5_path, "w") as h5:
        for ep in range(episode_idx, episode_idx + num_episodes):
            env = builder.make_env_for_episode(
                ep,
                max_steps=max_steps,
                include_front_depth=True,
                include_wrist_depth=True,
                include_front_camera_extrinsic=True,
                include_wrist_camera_extrinsic=True,
                include_front_camera_intrinsic=True,
                include_wrist_camera_intrinsic=True,
                include_available_multi_choices=True,
            )
            obs, info = env.reset()
            ep_group = h5.create_group(f"episode_{ep}")
            _write_setup(ep_group, info, ep)

            timestep = 0
            reset_n = len(obs.get("front_rgb_list", []))
            for i in range(reset_n):
                _write_step(
                    ep_group,
                    timestep,
                    obs,
                    i,
                    info,
                    None,
                    action_space_type,
                    # Reset frames have no action.  Mark all of them as video
                    # demo/context so EpisodeDatasetResolver starts action
                    # training from the first real env.step result.
                    is_video_demo=True,
                    is_subgoal_boundary=i == 0,
                )
                timestep += 1

            status = "ongoing"
            for step_i, action in enumerate(generate_sample_actions(action_space_type, env=env)):
                obs, _, terminated, truncated, info = env.step(action)
                status = str(info.get("status", "unknown"))
                n = len(obs.get("front_rgb_list", []))
                for i in range(n):
                    _write_step(
                        ep_group,
                        timestep,
                        obs,
                        i,
                        info,
                        action,
                        action_space_type,
                        is_video_demo=False,
                        is_subgoal_boundary=i == 0,
                    )
                    timestep += 1
                if status == "error":
                    print(f"episode {ep}: step error: {info.get('error_message', 'unknown')}")
                    break
                if _bool(terminated) or _bool(truncated) or step_i + 1 >= max_steps:
                    break

            ep_group.attrs["status"] = status
            ep_group.attrs["num_timesteps"] = timestep
            env.close()
            print(f"episode {ep}: wrote {timestep} timesteps, status={status}")

    print(f"saved {h5_path}")


if __name__ == "__main__":
    tyro.cli(collect)
