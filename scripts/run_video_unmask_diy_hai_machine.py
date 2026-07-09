"""Generate a deterministic VideoUnmask DIY demo video."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)

import gymnasium as gym


OUT_DIR = Path("outputs/occlusion_unmask_diy_hai_machine_2026-07-09")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb(obs, preferred: str):
    sensor = obs.get("sensor_data", {})
    if preferred in sensor and "rgb" in sensor[preferred]:
        arr = _to_np(sensor[preferred]["rgb"])
    else:
        keys = list(sensor.keys())
        if not keys:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        arr = _to_np(sensor[keys[0]]["rgb"])
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _frame(obs, label: str):
    front = _rgb(obs, "base_camera")
    wrist = _rgb(obs, "hand_camera")
    if front.shape[:2] != wrist.shape[:2]:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)
    frame = np.hstack([front, wrist])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _hold_action(env):
    qpos = env.unwrapped.agent.robot.get_qpos()
    qpos = _to_np(qpos).reshape(-1)
    return np.concatenate([qpos[:7], [1.0]]).astype(np.float32)


def _patch_planner(planner):
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


def main():
    out_path = OUT_DIR / "VideoUnmaskDIYHaiMachine_fixed_positions_smooth_cover_pickup.mp4"

    env = gym.make(
        "VideoUnmaskDIYHaiMachine",
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=0,
        difficulty="easy",
    )

    frames = []
    obs, _ = env.reset()
    frames.append(_frame(obs, "start: cubes visible, covers above"))

    cover_steps = (
        env.unwrapped.cover_start_step
        + env.unwrapped.cover_duration_steps
        + env.unwrapped.cover_settle_steps
    )
    for step in range(cover_steps):
        obs, _, _, _, _ = env.step(_hold_action(env))
        if step % 1 == 0:
            frames.append(_frame(obs, "smooth cover descent"))

    planner = FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    _patch_planner(planner)

    original_step = env.step

    def recording_step(action):
        result = original_step(action)
        frames.append(_frame(result[0], "robot picks up target cover"))
        return result

    env.step = recording_step

    for task in env.unwrapped.task_list:
        if "pick up the container" not in task.get("name", ""):
            continue
        result = task["solve"](env, planner)
        print("solve_result", type(result).__name__, result if isinstance(result, int) else "")
        break

    evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
    print("evaluation", evaluation)

    if frames:
        frames.extend([frames[-1]] * 30)
    env.close()

    imageio.mimsave(out_path, frames, fps=30, quality=8)
    print(out_path)
    print("frames", len(frames))


if __name__ == "__main__":
    main()
