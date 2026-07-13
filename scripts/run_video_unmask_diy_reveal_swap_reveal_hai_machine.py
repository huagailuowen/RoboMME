"""Generate a fixed-layout VideoUnmask reveal-swap-reveal demo.

The scene starts like ``run_video_unmask_diy_hai_machine.py``:

1. the scene starts fully covered,
2. the robot sequentially removes each cover and puts it back,
3. two covers swap positions,
4. the robot sequentially removes each cover and puts it back again.

This script is a local hai-machine demo script and intentionally keeps the
upstream RoboMME environments unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
import torch

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)
from robomme.robomme_env.utils.subgoal_planner_func import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


OUT_DIR = Path("outputs/occlusion_unmask_diy_reveal_swap_reveal_hai_machine_2026-07-09")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vec(value):
    return _to_np(value).reshape(-1).astype(np.float32)


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
        wrist = cv2.resize(
            wrist,
            (front.shape[1], front.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    frame = np.hstack([front, wrist])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _hold_action(env, gripper: float = 1.0):
    qpos = _vec(env.unwrapped.agent.robot.get_qpos())
    return np.concatenate([qpos[:7], [float(gripper)]]).astype(np.float32)


def _pose_from_actor(actor):
    return _vec(actor.pose.p)[:3], _vec(actor.pose.q)[:4]


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


def _wait(env, steps: int, gripper: float = 1.0):
    for _ in range(int(steps)):
        env.step(_hold_action(env, gripper=gripper))


def _set_actor_pose(env, actor, p, q):
    env.unwrapped._set_actor_pose_np(actor, np.asarray(p, dtype=np.float32), np.asarray(q, dtype=np.float32))


def _smoothstep(x: float):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def scripted_swap_bins(env, frames, bin_a, bin_b, label: str, steps: int = 80, lane_offset: float = 0.075):
    """Scripted shell-game-style swap for two covers.

    This keeps the three cubes untouched and moves only the two covers.  The
    covers remain near table height, so the scene looks like a covered-object
    swap instead of a reveal.
    """
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
        obs, _, _, _, _ = env.step(_hold_action(env, gripper=1.0))
        frames.append(_frame(obs, label))

    _set_actor_pose(env, bin_a, pb0, qa)
    _set_actor_pose(env, bin_b, pa0, qb)
    for _ in range(8):
        obs, _, _, _, _ = env.step(_hold_action(env, gripper=1.0))
        frames.append(_frame(obs, label))


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


def pick_show_put_back(env, planner, obj, show_offset_xy, label: str):
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
    print(f"finished {label}")


def main():
    out_path = OUT_DIR / "VideoUnmaskDIYHaiMachine_fully_covered_reveal_swap_reveal.mp4"

    env = gym.make(
        "VideoUnmaskDIYHaiMachine",
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=0,
        difficulty="easy",
        cover_start_step=-1,
        cover_duration_steps=1,
        cover_settle_steps=0,
    )

    frames = []
    obs, _ = env.reset()
    frames.append(_frame(obs, "start: fully covered"))

    planner = FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    _patch_planner(planner)

    phase = {"label": "sequence"}
    original_step = env.step

    def recording_step(action):
        result = original_step(action)
        frames.append(_frame(result[0], phase["label"]))
        return result

    env.step = recording_step

    before_swap_sequence = [
        ("before swap: reveal cover 0 and put back", env.unwrapped.bin_0, [0.00, 0.12]),
        ("before swap: reveal cover 1 and put back", env.unwrapped.bin_1, [-0.10, 0.00]),
        ("before swap: reveal cover 2 and put back", env.unwrapped.bin_2, [0.00, 0.12]),
    ]

    for label, obj, offset in before_swap_sequence:
        phase["label"] = label
        pick_show_put_back(env, planner, obj, offset, label)

    phase["label"] = "scripted swap: cover 0 <-> cover 1"
    recording_step = env.step
    env.step = original_step
    scripted_swap_bins(
        env,
        frames,
        env.unwrapped.bin_0,
        env.unwrapped.bin_1,
        "scripted swap: cover 0 <-> cover 1",
    )
    env.step = recording_step

    after_swap_sequence = [
        ("after swap: reveal cover 0 and put back", env.unwrapped.bin_0, [0.00, 0.12]),
        ("after swap: reveal cover 1 and put back", env.unwrapped.bin_1, [-0.10, 0.00]),
        ("after swap: reveal cover 2 and put back", env.unwrapped.bin_2, [0.00, 0.12]),
    ]

    for label, obj, offset in after_swap_sequence:
        phase["label"] = label
        pick_show_put_back(env, planner, obj, offset, label)

    phase["label"] = "done: all covers restored"
    _wait(env, 30, gripper=1.0)

    env.close()
    imageio.mimsave(out_path, frames, fps=30, quality=8)
    print(out_path)
    print("frames", len(frames))


if __name__ == "__main__":
    main()
