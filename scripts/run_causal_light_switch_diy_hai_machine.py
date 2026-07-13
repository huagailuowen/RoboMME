"""Record a physical red/blue causal light-switch video demo."""

from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import gymnasium as gym
import numpy as np
import torch

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    ScrewPlanFailure,
)
from robomme.robomme_env.utils.subgoal_planner_func import solve_button


OUTPUT_DIR = Path(
    "/home/yininghong/chenyuan/TTT-physics/repos/robomme_benchmark/outputs/"
    "causal_light_switch_diy_hai_machine_2026-07-12"
)
VIDEO_PATH = OUTPUT_DIR / "blue_controls_lamp_press_blue_blue_red_red.mp4"
METADATA_PATH = OUTPUT_DIR / "blue_controls_lamp_press_blue_blue_red_red.json"
FPS = 30


def _to_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb(obs):
    image = _to_np(obs["sensor_data"]["base_camera"]["rgb"])
    if image.ndim == 4:
        image = image[0]
    return image.astype(np.uint8)


def _hold_action(env, gripper=1.0):
    qpos = _to_np(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
    return np.concatenate([qpos[:7], [float(gripper)]]).astype(np.float32)


def _patch_planner(planner):
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def move_screw_then_rrt(*args, **kwargs):
        for _ in range(3):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if not (isinstance(result, int) and result == -1):
                return result
        for _ in range(3):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if not (isinstance(result, int) and result == -1):
                return result
        return -1

    planner.move_to_pose_with_screw = move_screw_then_rrt


def _encode_video(path: Path, frames: list[np.ndarray]):
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "CausalLightSwitchDIYHaiMachine",
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        control_button_color="blue",
    )
    frames = []
    phase = {"label": "Initial state"}
    try:
        obs, _ = env.reset(seed=20260712)
        planner = FailAwarePandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        _patch_planner(planner)

        def record(current_obs):
            image = _rgb(current_obs).copy()
            lamp_text = "LAMP: ON" if env.unwrapped.lamp_on else "LAMP: OFF"
            lamp_color = (30, 240, 255) if env.unwrapped.lamp_on else (210, 210, 210)
            cv2.rectangle(image, (8, 8), (376, 62), (20, 20, 20), -1)
            cv2.putText(
                image,
                phase["label"],
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                lamp_text,
                (18, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                lamp_color,
                2,
                cv2.LINE_AA,
            )
            frames.append(image)

        original_step = env.step

        def recording_step(action):
            result = original_step(action)
            record(result[0])
            return result

        env.step = recording_step
        record(obs)
        for _ in range(30):
            env.step(_hold_action(env))

        sequence = ["blue", "blue", "red", "red"]
        for index, color in enumerate(sequence, start=1):
            phase["label"] = f"Action {index}/4: press {color.upper()}"
            button = env.unwrapped.buttons[color]
            before = len(env.unwrapped.press_history)
            solve_button(env, planner, button)
            for _ in range(18):
                env.step(_hold_action(env, gripper=-1.0))
            after = len(env.unwrapped.press_history)
            if after != before + 1:
                raise RuntimeError(
                    f"Expected one {color} press event, observed {after - before}"
                )

        phase["label"] = "Finished: blue, blue, red, red"
        for _ in range(45):
            env.step(_hold_action(env, gripper=-1.0))

        history = list(env.unwrapped.press_history)
        observed_colors = [event["button_color"] for event in history]
        observed_lamp_states = [event["lamp_after"] for event in history]
        if observed_colors != ["blue", "blue", "red", "red"]:
            raise RuntimeError(f"Unexpected press history: {observed_colors}")
        if observed_lamp_states != [True, False, False, False]:
            raise RuntimeError(f"Unexpected lamp history: {observed_lamp_states}")

        _encode_video(VIDEO_PATH, frames)
        metadata = {
            "env_id": "CausalLightSwitchDIYHaiMachine",
            "control_button_color": "blue",
            "press_sequence": ["blue", "blue", "red", "red"],
            "expected_lamp_after_each_press": [True, False, False, False],
            "events": history,
            "fps": FPS,
            "frames": len(frames),
            "video": str(VIDEO_PATH),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"video={VIDEO_PATH}", flush=True)
        print(f"metadata={METADATA_PATH}", flush=True)
        print(f"events={history}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
