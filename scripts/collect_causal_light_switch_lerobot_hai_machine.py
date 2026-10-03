"""Collect a balanced 100-episode causal light-switch LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
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
from robomme.robomme_env.utils.subgoal_planner_func import solve_button


DEFAULT_OUTPUT_ROOT = Path(
    "/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT/data/"
    "robomme-lightSwitch"
)
DEFAULT_DATASET_NAME = "robomme_light_switch_100eps_hai-machine_lerobot"
COLORS = ("red", "blue")
IMAGE_SIZE = (224, 224)
FPS = 30
VIDEO_CRF = 18
JPEG_QUALITY = 98
X_RANGE = (-0.17, -0.06)
Y_RANGE = (-0.14, 0.14)
MIN_SEPARATION_M = 0.11
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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rgb(obs: dict[str, Any], camera: str) -> np.ndarray:
    image = _to_np(obs["sensor_data"][camera]["rgb"])
    if image.ndim == 5:
        image = image[0, 0]
    elif image.ndim == 4:
        image = image[0]
    image = image.astype(np.uint8)
    if image.shape[:2] != IMAGE_SIZE:
        image = cv2.resize(
            image,
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _robot_state(env) -> np.ndarray:
    qpos = _vec(env.unwrapped.agent.robot.get_qpos())
    gripper = float(np.mean(qpos[7:])) if qpos.size > 7 else 1.0
    return np.concatenate([qpos[:7], [gripper]]).astype(np.float32)


def _eef_state(env) -> np.ndarray:
    pose = env.unwrapped.agent.tcp.pose
    return np.concatenate([_vec(pose.p)[:3], _vec(pose.q)[:4]]).astype(np.float32)


def _hold_action(env, gripper: float = 1.0) -> np.ndarray:
    qpos = _vec(env.unwrapped.agent.robot.get_qpos())
    return np.concatenate([qpos[:7], [float(gripper)]]).astype(np.float32)


def _normalize_action(action, env) -> np.ndarray:
    if action is None:
        return _hold_action(env)
    action = _vec(action)
    if action.size >= 8:
        return action[:8]
    return np.pad(action, (0, 8 - action.size)).astype(np.float32)


def _button_lamp_state(env) -> np.ndarray:
    env_u = env.unwrapped
    red_p = _vec(env_u.buttons["red"].pose.p)[:3]
    blue_p = _vec(env_u.buttons["blue"].pose.p)[:3]
    return np.concatenate(
        [
            red_p,
            blue_p,
            [
                env_u._button_depth("red"),
                env_u._button_depth("blue"),
                float(env_u.lamp_on),
            ],
        ]
    ).astype(np.float32)


def _patch_lerobot_video_crf() -> None:
    global _VIDEO_ENCODER_PATCHED
    if _VIDEO_ENCODER_PATCHED:
        return
    original = lerobot_dataset_module.encode_video_frames

    def encode_with_crf(*args, **kwargs):
        kwargs["crf"] = VIDEO_CRF
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


def _patch_planner(planner) -> None:
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


def _sample_button_positions(rng: np.random.Generator) -> dict[str, list[float]]:
    for _ in range(10000):
        point_a = np.asarray(
            [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)], dtype=np.float32
        )
        point_b = np.asarray(
            [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)], dtype=np.float32
        )
        if float(np.linalg.norm(point_a - point_b)) < MIN_SEPARATION_M:
            continue
        if rng.random() < 0.5:
            red, blue = point_a, point_b
        else:
            red, blue = point_b, point_a
        return {"red": red.tolist(), "blue": blue.tolist()}
    raise RuntimeError("Could not sample a valid separated button layout")


def _balanced_cases(count: int, rng: np.random.Generator) -> list[dict[str, str]]:
    if count % 4 != 0:
        raise ValueError("episode count must be divisible by four for exact balance")
    repeats = count // 4
    cases = [
        {"control_color": control, "first_color": first}
        for control in COLORS
        for first in COLORS
        for _ in range(repeats)
    ]
    rng.shuffle(cases)
    return cases


def _expected_lamp_states(sequence: list[str], control_color: str) -> list[bool]:
    lamp_on = False
    states = []
    for color in sequence:
        if color == control_color:
            lamp_on = not lamp_on
        states.append(lamp_on)
    return states


def _create_dataset(
    root: Path,
    repo_id: str,
    *,
    action_dim: int = 8,
    action_names: list[str] | None = None,
) -> LeRobotDataset:
    if action_names is None:
        action_names = [f"action_{index}" for index in range(action_dim)]
    if len(action_names) != action_dim:
        raise ValueError("action_names length must equal action_dim")
    image_feature = {
        "dtype": "video",
        "shape": (3, IMAGE_SIZE[0], IMAGE_SIZE[1]),
        "names": ["channel", "height", "width"],
    }
    features = {
        "observation.images.image": dict(image_feature),
        "observation.images.wrist_image": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"state_{index}" for index in range(8)],
        },
        "observation.eef_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "observation.button_lamp_state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "red_x",
                "red_y",
                "red_z",
                "blue_x",
                "blue_y",
                "blue_z",
                "red_depth",
                "blue_depth",
                "lamp_on",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=FPS,
        robot_type="panda",
        features=features,
        use_videos=True,
        tolerance_s=1e-4,
        image_writer_processes=0,
        image_writer_threads=0,
        video_backend="pyav",
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )


class EpisodeRecorder:
    def __init__(
        self,
        dataset: LeRobotDataset,
        env,
        episode_index: int,
        action_transform=None,
    ):
        self.dataset = dataset
        self.env = env
        self.episode_index = episode_index
        self.frame_count = 0
        self.phase = "initial_observation"
        self.phase_start = 0
        self.phase_ranges: list[dict[str, Any]] = []
        self.action_transform = action_transform

    def set_phase(self, phase: str) -> None:
        if phase == self.phase:
            return
        self._close_phase()
        self.phase = phase
        self.phase_start = self.frame_count

    def _close_phase(self) -> None:
        if self.frame_count > self.phase_start:
            self.phase_ranges.append(
                {
                    "label": self.phase,
                    "start_frame": self.phase_start,
                    "end_frame_exclusive": self.frame_count,
                }
            )

    def add(self, obs: dict[str, Any], action=None) -> None:
        base_image = _rgb(obs, "base_camera")
        wrist_image = _rgb(obs, "hand_camera")
        action_value = _normalize_action(action, self.env)
        if self.action_transform is not None:
            action_value = self.action_transform(action_value)
        frame = {
            "observation.images.image": base_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": _robot_state(self.env),
            "observation.eef_state": _eef_state(self.env),
            "observation.button_lamp_state": _button_lamp_state(self.env),
            "action": action_value,
        }
        task = [
            "infer which colored buttons control the lamp",
            self.phase,
            "successful scripted physical-button rollout",
            "success",
        ]
        self.dataset.add_frame(frame, task=task)
        _write_episode_image(
            self.dataset,
            "observation.images.image",
            self.frame_count,
            base_image,
        )
        _write_episode_image(
            self.dataset,
            "observation.images.wrist_image",
            self.frame_count,
            wrist_image,
        )
        self.frame_count += 1

    def finish(self) -> None:
        self._close_phase()


def _wait(env, steps: int, gripper: float = -1.0) -> None:
    for _ in range(int(steps)):
        env.step(_hold_action(env, gripper=gripper))


def _collect_episode(
    env,
    planner,
    dataset: LeRobotDataset,
    episode_index: int,
    seed: int,
    case: dict[str, str],
    positions: dict[str, list[float]],
) -> dict[str, Any]:
    control_color = case["control_color"]
    first_color = case["first_color"]
    second_color = "blue" if first_color == "red" else "red"
    sequence = [first_color, first_color, second_color, second_color]
    expected_lamp = _expected_lamp_states(sequence, control_color)

    env.unwrapped.configure_episode(
        red_button_xy=positions["red"],
        blue_button_xy=positions["blue"],
        control_button_color=control_color,
    )
    obs, _ = env.reset(seed=seed)
    recorder = EpisodeRecorder(dataset, env, episode_index)
    original_step = env.step

    def recording_step(action):
        result = original_step(action)
        recorder.add(result[0], action)
        return result

    env.step = recording_step
    try:
        recorder.add(obs, _hold_action(env))
        _wait(env, 30, gripper=1.0)
        for press_index, color in enumerate(sequence):
            recorder.set_phase(f"press_{press_index + 1}_{color}")
            before = len(env.unwrapped.press_history)
            result = solve_button(env, planner, env.unwrapped.buttons[color])
            if isinstance(result, int) and result == -1:
                raise RuntimeError(f"planner failed on press {press_index + 1}")
            _wait(env, 18, gripper=-1.0)
            after = len(env.unwrapped.press_history)
            if after != before + 1:
                raise RuntimeError(
                    f"press {press_index + 1} expected one event, got {after - before}"
                )
        recorder.set_phase("final_observation")
        _wait(env, 35, gripper=-1.0)
        recorder.finish()
    finally:
        env.step = original_step

    history = list(env.unwrapped.press_history)
    observed_colors = [event["button_color"] for event in history]
    observed_lamp = [bool(event["lamp_after"]) for event in history]
    if observed_colors != sequence:
        raise RuntimeError(f"event sequence mismatch: {observed_colors} != {sequence}")
    if observed_lamp != expected_lamp:
        raise RuntimeError(f"lamp sequence mismatch: {observed_lamp} != {expected_lamp}")
    if env.unwrapped.lamp_on:
        raise RuntimeError("lamp must finish off after two controlling-button presses")

    dataset.save_episode()
    return {
        "episode_index": episode_index,
        "seed": seed,
        "control_button_color": control_color,
        "first_pressed_color": first_color,
        "press_sequence": sequence,
        "expected_lamp_after_each_press": expected_lamp,
        "button_positions_xy_m": positions,
        "button_separation_m": float(
            np.linalg.norm(np.asarray(positions["red"]) - np.asarray(positions["blue"]))
        ),
        "events": history,
        "frame_count": recorder.frame_count,
        "phase_ranges": recorder.phase_ranges,
    }


def _validate_dataset(
    root: Path,
    repo_id: str,
    expected_episodes: int,
    expected_action_dim: int = 8,
) -> dict[str, Any]:
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        tolerance_s=1e-4,
        video_backend="pyav",
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )
    if dataset.num_episodes != expected_episodes:
        raise RuntimeError(
            f"old loader found {dataset.num_episodes} episodes, expected {expected_episodes}"
        )
    if len(dataset) <= 0:
        raise RuntimeError("old loader found no frames")
    checked_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in checked_indices:
        frame = dataset[index]
        for image_key in (
            "observation.images.image",
            "observation.images.wrist_image",
        ):
            if tuple(frame[image_key].shape) != (3, 224, 224):
                raise RuntimeError(
                    f"bad {image_key} shape at frame {index}: {frame[image_key].shape}"
                )
        if tuple(frame["action"].shape) != (expected_action_dim,):
            raise RuntimeError(f"bad action shape at frame {index}")
        if tuple(frame["observation.button_lamp_state"].shape) != (9,):
            raise RuntimeError(f"bad causal-state shape at frame {index}")
    return {
        "loader": "FastWAM-TTT vendored LeRobot v2.1",
        "tolerance_s": 1e-4,
        "num_episodes": int(dataset.num_episodes),
        "num_frames": int(len(dataset)),
        "checked_frame_indices": checked_indices,
        "status": "passed",
    }


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.output_root / args.dataset_name
    metadata_path = dataset_root / "robomme_light_switch_generation_metadata.json"
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cases = _balanced_cases(args.episodes, rng)
    layouts = [_sample_button_positions(rng) for _ in cases]
    repo_id = f"local/{args.dataset_name}"
    metadata = {
        "status": "running",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "format": "LeRobot v2.1 per-episode",
        "env": {
            "env_id": "CausalLightSwitchDIYHaiMachine",
            "obs_mode": "rgb+depth+segmentation",
            "control_mode": "pd_joint_pos",
            "robot_uids": "panda_wristcam",
            "render_mode": "rgb_array",
            "camera_names": ["base_camera", "hand_camera"],
            "image_size": list(IMAGE_SIZE),
            "fps": FPS,
        },
        "randomization": {
            "base_seed": args.seed,
            "x_range_m": list(X_RANGE),
            "y_range_m": list(Y_RANGE),
            "minimum_button_separation_m": MIN_SEPARATION_M,
            "color_assignment_to_sampled_points": "random 50/50",
        },
        "design": {
            "episodes": args.episodes,
            "balance": "exact across control_color x first_pressed_color",
            "episodes_per_combination": args.episodes // 4,
            "press_rule": "first color twice, then the other color twice",
            "hidden_cause_leakage": "control color is stored only in episode metadata",
        },
        "per_frame_causal_state_order": [
            "red_x_m",
            "red_y_m",
            "red_z_m",
            "blue_x_m",
            "blue_y_m",
            "blue_z_m",
            "red_button_depth_m",
            "blue_button_depth_m",
            "lamp_on",
        ],
        "episodes": [],
    }
    _patch_lerobot_video_crf()
    dataset = _create_dataset(dataset_root, repo_id)
    _write_metadata(metadata_path, metadata)
    env = gym.make(
        "CausalLightSwitchDIYHaiMachine",
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    try:
        for episode_index, (case, positions) in enumerate(zip(cases, layouts)):
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
            _patch_planner(planner)
            episode = _collect_episode(
                env=env,
                planner=planner,
                dataset=dataset,
                episode_index=episode_index,
                seed=args.seed + episode_index,
                case=case,
                positions=positions,
            )
            metadata["episodes"].append(episode)
            _write_metadata(metadata_path, metadata)
            print(
                f"episode={episode_index + 1}/{args.episodes} "
                f"control={case['control_color']} first={case['first_color']} "
                f"frames={episode['frame_count']}",
                flush=True,
            )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        _write_metadata(metadata_path, metadata)
        raise
    finally:
        env.close()

    del dataset
    validation = _validate_dataset(dataset_root, repo_id, args.episodes)
    combination_counts = Counter(
        (episode["control_button_color"], episode["first_pressed_color"])
        for episode in metadata["episodes"]
    )
    expected_per_combination = args.episodes // 4
    if set(combination_counts.values()) != {expected_per_combination}:
        raise RuntimeError(f"imbalanced completed dataset: {combination_counts}")
    metadata["condition_counts"] = {
        f"control_{control}__first_{first}": count
        for (control, first), count in sorted(combination_counts.items())
    }
    metadata["validation"] = validation
    metadata["status"] = "completed"
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    _write_metadata(metadata_path, metadata)
    print(f"dataset={dataset_root}", flush=True)
    print(f"validation={validation}", flush=True)


if __name__ == "__main__":
    main()
