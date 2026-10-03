"""Collect 200 random-eight-press episodes with independent button causality."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_to_matrix,
)

import collect_causal_light_switch_lerobot_hai_machine as base
import collect_causal_light_switch_random8_lerobot_hai_machine as random8


DEFAULT_DATASET_NAME = (
    "robomme_light_switch_independent_controls_random8_200eps_hai-machine_lerobot"
)
DEFAULT_EPISODES = 200
DEFAULT_PRESSES = 8
DEFAULT_SEED = 20260714
FIXED_CLOSE_RED_XY = (-0.12, -0.055)
FIXED_CLOSE_BLUE_XY = (-0.12, 0.055)
SMALL_RANDOM_X_JITTER_M = 0.015
SMALL_RANDOM_Y_JITTER_M = 0.010
SMALL_RANDOM_MIN_SEPARATION_M = 0.090
SMALL_RANDOM_MAX_SEPARATION_M = 0.140
CAUSAL_CONFIGURATIONS = (
    ("neither", ()),
    ("red_only", ("red",)),
    ("blue_only", ("blue",)),
    ("both", ("red", "blue")),
)


def _sample_small_random_layout(
    rng: np.random.Generator,
    *,
    swap_color_sides: bool,
) -> dict[str, list[float]]:
    for _ in range(10000):
        left = np.asarray(FIXED_CLOSE_RED_XY, dtype=np.float32) + np.asarray(
            [
                rng.uniform(-SMALL_RANDOM_X_JITTER_M, SMALL_RANDOM_X_JITTER_M),
                rng.uniform(-SMALL_RANDOM_Y_JITTER_M, SMALL_RANDOM_Y_JITTER_M),
            ],
            dtype=np.float32,
        )
        right = np.asarray(FIXED_CLOSE_BLUE_XY, dtype=np.float32) + np.asarray(
            [
                rng.uniform(-SMALL_RANDOM_X_JITTER_M, SMALL_RANDOM_X_JITTER_M),
                rng.uniform(-SMALL_RANDOM_Y_JITTER_M, SMALL_RANDOM_Y_JITTER_M),
            ],
            dtype=np.float32,
        )
        separation = float(np.linalg.norm(left - right))
        if SMALL_RANDOM_MIN_SEPARATION_M <= separation <= SMALL_RANDOM_MAX_SEPARATION_M:
            red, blue = (right, left) if swap_color_sides else (left, right)
            return {"red": red.tolist(), "blue": blue.tolist()}
    raise RuntimeError("could not sample a valid small-random button layout")


def _balanced_side_swaps(
    cases: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[bool]:
    assignments = [False] * len(cases)
    causal_classes = [item[0] for item in CAUSAL_CONFIGURATIONS]
    for causal_index, causal_class in enumerate(causal_classes):
        initial_states = sorted(
            {
                bool(case["initial_lamp_on"])
                for case in cases
                if case["causal_class"] == causal_class
            }
        )
        for initial_lamp_on in initial_states:
            indices = [
                index
                for index, case in enumerate(cases)
                if case["causal_class"] == causal_class
                and bool(case["initial_lamp_on"]) == initial_lamp_on
            ]
            rng.shuffle(indices)
            swap_count = len(indices) // 2
            if len(indices) % 2:
                swap_count += (causal_index + int(initial_lamp_on)) % 2
            for index in indices[:swap_count]:
                assignments[index] = True
    if sum(assignments) * 2 != len(assignments):
        raise RuntimeError("could not construct an exactly balanced side assignment")
    return assignments


class AbsoluteEEFActionTransform:
    """Map absolute Panda joint targets to controller-native absolute TCP poses."""

    def __init__(self, env, planner):
        self.env = env
        self.model = planner.planner.pinocchio_model
        self.joint_count = len(planner.planner.user_joint_names)
        self.ee_link_index = planner.planner.link_name_2_idx[
            planner.planner.move_group
        ]
        self.previous_euler_xyz = None

    def __call__(self, joint_action) -> np.ndarray:
        joint_action = base._vec(joint_action)
        if joint_action.size < 8:
            raise ValueError(f"expected 8D joint action, got {joint_action.shape}")
        robot_qpos = base._vec(self.env.unwrapped.agent.robot.get_qpos())
        full_qpos = np.zeros(self.joint_count, dtype=np.float64)
        full_qpos[:7] = joint_action[:7]
        if self.joint_count > 7:
            tail_count = min(self.joint_count - 7, max(0, robot_qpos.size - 7))
            full_qpos[7 : 7 + tail_count] = robot_qpos[7 : 7 + tail_count]

        self.model.compute_forward_kinematics(full_qpos)
        pose = np.asarray(
            self.model.get_link_pose(self.ee_link_index),
            dtype=np.float64,
        )
        position = pose[:3]
        quaternion_wxyz = torch.as_tensor(pose[3:7], dtype=torch.float64)
        rotation_matrix = quaternion_to_matrix(quaternion_wxyz)
        euler_xyz = matrix_to_euler_angles(rotation_matrix, "XYZ").cpu().numpy()
        if self.previous_euler_xyz is not None:
            euler_xyz = euler_xyz + 2.0 * np.pi * np.round(
                (self.previous_euler_xyz - euler_xyz) / (2.0 * np.pi)
            )
        self.previous_euler_xyz = euler_xyz.copy()
        absolute_eef_action = np.concatenate(
            [position, euler_xyz, [float(joint_action[7])]]
        ).astype(np.float32)
        if not np.isfinite(absolute_eef_action).all():
            raise RuntimeError("non-finite absolute EEF action produced by FK")
        return absolute_eef_action


def _sample_cases(
    count: int,
    press_count: int,
    rng: np.random.Generator,
    initial_lamp_state: str = "off",
) -> list[dict[str, Any]]:
    if count % len(CAUSAL_CONFIGURATIONS) != 0:
        raise ValueError("episode count must be divisible by four for exact balance")
    if press_count < 2:
        raise ValueError("press count must be at least two")

    if initial_lamp_state not in {"off", "on", "random"}:
        raise ValueError("initial_lamp_state must be off, on, or random")
    if initial_lamp_state == "random":
        full_factor_count = len(CAUSAL_CONFIGURATIONS) * 2
        if count % full_factor_count != 0:
            raise ValueError(
                "episode count must be divisible by eight for causal x initial-state balance"
            )
        repeats = count // full_factor_count
        configurations = [
            (causal_class, control_colors, initial_lamp_on)
            for causal_class, control_colors in CAUSAL_CONFIGURATIONS
            for initial_lamp_on in (False, True)
            for _ in range(repeats)
        ]
    else:
        repeats = count // len(CAUSAL_CONFIGURATIONS)
        initial_lamp_on = initial_lamp_state == "on"
        configurations = [
            (causal_class, control_colors, initial_lamp_on)
            for causal_class, control_colors in CAUSAL_CONFIGURATIONS
            for _ in range(repeats)
        ]
    rng.shuffle(configurations)

    cases = []
    for causal_class, control_colors, initial_lamp_on in configurations:
        while True:
            sequence = [str(color) for color in rng.choice(base.COLORS, press_count)]
            if len(set(sequence)) == 2:
                break
        cases.append(
            {
                "causal_class": causal_class,
                "control_colors": list(control_colors),
                "initial_lamp_on": bool(initial_lamp_on),
                "press_sequence": sequence,
            }
        )
    return cases


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(base._jsonable(metadata), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--presses", type=int, default=DEFAULT_PRESSES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=base.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--fixed-close-button-layout", action="store_true")
    parser.add_argument("--small-random-button-layout", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--absolute-eef-action", action="store_true")
    parser.add_argument("--random-initial-lamp-state", action="store_true")
    parser.add_argument(
        "--initial-lamp-state",
        choices=("off", "on", "random"),
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.output_root / args.dataset_name
    metadata_path = dataset_root / "robomme_light_switch_independent_controls_metadata.json"
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    initial_lamp_state_mode = args.initial_lamp_state
    if initial_lamp_state_mode is None:
        initial_lamp_state_mode = (
            "random" if args.random_initial_lamp_state else "off"
        )
    elif args.random_initial_lamp_state and initial_lamp_state_mode != "random":
        raise ValueError(
            "--random-initial-lamp-state conflicts with a non-random "
            "--initial-lamp-state"
        )

    rng = np.random.default_rng(args.seed)
    cases = _sample_cases(
        args.episodes,
        args.presses,
        rng,
        initial_lamp_state=initial_lamp_state_mode,
    )
    if args.fixed_close_button_layout and args.small_random_button_layout:
        raise ValueError("fixed-close and small-random layouts are mutually exclusive")
    if args.fixed_close_button_layout:
        fixed_layout = {
            "red": list(FIXED_CLOSE_RED_XY),
            "blue": list(FIXED_CLOSE_BLUE_XY),
        }
        layouts = [
            {color: list(xy) for color, xy in fixed_layout.items()}
            for _ in cases
        ]
        layout_metadata = {
            "mode": "fixed_close",
            "red_button_xy_m": list(FIXED_CLOSE_RED_XY),
            "blue_button_xy_m": list(FIXED_CLOSE_BLUE_XY),
            "center_separation_m": float(
                np.linalg.norm(
                    np.asarray(FIXED_CLOSE_RED_XY)
                    - np.asarray(FIXED_CLOSE_BLUE_XY)
                )
            ),
        }
    elif args.small_random_button_layout:
        side_swaps = _balanced_side_swaps(cases, rng)
        layouts = [
            _sample_small_random_layout(rng, swap_color_sides=swap_color_sides)
            for swap_color_sides in side_swaps
        ]
        layout_metadata = {
            "mode": "small_random_around_fixed_close",
            "red_anchor_xy_m": list(FIXED_CLOSE_RED_XY),
            "blue_anchor_xy_m": list(FIXED_CLOSE_BLUE_XY),
            "independent_x_jitter_m": [
                -SMALL_RANDOM_X_JITTER_M,
                SMALL_RANDOM_X_JITTER_M,
            ],
            "independent_y_jitter_m": [
                -SMALL_RANDOM_Y_JITTER_M,
                SMALL_RANDOM_Y_JITTER_M,
            ],
            "minimum_center_separation_m": SMALL_RANDOM_MIN_SEPARATION_M,
            "maximum_center_separation_m": SMALL_RANDOM_MAX_SEPARATION_M,
            "color_side_assignment": "exactly balanced red-left/blue-right and red-right/blue-left",
        }
    else:
        layouts = [base._sample_button_positions(rng) for _ in cases]
        layout_metadata = {
            "mode": "random",
            "button_x_range_m": list(base.X_RANGE),
            "button_y_range_m": list(base.Y_RANGE),
            "minimum_button_separation_m": base.MIN_SEPARATION_M,
            "color_assignment_to_sampled_points": "random 50/50",
        }
    repo_id = f"local/{args.dataset_name}"
    episodes_per_configuration = args.episodes // 4
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
            "image_size": list(base.IMAGE_SIZE),
            "fps": base.FPS,
        },
        "causal_design": {
            "red_controls_lamp_probability": 0.5,
            "blue_controls_lamp_probability": 0.5,
            "independent_button_causality": True,
            "sampling_implementation": "exactly balanced finite dataset",
            "causal_classes": {
                "neither": episodes_per_configuration,
                "red_only": episodes_per_configuration,
                "blue_only": episodes_per_configuration,
                "both": episodes_per_configuration,
            },
            "initial_lamp_state": (
                {
                    "random": True,
                    "mode": "random",
                    "on_probability": 0.5,
                    "sampling_implementation": "exact full-factor balance",
                    "on_episodes": args.episodes // 2,
                    "off_episodes": args.episodes // 2,
                    "episodes_per_causal_class_and_initial_state": args.episodes // 8,
                }
                if initial_lamp_state_mode == "random"
                else {
                    "random": False,
                    "mode": initial_lamp_state_mode,
                    "on_probability": (
                        1.0 if initial_lamp_state_mode == "on" else 0.0
                    ),
                    "on_episodes": (
                        args.episodes if initial_lamp_state_mode == "on" else 0
                    ),
                    "off_episodes": (
                        args.episodes if initial_lamp_state_mode == "off" else 0
                    ),
                }
            ),
            "label_storage": (
                "causal labels are stored only in episode metadata, not observations "
                "or task text"
            ),
        },
        "randomization": {
            "base_seed": args.seed,
            "presses": (
                "independent red/blue p=0.5 draws, conditioned on both colors "
                "appearing at least once per episode"
            ),
        },
        "button_layout": layout_metadata,
        "design": {
            "episodes": args.episodes,
            "presses_per_episode": args.presses,
            "no_pause_mode": bool(args.no_pause),
            "explicit_wait_frames": {
                "initial": 0 if args.no_pause else 30,
                "after_each_press": 0 if args.no_pause else 18,
                "final": 0 if args.no_pause else 35,
                "repeated_gripper_close": 0 if args.no_pause else 6,
            },
        },
        "action_representation": (
            {
                "type": "absolute_eef_pose",
                "shape": [7],
                "order": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "position_unit": "meter",
                "rotation_unit": "radian",
                "rotation_convention": "XYZ Euler, temporally unwrapped by 2pi",
                "coordinate_frame": "robot_base",
                "gripper": {"closed": -1.0, "open": 1.0},
                "compatible_control_mode": "pd_ee_pose",
                "physical_execution_control_mode": "pd_joint_pos",
                "conversion": "forward kinematics of each absolute joint target",
            }
            if args.absolute_eef_action
            else {
                "type": "absolute_joint_position",
                "shape": [8],
                "order": [
                    "panda_joint1",
                    "panda_joint2",
                    "panda_joint3",
                    "panda_joint4",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                    "gripper",
                ],
                "joint_unit": "radian",
            }
        ),
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

    base._patch_lerobot_video_crf()
    action_names = (
        ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        if args.absolute_eef_action
        else None
    )
    dataset = base._create_dataset(
        dataset_root,
        repo_id,
        action_dim=7 if args.absolute_eef_action else 8,
        action_names=action_names,
    )
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
            planner = base.FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
            base._patch_planner(planner)
            action_transform = (
                AbsoluteEEFActionTransform(env, planner)
                if args.absolute_eef_action
                else None
            )
            episode = random8._collect_episode(
                env=env,
                planner=planner,
                dataset=dataset,
                episode_index=episode_index,
                seed=args.seed + episode_index,
                case=case,
                positions=positions,
                no_pause=args.no_pause,
                action_transform=action_transform,
            )
            episode["causal_class"] = case["causal_class"]
            episode["button_side_assignment"] = (
                "red_left_blue_right"
                if float(positions["red"][1]) < float(positions["blue"][1])
                else "red_right_blue_left"
            )
            metadata["episodes"].append(episode)
            _write_metadata(metadata_path, metadata)
            print(
                f"episode={episode_index + 1}/{args.episodes} "
                f"causal_class={case['causal_class']} "
                f"sequence={''.join(case['press_sequence'])} "
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
    validation = base._validate_dataset(
        dataset_root,
        repo_id,
        args.episodes,
        expected_action_dim=7 if args.absolute_eef_action else 8,
    )
    causal_counts = Counter(
        episode["causal_class"] for episode in metadata["episodes"]
    )
    expected_counts = Counter(
        {causal_class: episodes_per_configuration for causal_class, _ in CAUSAL_CONFIGURATIONS}
    )
    if causal_counts != expected_counts:
        raise RuntimeError(f"causal classes are imbalanced: {causal_counts}")

    initial_lamp_counts = Counter(
        bool(episode["initial_lamp_on"]) for episode in metadata["episodes"]
    )
    if initial_lamp_state_mode == "random":
        expected_initial_count = args.episodes // 2
        if initial_lamp_counts != Counter(
            {False: expected_initial_count, True: expected_initial_count}
        ):
            raise RuntimeError(
                f"initial lamp-state distribution is imbalanced: {initial_lamp_counts}"
            )
        causal_initial_counts = Counter(
            (episode["causal_class"], bool(episode["initial_lamp_on"]))
            for episode in metadata["episodes"]
        )
        expected_cell_count = args.episodes // 8
        if set(causal_initial_counts.values()) != {expected_cell_count}:
            raise RuntimeError(
                f"causal x initial-state cells are imbalanced: {causal_initial_counts}"
            )
    else:
        expected_initial_lamp_on = initial_lamp_state_mode == "on"
        if initial_lamp_counts != Counter(
            {expected_initial_lamp_on: args.episodes}
        ):
            raise RuntimeError(
                f"fixed initial lamp-state distribution is invalid: {initial_lamp_counts}"
            )
        causal_initial_counts = Counter(
            (episode["causal_class"], bool(episode["initial_lamp_on"]))
            for episode in metadata["episodes"]
        )

    all_presses = [
        color
        for episode in metadata["episodes"]
        for color in episode["press_sequence"]
    ]
    metadata["causal_class_counts"] = dict(causal_counts)
    metadata["initial_lamp_state_counts"] = {
        "off": initial_lamp_counts[False],
        "on": initial_lamp_counts[True],
    }
    metadata["causal_initial_state_counts"] = {
        f"{causal_class}__initial_{'on' if initial_lamp_on else 'off'}": count
        for (causal_class, initial_lamp_on), count in sorted(
            causal_initial_counts.items()
        )
    }
    metadata["press_color_counts"] = dict(Counter(all_presses))
    side_assignment_counts = Counter(
        episode["button_side_assignment"] for episode in metadata["episodes"]
    )
    if args.small_random_button_layout:
        expected_side_count = args.episodes // 2
        expected_sides = Counter(
            {
                "red_left_blue_right": expected_side_count,
                "red_right_blue_left": expected_side_count,
            }
        )
        if side_assignment_counts != expected_sides:
            raise RuntimeError(
                f"button side assignments are imbalanced: {side_assignment_counts}"
            )
        expected_per_causal_side = args.episodes // 8
        for causal_class, _ in CAUSAL_CONFIGURATIONS:
            counts = Counter(
                episode["button_side_assignment"]
                for episode in metadata["episodes"]
                if episode["causal_class"] == causal_class
            )
            if set(counts.values()) != {expected_per_causal_side}:
                raise RuntimeError(
                    f"side assignment is imbalanced for {causal_class}: {counts}"
                )
        if initial_lamp_state_mode == "random":
            expected_per_initial_side = args.episodes // 4
            for initial_lamp_on in (False, True):
                counts = Counter(
                    episode["button_side_assignment"]
                    for episode in metadata["episodes"]
                    if bool(episode["initial_lamp_on"]) == initial_lamp_on
                )
                if set(counts.values()) != {expected_per_initial_side}:
                    raise RuntimeError(
                        "side assignment is imbalanced for initial lamp state "
                        f"{initial_lamp_on}: {counts}"
                    )
    metadata["button_side_assignment_counts"] = dict(side_assignment_counts)
    metadata["validation"] = validation
    metadata["status"] = "completed"
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    _write_metadata(metadata_path, metadata)
    print(f"dataset={dataset_root}", flush=True)
    print(f"causal_class_counts={dict(causal_counts)}", flush=True)
    print(f"initial_lamp_state_counts={dict(initial_lamp_counts)}", flush=True)
    print(f"press_color_counts={dict(Counter(all_presses))}", flush=True)
    print(f"validation={validation}", flush=True)


if __name__ == "__main__":
    main()
