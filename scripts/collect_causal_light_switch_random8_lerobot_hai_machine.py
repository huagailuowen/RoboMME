"""Collect 100 causal light-switch episodes with eight random button presses."""

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

import collect_causal_light_switch_lerobot_hai_machine as base


DEFAULT_DATASET_NAME = "robomme_light_switch_random8_100eps_hai-machine_lerobot"
DEFAULT_EPISODES = 100
DEFAULT_PRESSES = 8
DEFAULT_SEED = 20260713


def _sample_cases(
    count: int,
    press_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if count % 2 != 0:
        raise ValueError("episode count must be even for exact controller balance")
    if press_count < 2:
        raise ValueError("press count must be at least two")

    controllers = ["red"] * (count // 2) + ["blue"] * (count // 2)
    rng.shuffle(controllers)
    cases = []
    for control_color in controllers:
        while True:
            sequence = [str(color) for color in rng.choice(base.COLORS, press_count)]
            if len(set(sequence)) == 2:
                break
        cases.append(
            {
                "control_color": control_color,
                "press_sequence": sequence,
            }
        )
    return cases


def _collect_episode(
    env,
    planner,
    dataset,
    episode_index: int,
    seed: int,
    case: dict[str, Any],
    positions: dict[str, list[float]],
    no_pause: bool = False,
    action_transform=None,
) -> dict[str, Any]:
    if "control_colors" in case:
        control_colors = tuple(sorted(str(color) for color in case["control_colors"]))
    else:
        control_colors = (str(case["control_color"]),)
    sequence = [str(color) for color in case["press_sequence"]]
    initial_lamp_on = bool(case.get("initial_lamp_on", False))
    lamp_on = initial_lamp_on
    expected_lamp = []
    for color in sequence:
        if color in control_colors:
            lamp_on = not lamp_on
        expected_lamp.append(lamp_on)

    env.unwrapped.configure_episode(
        red_button_xy=positions["red"],
        blue_button_xy=positions["blue"],
        control_button_colors=control_colors,
        initial_lamp_on=initial_lamp_on,
    )
    obs, _ = env.reset(seed=seed)
    recorder = base.EpisodeRecorder(
        dataset,
        env,
        episode_index,
        action_transform=action_transform,
    )
    original_step = env.step
    original_close_gripper = planner.close_gripper

    if no_pause:
        planner.gripper_state = planner.CLOSED

        def set_closed_without_steps(*args, gripper_state=None, **kwargs):
            planner.gripper_state = (
                planner.CLOSED if gripper_state is None else gripper_state
            )
            return None

        planner.close_gripper = set_closed_without_steps

    def recording_step(action):
        result = original_step(action)
        recorder.add(result[0], action)
        return result

    env.step = recording_step
    try:
        recorder.add(
            obs,
            base._hold_action(env, gripper=-1.0 if no_pause else 1.0),
        )
        if not no_pause:
            base._wait(env, 30, gripper=1.0)
        for press_index, color in enumerate(sequence):
            recorder.set_phase(f"random_press_{press_index + 1}_{color}")
            before = len(env.unwrapped.press_history)
            result = base.solve_button(env, planner, env.unwrapped.buttons[color])
            if isinstance(result, int) and result == -1:
                raise RuntimeError(f"planner failed on press {press_index + 1}")
            if not no_pause:
                base._wait(env, 18, gripper=-1.0)
            after = len(env.unwrapped.press_history)
            if after != before + 1:
                raise RuntimeError(
                    f"press {press_index + 1} expected one event, got {after - before}"
                )
        recorder.set_phase("final_observation")
        if not no_pause:
            base._wait(env, 35, gripper=-1.0)
        recorder.finish()
    finally:
        env.step = original_step
        planner.close_gripper = original_close_gripper

    history = list(env.unwrapped.press_history)
    observed_colors = [event["button_color"] for event in history]
    observed_lamp = [bool(event["lamp_after"]) for event in history]
    if observed_colors != sequence:
        raise RuntimeError(f"event sequence mismatch: {observed_colors} != {sequence}")
    if observed_lamp != expected_lamp:
        raise RuntimeError(f"lamp sequence mismatch: {observed_lamp} != {expected_lamp}")
    if bool(env.unwrapped.lamp_on) != expected_lamp[-1]:
        raise RuntimeError("final lamp state does not match the causal event history")

    dataset.save_episode()
    return {
        "episode_index": episode_index,
        "seed": seed,
        "control_button_color": control_colors[0] if len(control_colors) == 1 else None,
        "control_button_colors": list(control_colors),
        "red_controls_lamp": "red" in control_colors,
        "blue_controls_lamp": "blue" in control_colors,
        "initial_lamp_on": initial_lamp_on,
        "press_sequence": sequence,
        "press_color_counts": dict(Counter(sequence)),
        "expected_lamp_after_each_press": expected_lamp,
        "final_lamp_on": expected_lamp[-1],
        "button_positions_xy_m": positions,
        "button_separation_m": float(
            np.linalg.norm(np.asarray(positions["red"]) - np.asarray(positions["blue"]))
        ),
        "events": history,
        "frame_count": recorder.frame_count,
        "phase_ranges": recorder.phase_ranges,
        "no_pause_mode": bool(no_pause),
    }


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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.output_root / args.dataset_name
    metadata_path = dataset_root / "robomme_light_switch_random8_metadata.json"
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cases = _sample_cases(args.episodes, args.presses, rng)
    layouts = [base._sample_button_positions(rng) for _ in cases]
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
            "image_size": list(base.IMAGE_SIZE),
            "fps": base.FPS,
        },
        "randomization": {
            "base_seed": args.seed,
            "control_button_color": "exactly balanced 50/50",
            "presses": (
                "independent red/blue p=0.5 draws, conditioned on both colors "
                "appearing at least once per episode"
            ),
            "button_x_range_m": list(base.X_RANGE),
            "button_y_range_m": list(base.Y_RANGE),
            "minimum_button_separation_m": base.MIN_SEPARATION_M,
            "color_assignment_to_sampled_points": "random 50/50",
        },
        "design": {
            "episodes": args.episodes,
            "presses_per_episode": args.presses,
            "controller_balance": args.episodes // 2,
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

    base._patch_lerobot_video_crf()
    dataset = base._create_dataset(dataset_root, repo_id)
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
                f"control={case['control_color']} "
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
    validation = base._validate_dataset(dataset_root, repo_id, args.episodes)
    controller_counts = Counter(
        episode["control_button_color"] for episode in metadata["episodes"]
    )
    expected_controller_count = args.episodes // 2
    if controller_counts != Counter(
        {"red": expected_controller_count, "blue": expected_controller_count}
    ):
        raise RuntimeError(f"controller distribution is imbalanced: {controller_counts}")

    all_presses = [
        color
        for episode in metadata["episodes"]
        for color in episode["press_sequence"]
    ]
    per_press_position_counts = []
    for press_index in range(args.presses):
        counts = Counter(
            episode["press_sequence"][press_index]
            for episode in metadata["episodes"]
        )
        per_press_position_counts.append(
            {"press_index": press_index, "red": counts["red"], "blue": counts["blue"]}
        )
    metadata["controller_counts"] = dict(controller_counts)
    metadata["press_color_counts"] = dict(Counter(all_presses))
    metadata["per_press_position_counts"] = per_press_position_counts
    metadata["validation"] = validation
    metadata["status"] = "completed"
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    _write_metadata(metadata_path, metadata)
    print(f"dataset={dataset_root}", flush=True)
    print(f"controller_counts={dict(controller_counts)}", flush=True)
    print(f"press_color_counts={dict(Counter(all_presses))}", flush=True)
    print(f"validation={validation}", flush=True)


if __name__ == "__main__":
    main()
