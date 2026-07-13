"""Reconstruct RoboMME occlusion scenes using generation metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch

from robomme.robomme_env import *  # noqa: F401,F403


def _np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _hold_action(env) -> np.ndarray:
    qpos = _np(env.unwrapped.agent.robot.get_qpos())
    return np.concatenate([qpos[:7], [1.0]]).astype(np.float32)


def _robot_state(env) -> np.ndarray:
    qpos = _np(env.unwrapped.agent.robot.get_qpos())
    gripper = float(np.mean(qpos[7:])) if qpos.size > 7 else 1.0
    return np.concatenate([qpos[:7], [gripper]]).astype(np.float32)


def _eef_state(env) -> np.ndarray:
    pose = env.unwrapped.agent.tcp.pose
    return np.concatenate([_np(pose.p)[:3], _np(pose.q)[:4]]).astype(np.float32)


def _quat_error_rad(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    actual = actual / max(float(np.linalg.norm(actual)), 1e-12)
    expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
    dot = float(np.clip(abs(np.dot(actual, expected)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _camera_rgb(obs: dict[str, Any]) -> np.ndarray:
    rgb = _np(obs["sensor_data"]["base_camera"]["rgb"])
    shape = obs["sensor_data"]["base_camera"]["rgb"].shape
    rgb = rgb.reshape(shape)
    if rgb.ndim == 4:
        rgb = rgb[0]
    return rgb.astype(np.uint8)


def _select_episodes(episodes: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = []
    for missing in (True, False):
        match = next((ep for ep in episodes if bool(ep["missing_enabled"]) == missing), None)
        if match is not None and match not in selected:
            selected.append(match)
        if len(selected) >= count:
            return selected
    for episode in episodes:
        if episode not in selected:
            selected.append(episode)
        if len(selected) >= count:
            break
    return selected


def reconstruct_episode(episode: dict[str, Any], output_dir: Path, tolerance: float):
    env_kwargs = dict(episode["env_kwargs"])
    env_id = env_kwargs.pop("env_id", "VideoUnmaskDIYHaiMachine")
    env = gym.make(env_id, **env_kwargs)
    try:
        obs, _ = env.reset(seed=int(episode["seed"]))
        for _ in range(int(episode["cover_animation_steps"])):
            obs, *_ = env.step(_hold_action(env))

        expected_scene = episode["initial_scene_after_covering"]
        object_results = []
        for item in expected_scene["bins"]:
            actor = getattr(env.unwrapped, f"bin_{item['index']}")
            actual = np.concatenate([_np(actor.pose.p)[:3], _np(actor.pose.q)[:4]])
            expected = np.asarray(item["pose"], dtype=np.float64)
            object_results.append(
                {
                    "name": f"bin_{item['index']}",
                    "present": True,
                    "position_error_m": float(np.linalg.norm(actual[:3] - expected[:3])),
                    "orientation_error_rad": _quat_error_rad(actual[3:], expected[3:]),
                }
            )

        logical_presence = set(env.unwrapped.diy_present_cube_indices)
        for item in expected_scene["cubes"]:
            idx = int(item["index"])
            result = {
                "name": f"cube_{idx}",
                "present": idx in logical_presence,
                "expected_present": bool(item["present"]),
            }
            if item["present"]:
                actor = getattr(env.unwrapped, f"target_cube_{idx}")
                actual = np.concatenate([_np(actor.pose.p)[:3], _np(actor.pose.q)[:4]])
                expected = np.asarray(item["pose"], dtype=np.float64)
                result["position_error_m"] = float(np.linalg.norm(actual[:3] - expected[:3]))
                result["orientation_error_rad"] = _quat_error_rad(actual[3:], expected[3:])
            object_results.append(result)

        robot_error = float(
            np.max(
                np.abs(
                    _robot_state(env)
                    - np.asarray(episode["initial_robot_state_after_covering"], dtype=np.float32)
                )
            )
        )
        eef_error = float(
            np.max(
                np.abs(
                    _eef_state(env)
                    - np.asarray(episode["initial_eef_state_after_covering"], dtype=np.float32)
                )
            )
        )
        pose_errors = [item.get("position_error_m", 0.0) for item in object_results]
        orientation_errors = [item.get("orientation_error_rad", 0.0) for item in object_results]
        presence_ok = all(
            item.get("present") == item.get("expected_present", item.get("present"))
            for item in object_results
        )
        passed = bool(
            presence_ok
            and max(pose_errors, default=0.0) <= tolerance
            and max(orientation_errors, default=0.0) <= tolerance
            and robot_error <= tolerance
            and eef_error <= tolerance
        )

        image_path = output_dir / f"episode_{episode['episode_index']:03d}_reconstructed.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(_camera_rgb(obs), cv2.COLOR_RGB2BGR))
        return {
            "episode_index": episode["episode_index"],
            "case_type": episode["case_type"],
            "seed": episode["seed"],
            "missing_cube_index": episode["missing_cube_index"],
            "passed": passed,
            "tolerance": tolerance,
            "max_position_error_m": max(pose_errors, default=0.0),
            "max_orientation_error_rad": max(orientation_errors, default=0.0),
            "max_robot_state_error": robot_error,
            "max_eef_state_error": eef_error,
            "presence_ok": presence_ok,
            "objects": object_results,
            "reconstructed_image": str(image_path),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.dataset_root / "robomme_occlusion_generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.dataset_root.parent / "reconstruction_validation" / metadata["case_type"]
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_episodes(metadata["episodes"], max(1, args.sample_count))
    reports = [
        reconstruct_episode(dict(episode), output_dir, float(args.tolerance))
        for episode in selected
    ]
    result = {
        "metadata_source": str(metadata_path),
        "metadata_only_reconstruction": True,
        "case_type": metadata["case_type"],
        "passed": all(report["passed"] for report in reports),
        "episodes": reports,
    }
    report_path = output_dir / "metadata_reconstruction_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
