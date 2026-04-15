#!/usr/bin/env python3
"""Summarize RBY1 pick eval trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decode_json_bytes(value) -> dict | None:
    raw = bytes(value.tolist()).split(b"\x00", 1)[0]
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def close_windows(close_steps: list[int]) -> list[tuple[int, int]]:
    windows = []
    if close_steps:
        window_start = close_steps[0]
        previous = close_steps[0]
        for step in close_steps[1:]:
            if step == previous + 1:
                previous = step
            else:
                windows.append((window_start, previous))
                window_start = step
                previous = step
        windows.append((window_start, previous))
    return windows


def format_windows(windows: list[tuple[int, int]]) -> str:
    return ",".join(
        f"{start}-{end}" if start != end else str(start)
        for start, end in windows
    )


def summarize_hand(traj, hand: str) -> dict:
    n_steps = traj["obs/agent/qpos"].shape[0]
    qpos_distances = [None] * n_steps
    touches = [False] * n_steps
    held = [False] * n_steps
    commands = [None] * n_steps
    close_steps = []

    for step in range(n_steps):
        qpos = decode_json_bytes(traj["obs/agent/qpos"][step])
        grasp = decode_json_bytes(traj["obs/extra/grasp_state_pickup_obj"][step])
        action = decode_json_bytes(traj["actions/commanded_action"][step])

        gripper_key = f"{hand}_gripper"
        if qpos is not None and qpos.get(gripper_key) is not None:
            gripper_qpos = qpos[gripper_key]
            qpos_distances[step] = sum(abs(float(value)) for value in gripper_qpos)
        if grasp is not None and grasp.get(gripper_key) is not None:
            touches[step] = bool(grasp[gripper_key]["touching"])
            held[step] = bool(grasp[gripper_key]["held"])
        if action is not None and action.get(gripper_key) is not None:
            command = float(action[gripper_key][0])
            commands[step] = command
            if command > 0.0:
                close_steps.append(step)

    finite_qpos = [value for value in qpos_distances if value is not None]
    finite_commands = [value for value in commands if value is not None]
    return {
        "qpos_distances": qpos_distances,
        "qpos_min": min(finite_qpos) if finite_qpos else None,
        "qpos_final": finite_qpos[-1] if finite_qpos else None,
        "touch_any": any(touches),
        "held_any": any(held),
        "touch_steps": [step for step, value in enumerate(touches) if value],
        "held_steps": [step for step, value in enumerate(held) if value],
        "commands": commands,
        "cmd_min": min(finite_commands) if finite_commands else None,
        "cmd_max": max(finite_commands) if finite_commands else None,
        "first_close_step": close_steps[0] if close_steps else None,
        "close_windows": close_windows(close_steps),
    }


def image_center_distance_at_step(traj, hand: str, camera: str, step: int) -> float | None:
    hand_base = f"obs/extra/object_image_points/{hand}_gripper/{camera}"
    obj_base = f"obs/extra/object_image_points/pickup_obj/{camera}"
    if f"{hand_base}/points" not in traj or f"{obj_base}/points" not in traj:
        return None

    hand_count = int(np.asarray(traj[f"{hand_base}/num_points"][step]).reshape(-1)[0])
    obj_count = int(np.asarray(traj[f"{obj_base}/num_points"][step]).reshape(-1)[0])
    if hand_count <= 0 or obj_count <= 0:
        return None

    hand_points = np.asarray(traj[f"{hand_base}/points"][step])[:hand_count]
    obj_points = np.asarray(traj[f"{obj_base}/points"][step])[:obj_count]
    return float(np.linalg.norm(hand_points.mean(axis=0) - obj_points.mean(axis=0)))


def tcp_distances(traj, key: str, obj: np.ndarray) -> np.ndarray | None:
    dataset = f"obs/extra/{key}"
    if dataset not in traj:
        return None
    tcp = np.asarray(traj[dataset])[:, :3]
    return np.linalg.norm(tcp - obj, axis=1)


def task_info_at(traj, step: int) -> dict:
    if "obs/extra/task_info" not in traj:
        return {}
    return decode_json_bytes(traj["obs/extra/task_info"][step]) or {}


def format_optional_float(value, precision: int = 4) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{precision}f}"


def summarize_h5(path: Path) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        for traj_key in sorted(k for k in f.keys() if k.startswith("traj_")):
            traj = f[traj_key]
            tcp = np.asarray(traj["obs/extra/tcp_pose"])[:, :3]
            obj = np.asarray(traj["obs/extra/obj_start"])[:, :3]
            distances = np.linalg.norm(tcp - obj, axis=1)
            min_step = int(np.argmin(distances))

            left = summarize_hand(traj, "left")
            right = summarize_hand(traj, "right")
            left_cmd_at_min = left["commands"][min_step]
            right_cmd_at_min = right["commands"][min_step]
            left_tcp_distances = tcp_distances(traj, "left_tcp_pose", obj)
            right_tcp_distances = tcp_distances(traj, "right_tcp_pose", obj)
            info = task_info_at(traj, min_step)

            rows.append(
                {
                    "file": str(path),
                    "traj": traj_key,
                    "min_tcp_obj_dist_m": float(distances[min_step]),
                    "min_dist_step": min_step,
                    "left_touch_any": left["touch_any"],
                    "right_touch_any": right["touch_any"],
                    "left_held_any": left["held_any"],
                    "right_held_any": right["held_any"],
                    "success_any": bool(np.asarray(traj["success"]).any()),
                    "fail_final": bool(np.asarray(traj["fail"])[-1]),
                    "obj_delta_m": float(np.linalg.norm(obj[-1] - obj[0])),
                    "left_gripper_dist_min_m": left["qpos_min"],
                    "left_gripper_dist_final_m": left["qpos_final"],
                    "right_gripper_dist_min_m": right["qpos_min"],
                    "right_gripper_dist_final_m": right["qpos_final"],
                    "left_cmd_min": left["cmd_min"],
                    "left_cmd_max": left["cmd_max"],
                    "right_cmd_min": right["cmd_min"],
                    "right_cmd_max": right["cmd_max"],
                    "left_cmd_at_min_dist": left_cmd_at_min,
                    "right_cmd_at_min_dist": right_cmd_at_min,
                    "first_left_close_step": left["first_close_step"],
                    "first_right_close_step": right["first_close_step"],
                    "left_close_windows": left["close_windows"],
                    "right_close_windows": right["close_windows"],
                    "tcp_obj_delta_at_min_m": (tcp[min_step] - obj[min_step]).tolist(),
                    "left_tcp_obj_dist_min_m": float(np.min(left_tcp_distances))
                    if left_tcp_distances is not None
                    else None,
                    "left_tcp_obj_dist_min_step": int(np.argmin(left_tcp_distances))
                    if left_tcp_distances is not None
                    else None,
                    "right_tcp_obj_dist_min_m": float(np.min(right_tcp_distances))
                    if right_tcp_distances is not None
                    else None,
                    "right_tcp_obj_dist_min_step": int(np.argmin(right_tcp_distances))
                    if right_tcp_distances is not None
                    else None,
                    "object_contact_count_at_min": info.get("object_contact_count"),
                    "robot_object_contact_count_at_min": info.get("robot_object_contact_count"),
                    "left_finger_contact_at_min": info.get("left_finger_contact"),
                    "right_finger_contact_at_min": info.get("right_finger_contact"),
                    "left_l1_body_obj_dist_at_min": info.get("left_l1_body_obj_dist"),
                    "left_l2_body_obj_dist_at_min": info.get("left_l2_body_obj_dist"),
                    "left_finger_midpoint_obj_dist_at_min": info.get(
                        "left_finger_midpoint_obj_dist"
                    ),
                    "exo_left_obj_center_dist_px_at_min": image_center_distance_at_step(
                        traj, "left", "exo_camera_1", min_step
                    ),
                    "exo_right_obj_center_dist_px_at_min": image_center_distance_at_step(
                        traj, "right", "exo_camera_1", min_step
                    ),
                }
            )
    return rows


def inspect_window(path: Path, traj_key: str, center_step: int | None, radius: int) -> None:
    with h5py.File(path, "r") as f:
        traj = f[traj_key]
        tcp = np.asarray(traj["obs/extra/tcp_pose"])[:, :3]
        obj = np.asarray(traj["obs/extra/obj_start"])[:, :3]
        distances = np.linalg.norm(tcp - obj, axis=1)
        left_tcp_distances = tcp_distances(traj, "left_tcp_pose", obj)
        right_tcp_distances = tcp_distances(traj, "right_tcp_pose", obj)
        if center_step is None:
            center_step = int(np.argmin(distances))

        start = max(0, center_step - radius)
        end = min(len(distances) - 1, center_step + radius)
        print(
            "step\ttcp_obj_m\tdelta_xyz_m\tleft_cmd\tright_cmd\tleft_qsum\t"
            "right_qsum\tleft_touch\tright_touch\tleft_held\tright_held\t"
            "left_tcp_m\tright_tcp_m\tobj_contacts\trobot_obj_contacts\t"
            "left_finger_contact\tright_finger_contact\tleft_l1_obj_m\t"
            "left_l2_obj_m\tleft_mid_obj_m\tobj_delta_m"
        )
        for step in range(start, end + 1):
            qpos = decode_json_bytes(traj["obs/agent/qpos"][step]) or {}
            grasp = decode_json_bytes(traj["obs/extra/grasp_state_pickup_obj"][step]) or {}
            action = decode_json_bytes(traj["actions/commanded_action"][step]) or {}
            info = task_info_at(traj, step)

            def qsum(hand: str) -> float | None:
                values = qpos.get(f"{hand}_gripper")
                return sum(abs(float(value)) for value in values) if values is not None else None

            def command(hand: str) -> float | None:
                values = action.get(f"{hand}_gripper")
                return float(values[0]) if values is not None else None

            def grasp_flag(hand: str, flag: str) -> bool | None:
                values = grasp.get(f"{hand}_gripper")
                return bool(values[flag]) if values is not None else None

            delta = ",".join(f"{value:.3f}" for value in (tcp[step] - obj[step]))
            obj_delta = float(np.linalg.norm(obj[step] - obj[0]))
            print(
                f"{step}\t{distances[step]:.4f}\t{delta}\t{command('left')}\t"
                f"{command('right')}\t{qsum('left')}\t{qsum('right')}\t"
                f"{grasp_flag('left', 'touching')}\t{grasp_flag('right', 'touching')}\t"
                f"{grasp_flag('left', 'held')}\t{grasp_flag('right', 'held')}\t"
                f"{left_tcp_distances[step] if left_tcp_distances is not None else None}\t"
                f"{right_tcp_distances[step] if right_tcp_distances is not None else None}\t"
                f"{info.get('object_contact_count')}\t"
                f"{info.get('robot_object_contact_count')}\t"
                f"{info.get('left_finger_contact')}\t"
                f"{info.get('right_finger_contact')}\t"
                f"{format_optional_float(info.get('left_l1_body_obj_dist'))}\t"
                f"{format_optional_float(info.get('left_l2_body_obj_dist'))}\t"
                f"{format_optional_float(info.get('left_finger_midpoint_obj_dist'))}\t"
                f"{obj_delta:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="H5 files or run directories to summarize")
    parser.add_argument("--inspect-traj", help="Print a per-step window for one trajectory")
    parser.add_argument(
        "--inspect-step",
        type=int,
        default=None,
        help="Center step for --inspect-traj. Defaults to the minimum TCP-object distance step.",
    )
    parser.add_argument("--inspect-radius", type=int, default=8)
    args = parser.parse_args()

    h5_paths: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            h5_paths.extend(sorted(path.rglob("trajectories_batch_*.h5")))
        else:
            h5_paths.append(path)

    if args.inspect_traj:
        if len(h5_paths) != 1:
            raise SystemExit("--inspect-traj requires exactly one H5 file or one run directory with one H5")
        traj_key = args.inspect_traj
        if traj_key.isdigit():
            traj_key = f"traj_{traj_key}"
        inspect_window(h5_paths[0], traj_key, args.inspect_step, args.inspect_radius)
        return

    rows = []
    for path in h5_paths:
        rows.extend(summarize_h5(path))

    rows.sort(key=lambda row: row["min_tcp_obj_dist_m"])
    print(
        "rank\ttraj\tmin_dist_m\tstep\tdelta_xyz_m\tleft_cmd_at_min\t"
        "right_cmd_at_min\tleft_first_close\tright_first_close\tleft_close_windows\t"
        "right_close_windows\tleft_touch\tright_touch\tleft_held\tright_held\t"
        "success\tobj_delta_m\tleft_grip_min_m\tright_grip_min_m\t"
        "left_tcp_min_m\tright_tcp_min_m\tobj_contacts_at_min\t"
        "robot_obj_contacts_at_min\tleft_finger_contact_at_min\t"
        "right_finger_contact_at_min\tleft_l1_obj_m_at_min\t"
        "left_l2_obj_m_at_min\tleft_mid_obj_m_at_min\t"
        "exo_left_px_at_min\texo_right_px_at_min\tfile"
    )
    for rank, row in enumerate(rows, start=1):
        delta = ",".join(f"{value:.3f}" for value in row["tcp_obj_delta_at_min_m"])
        print(
            f"{rank}\t{row['traj']}\t{row['min_tcp_obj_dist_m']:.4f}\t"
            f"{row['min_dist_step']}\t{delta}\t{row['left_cmd_at_min_dist']}\t"
            f"{row['right_cmd_at_min_dist']}\t{row['first_left_close_step']}\t"
            f"{row['first_right_close_step']}\t{format_windows(row['left_close_windows'])}\t"
            f"{format_windows(row['right_close_windows'])}\t"
            f"{row['left_touch_any']}\t{row['right_touch_any']}\t"
            f"{row['left_held_any']}\t{row['right_held_any']}\t{row['success_any']}\t"
            f"{row['obj_delta_m']:.4f}\t"
            f"{row['left_gripper_dist_min_m']:.4f}\t"
            f"{row['right_gripper_dist_min_m']:.4f}\t"
            f"{row['left_tcp_obj_dist_min_m']}\t"
            f"{row['right_tcp_obj_dist_min_m']}\t"
            f"{row['object_contact_count_at_min']}\t"
            f"{row['robot_object_contact_count_at_min']}\t"
            f"{row['left_finger_contact_at_min']}\t"
            f"{row['right_finger_contact_at_min']}\t"
            f"{format_optional_float(row['left_l1_body_obj_dist_at_min'])}\t"
            f"{format_optional_float(row['left_l2_body_obj_dist_at_min'])}\t"
            f"{format_optional_float(row['left_finger_midpoint_obj_dist_at_min'])}\t"
            f"{row['exo_left_obj_center_dist_px_at_min']:.1f}\t"
            f"{row['exo_right_obj_center_dist_px_at_min']:.1f}\t{row['file']}"
        )


if __name__ == "__main__":
    main()
