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


def summarize_h5(path: Path) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        for traj_key in sorted(k for k in f.keys() if k.startswith("traj_")):
            traj = f[traj_key]
            tcp = np.asarray(traj["obs/extra/tcp_pose"])[:, :3]
            obj = np.asarray(traj["obs/extra/obj_start"])[:, :3]
            distances = np.linalg.norm(tcp - obj, axis=1)
            min_step = int(np.argmin(distances))

            left_distances = []
            touches = []
            held = []
            commanded_left = []
            for i in range(traj["obs/agent/qpos"].shape[0]):
                qpos = decode_json_bytes(traj["obs/agent/qpos"][i])
                grasp = decode_json_bytes(traj["obs/extra/grasp_state_pickup_obj"][i])
                action = decode_json_bytes(traj["actions/commanded_action"][i])
                if qpos is not None:
                    left_qpos = qpos["left_gripper"]
                    left_distances.append(abs(float(left_qpos[0])) + abs(float(left_qpos[1])))
                if grasp is not None:
                    touches.append(bool(grasp["left_gripper"]["touching"]))
                    held.append(bool(grasp["left_gripper"]["held"]))
                if action is not None and action.get("left_gripper") is not None:
                    commanded_left.append(float(action["left_gripper"][0]))

            rows.append(
                {
                    "file": str(path),
                    "traj": traj_key,
                    "min_tcp_obj_dist_m": float(distances[min_step]),
                    "min_dist_step": min_step,
                    "touch_any": any(touches),
                    "held_any": any(held),
                    "success_any": bool(np.asarray(traj["success"]).any()),
                    "fail_final": bool(np.asarray(traj["fail"])[-1]),
                    "obj_delta_m": float(np.linalg.norm(obj[-1] - obj[0])),
                    "left_gripper_dist_min_m": min(left_distances) if left_distances else None,
                    "left_gripper_dist_final_m": left_distances[-1] if left_distances else None,
                    "left_cmd_min": min(commanded_left) if commanded_left else None,
                    "left_cmd_max": max(commanded_left) if commanded_left else None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="H5 files or run directories to summarize")
    args = parser.parse_args()

    h5_paths: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            h5_paths.extend(sorted(path.rglob("trajectories_batch_*.h5")))
        else:
            h5_paths.append(path)

    rows = []
    for path in h5_paths:
        rows.extend(summarize_h5(path))

    rows.sort(key=lambda row: row["min_tcp_obj_dist_m"])
    print(
        "rank\ttraj\tmin_dist_m\tstep\ttouch\theld\tsuccess\tobj_delta_m\tleft_grip_min_m\tfile"
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank}\t{row['traj']}\t{row['min_tcp_obj_dist_m']:.4f}\t"
            f"{row['min_dist_step']}\t{row['touch_any']}\t{row['held_any']}\t"
            f"{row['success_any']}\t{row['obj_delta_m']:.4f}\t"
            f"{row['left_gripper_dist_min_m']:.4f}\t{row['file']}"
        )


if __name__ == "__main__":
    main()
