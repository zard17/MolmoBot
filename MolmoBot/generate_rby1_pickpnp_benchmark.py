"""
Generate a pick-and-place benchmark for RBY1 evaluation.

Creates benchmark episodes for the RBY1M mobile manipulator in iTHOR scenes,
designed for comparing default (mobile base active) vs frozen base conditions.

Objects are placed within arm reach so the frozen-base condition is feasible.

Usage:
    python generate_rby1_pickpnp_benchmark.py
    python generate_rby1_pickpnp_benchmark.py --output_path benchmarks/custom/benchmark.json --num_episodes 5
"""

import json
import argparse
from pathlib import Path


# RBY1 default init_qpos (from RBY1Config)
RBY1_INIT_QPOS = {
    "base": [0.0, 0.0, 0.0],           # x, y, theta
    "head": [0.0, 0.6],                # pan, tilt (looking down ~34 deg)
    "left_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "left_gripper": [-0.05],            # open
    "right_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "right_gripper": [-0.05],           # open
    "torso": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Robot placed at this world position in iTHOR house 321
ROBOT_BASE_POSE = [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # x, y, z, qw, qx, qy, qz

# Object positions within arm reach (~0.3-0.6m from base)
# All at table height (~1.0m), within static arm workspace
EPISODE_CONFIGS = [
    {
        "object_name": "book",
        "object_uid": "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0",
        "object_path": "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml",
        "pickup_pos": [1.7, 2.0, 1.0, 0, 0, 0, 1],     # directly in front, ~0.3m
        "goal_pos":   [1.7, 2.0, 1.2, 0, 0, 0, 1],      # lift 0.2m
        "task_desc": "Pick up the book",
    },
    {
        "object_name": "book",
        "object_uid": "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0",
        "object_path": "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml",
        "pickup_pos": [1.65, 2.15, 1.0, 0, 0, 0, 1],    # front-left, ~0.38m
        "goal_pos":   [1.65, 2.15, 1.2, 0, 0, 0, 1],
        "task_desc": "Pick up the book",
    },
    {
        "object_name": "book",
        "object_uid": "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0",
        "object_path": "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml",
        "pickup_pos": [1.65, 1.85, 1.0, 0, 0, 0, 1],    # front-right, ~0.38m
        "goal_pos":   [1.65, 1.85, 1.2, 0, 0, 0, 1],
        "task_desc": "Pick up the book",
    },
    {
        "object_name": "book",
        "object_uid": "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0",
        "object_path": "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml",
        "pickup_pos": [1.6, 2.0, 1.0, 0, 0, 0, 1],      # further front, ~0.4m
        "goal_pos":   [1.6, 2.0, 1.2, 0, 0, 0, 1],
        "task_desc": "Pick up the book",
    },
    {
        "object_name": "book",
        "object_uid": "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0",
        "object_path": "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml",
        "pickup_pos": [1.55, 2.1, 1.0, 0, 0, 0, 1],     # front-left far, ~0.46m
        "goal_pos":   [1.55, 2.1, 1.2, 0, 0, 0, 1],
        "task_desc": "Pick up the book",
    },
]


def create_rby1_episode(episode_id, config):
    """Create a single RBY1 pick-and-place benchmark episode."""
    return {
        "source": {
            "h5_file": "",
            "traj_key": "",
            "episode_length": 50,
            "camera_system_class": "RBY1GoProD455CameraSystem",
            "source_data_date": "2026-04-15",
            "benchmark_created_date": "2026-04-15",
        },
        "house_index": 321,
        "scene_dataset": "ithor",
        "data_split": "val",
        "seed": episode_id,
        "robot": {
            "robot_name": "rby1m",
            "init_qpos": RBY1_INIT_QPOS,
        },
        "img_resolution": [1024, 576],
        "cameras": [],  # RBY1GoProD455CameraSystem handles camera setup
        "scene_modifications": {
            "added_objects": {
                config["object_uid"]: config["object_path"],
            },
            "object_poses": {
                config["object_uid"]: config["pickup_pos"],
            },
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            "robot_base_pose": ROBOT_BASE_POSE,
            "pickup_obj_name": config["object_uid"],
            "pickup_obj_start_pose": config["pickup_pos"],
            "pickup_obj_goal_pose": config["goal_pos"],
            "succ_pos_threshold": 0.05,
        },
        "language": {
            "task_description": config["task_desc"],
            "referral_expressions": {
                "pickup_obj_name": config["object_name"],
            },
        },
    }


def generate_benchmark(output_path, num_episodes=5):
    """Generate RBY1 pick-and-place benchmark."""
    num_episodes = min(num_episodes, len(EPISODE_CONFIGS))
    episodes = [
        create_rby1_episode(i, EPISODE_CONFIGS[i])
        for i in range(num_episodes)
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(episodes, f, indent=2)

    print(f"Generated RBY1 pick&place benchmark with {len(episodes)} episodes at: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate RBY1 pick-and-place benchmark for freeze test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_path", type=str,
        default="benchmarks/rby1_pickpnp_benchmark/benchmark.json",
        help="Output path for the generated benchmark",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5,
        help="Number of episodes to generate (max 5)",
    )
    args = parser.parse_args()

    generate_benchmark(args.output_path, args.num_episodes)


if __name__ == "__main__":
    main()
