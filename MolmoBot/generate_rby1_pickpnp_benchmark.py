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

# Robot starts at the origin in the local custom desk scene.
ROBOT_BASE_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # x, y, z, qw, qx, qy, qz

# Robot-mounted camera specs required by JsonEvalTaskSampler.
# These mirror the RBY1 MJCF camera names and mount poses. The exocentric
# camera is recorded for debugging only; the policy still consumes the three
# trained RBY1 camera names from its policy config.
RBY1_CAMERAS = [
    {
        "name": "head_camera",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/link_head_2", "link_head_2"],
        "camera_offset": [0.05, 0.0, 0.05],
        "lookat_offset": [1.0, 0.0, 0.0],
        "camera_quaternion": [0.5, 0.5, -0.5, -0.5],
        "fov": 139.0,
        "record_depth": False,
    },
    {
        "name": "wrist_camera_l",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/link_left_arm_6", "link_left_arm_6"],
        "camera_offset": [0.0, -0.1, -0.15],
        "lookat_offset": [0.0, 0.0, -1.0],
        "camera_quaternion": [0.0, 0.0, -0.258819, 0.965926],
        "fov": 58.0,
        "record_depth": True,
    },
    {
        "name": "wrist_camera_r",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/link_right_arm_6", "link_right_arm_6"],
        "camera_offset": [0.0, 0.1, -0.15],
        "lookat_offset": [0.0, 0.0, -1.0],
        "camera_quaternion": [0.965926, -0.258819, 0.0, 0.0],
        "fov": 58.0,
        "record_depth": True,
    },
    {
        "name": "exo_camera_1",
        "type": "exocentric",
        "pos": [1.35, 1.05, 1.35],
        "forward": [-0.64, -0.64, -0.42],
        "up": [0.0, 0.0, 1.0],
        "fov": 70.0,
        "record_depth": False,
    },
]

# Object positions within arm reach (~0.3-0.6m from base).
# Salt_Shaker_1 is the Thor pickup object used by the working batch eval.
SALT_SHAKER_UID = "Salt_Shaker_1"
SALT_SHAKER_BODY_NAME = "/Salt_Shaker_1"
SALT_SHAKER_PATH = (
    "objects/thor/Kitchen Objects/SaltShaker/Prefabs/"
    "Salt_Shaker_1/Salt_Shaker_1.xml"
)

EPISODE_CONFIGS = [
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.45, 0.25, 0.79, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.45, 0.25, 0.99, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
    },
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.445, 0.28, 0.79, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.445, 0.28, 0.99, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
    },
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.465, 0.28, 0.79, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.465, 0.28, 0.99, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
    },
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.445, 0.30, 0.79, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.445, 0.30, 0.99, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
    },
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.465, 0.30, 0.79, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.465, 0.30, 0.99, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
    },
    {
        "object_name": "salt shaker",
        "object_uid": SALT_SHAKER_UID,
        "object_body_name": SALT_SHAKER_BODY_NAME,
        "object_path": SALT_SHAKER_PATH,
        "pickup_pos": [0.445, 0.28, 0.805, 0.7071068, 0.7071068, 0, 0],
        "goal_pos":   [0.445, 0.28, 1.005, 0.7071068, 0.7071068, 0, 0],
        "task_desc": "Pick up the salt shaker",
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
        "house_index": 0,
        "scene_dataset": "rby1-custom",
        "data_split": "val",
        "seed": episode_id,
        "robot": {
            "robot_name": "rby1m",
            "init_qpos": RBY1_INIT_QPOS,
        },
        "img_resolution": [1024, 576],
        "cameras": RBY1_CAMERAS,
        "scene_modifications": {
            "added_objects": {
                config["object_body_name"]: config["object_path"],
            },
            "object_poses": {
                config["object_body_name"]: config["pickup_pos"],
            },
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            "robot_base_pose": ROBOT_BASE_POSE,
            "pickup_obj_name": config["object_body_name"],
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
