"""
Generate a minimal benchmark for MolmoBot evaluation using only local assets.

This script creates self-contained benchmark episodes that don't require external h5 file references,
using the downloaded iTHOR scenes and THOR objects.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def create_minimal_episode(
    episode_id: int,
    house_index: int = 321,
    pickup_object_name: str = "book",
    task_description: str = "Pick up the book"
) -> Dict[str, Any]:
    """Create a minimal benchmark episode with proper camera and object placement."""
    
    # Robot initial joint positions (safe home position)
    default_arm_qpos = [0, -0.7853981633974483, 0, -2.3561944901923453, 0, 1.5707963267948966, 0]
    
    # Object name for the scene
    book_obj_name = "book_1d86bd20453959c2ac45aaec57a3e0b1_1_0_0"
    
    # Create episode structure
    episode = {
        "source": {
            "h5_file": "",  # Empty string - no external h5 file reference
            "traj_key": "",
            "episode_length": 50,
            "camera_system_class": "FrankaOmniPurposeCameraSystem",
            "source_data_date": "2026-04-06",
            "benchmark_created_date": "2026-04-06"
        },
        "house_index": house_index,
        "scene_dataset": "ithor",
        "data_split": "val",
        "seed": episode_id,
        "robot": {
            "robot_name": "franka_droid",
            "init_qpos": {
                "base": [],
                "arm": default_arm_qpos,
                "gripper": [0.04, 0.04]
            }
        },
        "img_resolution": [224, 224],
        "cameras": [
            {
                "name": "exo_camera_1",
                "type": "exocentric",
                "pos": [3.5, 3.5, 1.5],
                "up": [0, 0, 1],
                "forward": [-0.707, -0.707, -0.0],
                "fov": 70.0,
                "record_depth": False
            },
            {
                "name": "wrist_camera",
                "type": "robot_mounted",
                "reference_body_names": ["robot_0/gripper/base"],
                "camera_offset": [0.1, 0.1, 0.1],
                "lookat_offset": [0.0, 0.0, 0.5],
                "camera_quaternion": [0, 0, 0, 1],
                "fov": 60.0,
                "record_depth": False
            }
        ],
        "scene_modifications": {
            "added_objects": {
                book_obj_name: "objects/thor/RoboTHOR Objects/RoboTHOR_Assets_SmallObjects/Background_SmallObjects/Book/Prefabs/RoboTHOR_book_ai2_2_v/RoboTHOR_book_ai2_2_v.xml"
            },
            "object_poses": {
                book_obj_name: [1.8, 2.2, 1.0, 0, 0, 0, 1]
            }
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            "robot_base_pose": [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": book_obj_name,
            "pickup_obj_start_pose": [1.8, 2.2, 1.0, 0, 0, 0, 1],
            "pickup_obj_goal_pose": [1.8, 2.2, 1.2, 0, 0, 0, 1],
            "succ_pos_threshold": 0.01
        },
        "language": {
            "task_description": task_description,
            "referral_expressions": {
                "pickup_obj_name": pickup_object_name
            }
        }
    }
    
    return episode


def generate_minimal_benchmark(
    output_path: str,
    num_episodes: int = 3
):
    """Generate a minimal benchmark with specified number of episodes."""
    
    episodes = []
    
    # Create 1 episode with fixed configuration
    episodes.append(create_minimal_episode(
        episode_id=0,
        house_index=321,
        pickup_object_name="book",
        task_description="Pick up the book"
    ))
    
    # Write benchmark to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(episodes, f, indent=2)
    
    print(f"Generated minimal benchmark with {len(episodes)} episodes at: {output_file}")
    return output_file


def main():
    """Generate minimal benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate minimal MolmoSpaces benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="benchmarks/minimal_benchmark/benchmark.json",
        help="Output path for the generated benchmark"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to generate"
    )
    
    args = parser.parse_args()
    
    generate_minimal_benchmark(args.output_path, args.num_episodes)


if __name__ == "__main__":
    main()