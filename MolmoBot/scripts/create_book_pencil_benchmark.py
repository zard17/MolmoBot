"""
Generate a benchmark JSON for book-to-shelf and pencil-to-cup pick-and-place tasks.

Uses ProcTHOR val house 0 as the base scene (same as demo_policy.ipynb).
Adds Book_3, Pencil_1, Cup_5, and Shelving_Unit_206_1 as auxiliary objects.

Usage:
    python scripts/create_book_pencil_benchmark.py

Output:
    benchmarks/franka_book_pencil_pick_place/benchmark.json
    benchmarks/franka_book_pencil_pick_place/benchmark_metadata.json
"""

import json
import sys
from datetime import date
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_procthor_10k_houses
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.object_metadata import ObjectMeta

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"

# ---- Scene constants (from demo_policy.ipynb, ProcTHOR val house 0) ----

HOUSES = get_procthor_10k_houses(split="val")
HOUSE_INDEX = 0
SCENE_DATASET = "procthor-10k"
DATA_SPLIT = "val"

# Robot placement: same area as demo, facing +Y (yaw=90)
ROBOT_YAW_DEG = 90.0
ROBOT_POS = [6.8, 9.75]
ROBOT_BASE_Z = 0.0  # floor level; FrankaRobotConfig.base_size handles pedestal

_robot_quat = R.from_euler("z", ROBOT_YAW_DEG, degrees=True).as_quat(scalar_first=True)
ROBOT_BASE_POSE = [ROBOT_POS[0], ROBOT_POS[1], ROBOT_BASE_Z] + _robot_quat.tolist()

# Default Franka home qpos
ROBOT_INIT_QPOS = {
    "base": [],
    "arm": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "gripper": [0.003, 0.003],
}

IMG_RESOLUTION = [624, 352]

# Camera specs (from real benchmark — robot-mounted on fr3_link0 and gripper)
CAMERAS = [
    {
        "name": "wrist_camera",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/gripper/base"],
        "camera_offset": [0.031, 0.074, 0.022],
        "lookat_offset": [0.0, 0.0, 0.08],
        "camera_quaternion": [-0.0056, -0.001, 0.9856, 0.169],
        "fov": 56.74,
        "record_depth": False,
    },
    {
        "name": "exo_camera_1",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/fr3_link0"],
        "camera_offset": [0.1, 0.57, 0.66],
        "lookat_offset": [0.0, 0.0, 0.08],
        "camera_quaternion": [-0.3633, -0.1241, 0.4263, 0.8191],
        "fov": 71.0,
        "record_depth": False,
    },
]

# Thor convention: assets use Y-up, MuJoCo uses Z-up → rotate x +90°
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True).tolist()


def asset_rel_path(uid: str) -> str:
    """Return the asset XML path relative to ASSETS_DIR."""
    xml_path = install_uid(uid)
    return str(xml_path.relative_to(ASSETS_DIR))


def bounding_box(uid: str) -> dict:
    """Return bounding box dict {x, y, z} in the asset's native frame."""
    return ObjectMeta.annotation(uid).get("boundingBox", {})


def make_pose(x: float, y: float, z: float, quat: list[float] | None = None) -> list[float]:
    """Build a 7-element pose [x, y, z, qw, qx, qy, qz]."""
    if quat is None:
        quat = THOR_QUAT
    return [x, y, z] + list(quat)


def find_countertop_surface_z() -> float:
    """Load the ProcTHOR house and return the countertop surface Z near the robot."""
    house_xml = HOUSES[DATA_SPLIT][HOUSE_INDEX]["base"]
    spec = mujoco.MjSpec.from_file(house_xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Find the countertop body closest to robot
    robot_xy = np.array(ROBOT_POS)
    best_z = 0.94  # fallback
    best_dist = float("inf")
    for i in range(model.nbody):
        name = model.body(i).name
        if "countertop" in name:
            pos = data.xpos[i]
            dist = np.linalg.norm(pos[:2] - robot_xy)
            if dist < best_dist:
                best_dist = dist
                best_z = pos[2]
    return best_z


def create_mug_to_shelf_episode(countertop_z: float) -> dict:
    """Episode: pick Mug_1 from countertop, place on Shelving_Unit_206_1.

    Note: All Thor Book assets are articulated (openable cover with hinge joint),
    which requires grasp files that don't exist for these assets. We use Mug_1
    (rigid) instead as the pickup object.
    """
    mug_uid = "Mug_1"
    shelf_uid = "Shelving_Unit_206_1"

    mug_bb = bounding_box(mug_uid)
    shelf_bb = bounding_box(shelf_uid)

    # Mug on the countertop in front of robot (robot faces +Y from [6.8, 9.75])
    # Countertop is around [6.37, 10.03] — place mug on it
    mug_z = countertop_z + mug_bb["z"] / 2 + 0.01
    mug_pose = make_pose(6.5, 10.1, mug_z)

    # Shelf to the right of the countertop, on the floor
    # After x-90 rotation, shelf's Z axis (height=2.2m) maps to MuJoCo Z
    shelf_z = shelf_bb["z"] / 2
    shelf_pose = make_pose(7.3, 10.2, shelf_z)

    # Goal pose: mug lifted 5cm above start
    mug_goal_pose = make_pose(6.5, 10.1, mug_z + 0.05)

    return {
        "source": None,
        "house_index": HOUSE_INDEX,
        "scene_dataset": SCENE_DATASET,
        "data_split": DATA_SPLIT,
        "seed": None,
        "robot": {
            "robot_name": "franka_droid",
            "init_qpos": ROBOT_INIT_QPOS,
        },
        "img_resolution": IMG_RESOLUTION,
        "cameras": CAMERAS,
        "scene_modifications": {
            "added_objects": {
                f"pickup_object/{mug_uid}": asset_rel_path(mug_uid),
                f"place_receptacle/{shelf_uid}": asset_rel_path(shelf_uid),
            },
            "object_poses": {
                f"pickup_object/{mug_uid}": mug_pose,
                f"place_receptacle/{shelf_uid}": shelf_pose,
            },
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
            "task_type": "pick_and_place",
            "robot_base_pose": ROBOT_BASE_POSE,
            "pickup_obj_name": f"pickup_object/{mug_uid}",
            "pickup_obj_start_pose": mug_pose,
            "pickup_obj_goal_pose": mug_goal_pose,
            "succ_pos_threshold": 0.03,
            "place_receptacle_name": f"place_receptacle/{shelf_uid}",
            "place_receptacle_start_pose": shelf_pose,
            "receptacle_supported_weight_frac": 0.5,
            "max_place_receptacle_pos_displacement": 0.15,
            "max_place_receptacle_rot_displacement": float(np.deg2rad(60)),
        },
        "task_relevant_objects": [
            f"pickup_object/{mug_uid}",
            f"place_receptacle/{shelf_uid}",
        ],
        "language": {
            "task_description": "Pick up the mug and place it on the shelf",
            "referral_expressions": {
                "pickup_name": "mug",
                "place_name": "shelf",
            },
        },
    }


def create_pencil_to_cup_episode(countertop_z: float) -> dict:
    """Episode: pick Pencil_1 from countertop, place in Cup_5."""
    pencil_uid = "Pencil_1"
    cup_uid = "Cup_5"

    pencil_bb = bounding_box(pencil_uid)
    cup_bb = bounding_box(cup_uid)

    # Pencil on countertop, slightly offset from the book position
    pencil_z = countertop_z + pencil_bb["z"] / 2 + 0.01
    pencil_pose = make_pose(6.6, 10.15, pencil_z)

    # Cup on countertop nearby
    cup_z = countertop_z + cup_bb["z"] / 2 + 0.01
    cup_pose = make_pose(6.9, 10.1, cup_z)

    # Goal pose: pencil lifted 5cm
    pencil_goal_pose = make_pose(6.6, 10.15, pencil_z + 0.05)

    return {
        "source": None,
        "house_index": HOUSE_INDEX,
        "scene_dataset": SCENE_DATASET,
        "data_split": DATA_SPLIT,
        "seed": None,
        "robot": {
            "robot_name": "franka_droid",
            "init_qpos": ROBOT_INIT_QPOS,
        },
        "img_resolution": IMG_RESOLUTION,
        "cameras": CAMERAS,
        "scene_modifications": {
            "added_objects": {
                f"pickup_object/{pencil_uid}": asset_rel_path(pencil_uid),
                f"place_receptacle/{cup_uid}": asset_rel_path(cup_uid),
            },
            "object_poses": {
                f"pickup_object/{pencil_uid}": pencil_pose,
                f"place_receptacle/{cup_uid}": cup_pose,
            },
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
            "task_type": "pick_and_place",
            "robot_base_pose": ROBOT_BASE_POSE,
            "pickup_obj_name": f"pickup_object/{pencil_uid}",
            "pickup_obj_start_pose": pencil_pose,
            "pickup_obj_goal_pose": pencil_goal_pose,
            "succ_pos_threshold": 0.03,
            "place_receptacle_name": f"place_receptacle/{cup_uid}",
            "place_receptacle_start_pose": cup_pose,
            "receptacle_supported_weight_frac": 0.5,
            "max_place_receptacle_pos_displacement": 0.15,
            "max_place_receptacle_rot_displacement": float(np.deg2rad(60)),
        },
        "task_relevant_objects": [
            f"pickup_object/{pencil_uid}",
            f"place_receptacle/{cup_uid}",
        ],
        "language": {
            "task_description": "Pick up the pencil and put it in the cup",
            "referral_expressions": {
                "pickup_name": "pencil",
                "place_name": "cup",
            },
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Finding countertop surface height in ProcTHOR val house 0...")
    countertop_z = find_countertop_surface_z()
    print(f"  countertop_z = {countertop_z:.4f}")

    print("Building episodes...")
    episodes = [
        create_mug_to_shelf_episode(countertop_z),
        create_pencil_to_cup_episode(countertop_z),
    ]

    # Validate against EpisodeSpec
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    for i, ep in enumerate(episodes):
        EpisodeSpec.model_validate(ep)
        print(f"  Episode {i} ({ep['task']['task_type']}): validated OK")

    # Write benchmark.json
    benchmark_path = OUTPUT_DIR / "benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Wrote {benchmark_path}")

    # Write benchmark_metadata.json
    today = date.today().isoformat()
    metadata = {
        "description": "Franka book-to-shelf and pencil-to-cup pick-and-place benchmark",
        "created_at": today,
        "num_episodes": len(episodes),
        "num_houses": 1,
        "task_cls_counts": {
            "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask": len(episodes),
        },
        "robot_counts": {"franka_droid": len(episodes)},
        "benchmark_created_date": today,
    }
    metadata_path = OUTPUT_DIR / "benchmark_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")

    print("\nDone! Run evaluation with:")
    print(
        f"  python launch_scripts/run_eval.py \\\n"
        f"    --checkpoint_path <path> \\\n"
        f"    --benchmark_path {OUTPUT_DIR.relative_to(Path(__file__).resolve().parent.parent)} \\\n"
        f"    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \\\n"
        f"    --task_horizon 600"
    )


if __name__ == "__main__":
    main()
