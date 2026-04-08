"""
Generate a benchmark JSON for pick-and-place tasks with a custom minimal scene.

Scene: desk + bookcase side-by-side in front of a Franka robot.
Tasks:
  1. Pick tissue box from desk → place in bookcase shelf
  2. Pick pencil from desk → place in cup on desk

Usage:
    python scripts/create_book_pencil_benchmark.py

Output:
    benchmarks/franka_book_pencil_pick_place/benchmark.json
    benchmarks/franka_book_pencil_pick_place/benchmark_metadata.json
    benchmarks/franka_book_pencil_pick_place/custom_scene.xml
    benchmarks/franka_book_pencil_pick_place/custom_scene_ceiling.xml
"""

import json
import shutil
from datetime import date
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import molmo_spaces
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.utils.lazy_loading_utils import install_uid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"
HOUSE_BASE_XML = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"

# Use absolute path for scene_dataset — triggers custom path handling in task_sampler.
# We patch the ceiling variant to also point to our scene file.

THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
ROBOT_BASE_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

ROBOT_INIT_QPOS = {
    "base": [],
    "arm": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "gripper": [0.003, 0.003],
}

IMG_RESOLUTION = [624, 352]

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

DESK_SURFACE_Z = 0.76


def build_custom_scene_xml() -> None:
    """Build minimal scene with ground, desk, bookcase, and Franka robot.

    Saves to ASSETS_DIR/scenes/custom-benchmark-val/ using ProcTHOR naming
    conventions (val_0.xml, val_0_ceiling.xml) so the eval pipeline can
    discover the scene files via get_scenes().
    """
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Ground plane
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0],
        contype=8,
        conaffinity=15,
    )

    # --- Layout (top-down view, robot at origin facing +X) ---
    # Robot base pedestal is 0.5x0.5m, arm reaches ~0.85m.
    # Desk to the right (+X, +Y), bookcase to the right (+X, -Y).
    # Both open-side facing the robot so it can reach objects and shelves.

    # Desk (static) — to the left of robot (+Y side)
    desk_xml = install_uid("RoboTHOR_desk_lisabo")
    desk_spec = mujoco.MjSpec.from_file(str(desk_xml))
    desk_body = desk_spec.worldbody.bodies[0]
    for j in desk_body.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            j.damping = 1e10
    desk_frame = spec.worldbody.add_frame(pos=[0.7, 0.4, 0.38], quat=THOR_QUAT)
    desk_frame.attach_body(desk_body, "desk/", "")

    # Bookcase (static) — to the right (-Y side), open shelves facing robot
    shelf_xml = install_uid("Shelving_Unit_206_1")
    shelf_spec = mujoco.MjSpec.from_file(str(shelf_xml))
    shelf_body = shelf_spec.worldbody.bodies[0]
    for j in shelf_body.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            j.damping = 1e10
    shelf_quat = (R.from_euler("z", 180, degrees=True) * R.from_euler("x", 90, degrees=True)).as_quat(scalar_first=True)
    shelf_frame = spec.worldbody.add_frame(pos=[0.55, -0.5, 0.35], quat=shelf_quat)
    shelf_frame.attach_body(shelf_body, "bookcase/", "")

    # Robot at origin, facing +X toward furniture
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(
        robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace,
        pos=[0, 0], quat=[1, 0, 0, 0],
    )

    model = spec.compile()
    print(f"  Scene compiled: {model.nbody} bodies, {model.ngeom} geoms")

    xml_string = spec.to_xml()

    scene_path = OUTPUT_DIR / "custom_scene.xml"
    with open(scene_path, "w") as f:
        f.write(xml_string)

    print(f"  Saved scene to {scene_path}")
    return scene_path


def asset_rel_path(uid: str) -> str:
    return str(install_uid(uid).relative_to(ASSETS_DIR))


def make_pose(x, y, z, quat=None):
    if quat is None:
        quat = THOR_QUAT.tolist()
    return [x, y, z] + list(quat)


def create_box_to_bookcase_episode(scene_xml: str) -> dict:
    """Pick Tissue_Box_1 from desk → place in bookcase shelf."""
    uid = "Tissue_Box_1"
    obj_pose = make_pose(0.6, 0.4, DESK_SURFACE_Z + 0.08)
    obj_goal = make_pose(0.6, 0.4, DESK_SURFACE_Z + 0.13)
    shelf_pose = make_pose(0.55, -0.5, 0.9)

    return {
        "source": None,
        "house_index": 0,
        "scene_dataset": scene_xml,
        "data_split": "val",
        "seed": None,
        "robot": {"robot_name": "franka_droid", "init_qpos": ROBOT_INIT_QPOS},
        "img_resolution": IMG_RESOLUTION,
        "cameras": CAMERAS,
        "scene_modifications": {
            "added_objects": {f"pickup_object/{uid}": asset_rel_path(uid)},
            "object_poses": {f"pickup_object/{uid}": obj_pose},
            "removed_objects": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
            "task_type": "pick_and_place",
            "robot_base_pose": ROBOT_BASE_POSE,
            "pickup_obj_name": f"pickup_object/{uid}",
            "pickup_obj_start_pose": obj_pose,
            "pickup_obj_goal_pose": obj_goal,
            "succ_pos_threshold": 0.03,
            "place_receptacle_name": "bookcase/Shelving_Unit_206_1",
            "place_receptacle_start_pose": shelf_pose,
            "receptacle_supported_weight_frac": 0.5,
            "max_place_receptacle_pos_displacement": 0.15,
            "max_place_receptacle_rot_displacement": float(np.deg2rad(60)),
        },
        "task_relevant_objects": [f"pickup_object/{uid}", "bookcase/Shelving_Unit_206_1"],
        "language": {
            "task_description": "Pick up the tissue box from the desk and place it in the bookcase",
            "referral_expressions": {"pickup_name": "tissue box", "place_name": "bookcase"},
        },
    }


def create_pencil_to_cup_episode(scene_xml: str) -> dict:
    """Pick Pencil_1 from desk → place in Cup_5 on desk."""
    pencil_uid = "Pencil_1"
    cup_uid = "Cup_5"

    pencil_pose = make_pose(0.6, 0.25, DESK_SURFACE_Z + 0.02)
    pencil_goal = make_pose(0.6, 0.25, DESK_SURFACE_Z + 0.07)
    cup_pose = make_pose(0.5, 0.5, DESK_SURFACE_Z + 0.08)

    return {
        "source": None,
        "house_index": 0,
        "scene_dataset": scene_xml,
        "data_split": "val",
        "seed": None,
        "robot": {"robot_name": "franka_droid", "init_qpos": ROBOT_INIT_QPOS},
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
            "pickup_obj_goal_pose": pencil_goal,
            "succ_pos_threshold": 0.03,
            "place_receptacle_name": f"place_receptacle/{cup_uid}",
            "place_receptacle_start_pose": cup_pose,
            "receptacle_supported_weight_frac": 0.5,
            "max_place_receptacle_pos_displacement": 0.15,
            "max_place_receptacle_rot_displacement": float(np.deg2rad(60)),
        },
        "task_relevant_objects": [f"pickup_object/{pencil_uid}", f"place_receptacle/{cup_uid}"],
        "language": {
            "task_description": "Pick up the pencil and put it in the cup",
            "referral_expressions": {"pickup_name": "pencil", "place_name": "cup"},
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building custom scene XML...")
    scene_path = build_custom_scene_xml()
    scene_xml = str(scene_path.resolve())

    print("Building episodes...")
    episodes = [
        create_box_to_bookcase_episode(scene_xml),
        create_pencil_to_cup_episode(scene_xml),
    ]

    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    for i, ep in enumerate(episodes):
        EpisodeSpec.model_validate(ep)
        print(f"  Episode {i}: {ep['language']['task_description']} — OK")

    with open(OUTPUT_DIR / "benchmark.json", "w") as f:
        json.dump(episodes, f, indent=2)

    today = date.today().isoformat()
    metadata = {
        "description": "Franka tissue-box-to-bookcase and pencil-to-cup benchmark (custom scene)",
        "created_at": today,
        "num_episodes": len(episodes),
        "num_houses": 1,
        "task_cls_counts": {
            "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask": len(episodes),
        },
        "robot_counts": {"franka_droid": len(episodes)},
        "benchmark_created_date": today,
    }
    with open(OUTPUT_DIR / "benchmark_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote benchmark to {OUTPUT_DIR}")
    print("View scene:  python scripts/view_benchmark_scene.py")
    print("Run eval:    python launch_scripts/run_eval.py \\")
    print("               --checkpoint_path <path> \\")
    print("               --benchmark_path benchmarks/franka_book_pencil_pick_place \\")
    print("               --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \\")
    print("               --task_horizon 600")


if __name__ == "__main__":
    main()
