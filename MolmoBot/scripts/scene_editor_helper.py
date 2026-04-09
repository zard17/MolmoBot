"""
Export/import helper for editing benchmark scenes with mujoco-scene-editor (mjedit).

Workflow:
    1. Export: Merges custom_scene.xml + robot + objects into a single editable XML
    2. Edit:   Open in mjedit, drag objects to desired positions, save
    3. Import: Read edited XML, extract positions, update benchmark.json

Usage:
    python scripts/scene_editor_helper.py export
    mjedit benchmarks/franka_book_pencil_pick_place/full_scene_editable.xml
    python scripts/scene_editor_helper.py import
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import molmo_spaces
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.utils.lazy_loading_utils import install_uid

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"
EDITABLE_XML = BENCHMARK_DIR / "full_scene_editable.xml"
BENCHMARK_JSON = BENCHMARK_DIR / "benchmark.json"
CUSTOM_SCENE_XML = BENCHMARK_DIR / "custom_scene.xml"
HOUSE_BASE_XML = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)


def export_scene():
    """Build a single XML with everything for visual editing in mjedit."""
    print("Exporting full scene for editing...")

    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Ground
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0], contype=8, conaffinity=15,
    )

    # Load current furniture positions from custom_scene.xml if it exists
    # Otherwise use defaults from create_book_pencil_benchmark.py
    desk_pos = [0.75, 0.25, 0.0]
    bookcase_pos = [0.55, -0.55, 0.0]

    # Simplified desk — single body so it moves as one unit in mjedit
    desk = spec.worldbody.add_body(name="desk", pos=desk_pos)
    desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"),
                        (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36],
            rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

    # Simplified bookcase — single body
    bc = spec.worldbody.add_body(name="bookcase", pos=bookcase_pos)
    bc_c = [0.7, 0.6, 0.4, 1.0]
    bc.add_geom(name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01], rgba=bc_c, contype=8, conaffinity=15)
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc.add_geom(name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.29, 0.15, 0.01], pos=[0, 0, sz], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6], rgba=bc_c, contype=8, conaffinity=15)

    # Robot
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])

    # Dynamic objects from benchmark.json
    if BENCHMARK_JSON.exists():
        with open(BENCHMARK_JSON) as f:
            episodes = json.load(f)

        # Collect all unique objects across episodes
        added = {}
        poses = {}
        for ep in episodes:
            sm = ep.get("scene_modifications", {})
            for name, rel_path in sm.get("added_objects", {}).items():
                if name not in added:
                    added[name] = rel_path
            for name, pose in sm.get("object_poses", {}).items():
                if name not in poses:
                    poses[name] = pose

        for obj_name, rel_path in added.items():
            xml_path = ASSETS_DIR / rel_path
            if not xml_path.exists():
                # Try install_uid for Thor assets
                uid = obj_name.split("/")[-1]
                try:
                    xml_path = install_uid(uid)
                except Exception:
                    print(f"  WARNING: Could not find asset for {obj_name}, skipping")
                    continue

            obj_spec = mujoco.MjSpec.from_file(str(xml_path))
            body = obj_spec.worldbody.bodies[0]
            if not body.first_joint():
                body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)

            pose = poses.get(obj_name, [0, 0, 1, 1, 0, 0, 0])
            pos = pose[:3]
            quat = pose[3:7]

            frame = spec.worldbody.add_frame(pos=pos, quat=quat)
            parts = obj_name.split("/")
            prefix = "/".join(parts[:-1]) + "/"
            frame.attach_body(body, prefix, "")
            print(f"  Added {obj_name} at pos={pos}")
    else:
        print("  No benchmark.json found, exporting scene without objects")

    model = spec.compile()
    print(f"  Compiled: {model.nbody} bodies, {model.ngeom} geoms")

    with open(EDITABLE_XML, "w") as f:
        f.write(spec.to_xml())
    print(f"  Saved to {EDITABLE_XML}")
    print(f"\nNow edit with: mjedit {EDITABLE_XML}")


def import_scene():
    """Read edited XML and extract updated positions into benchmark.json."""
    if not EDITABLE_XML.exists():
        print(f"ERROR: {EDITABLE_XML} not found. Run 'export' first, then edit with mjedit.")
        sys.exit(1)

    print(f"Importing positions from {EDITABLE_XML}...")

    # Re-export the scene to get a compilable model (the mjedit-saved XML
    # may have issues with duplicate defaults). We rebuild the same way
    # as export but load body positions from the edited XML via ElementTree.
    import xml.etree.ElementTree as ET
    tree = ET.parse(EDITABLE_XML)
    root = tree.getroot()

    # Extract body positions from the edited XML
    body_positions = {}
    for body_elem in root.iter("body"):
        name = body_elem.get("name", "")
        pos_str = body_elem.get("pos", "0 0 0")
        quat_str = body_elem.get("quat", "1 0 0 0")
        pos = [float(x) for x in pos_str.split()]
        quat = [float(x) for x in quat_str.split()]
        if name:
            body_positions[name] = {"pos": pos, "quat": quat}

    # Also check for frames (objects attached via frames)
    for frame_elem in root.iter("frame"):
        pos_str = frame_elem.get("pos", "0 0 0")
        quat_str = frame_elem.get("quat", "1 0 0 0")
        pos = [float(x) for x in pos_str.split()]
        quat = [float(x) for x in quat_str.split()]
        # Find child body inside frame
        for child in frame_elem:
            if child.tag == "body":
                name = child.get("name", "")
                if name:
                    body_positions[name] = {"pos": pos, "quat": quat}

    # body_positions already populated from XML parsing above

    # Print furniture positions (user must update scripts manually)
    print("\n=== Furniture positions (update in create_book_pencil_benchmark.py + view scripts) ===")
    for name in ["desk", "bookcase"]:
        if name in body_positions:
            p = body_positions[name]
            print(f"  {name}: pos={[round(x, 3) for x in p['pos']]}")

    # Update benchmark.json
    if not BENCHMARK_JSON.exists():
        print(f"\nNo benchmark.json to update.")
        return

    with open(BENCHMARK_JSON) as f:
        episodes = json.load(f)

    print("\n=== Object positions (updating benchmark.json) ===")
    changed = False
    for ep in episodes:
        sm = ep.get("scene_modifications", {})
        task = ep.get("task", {})

        # Update object_poses
        for obj_name in list(sm.get("object_poses", {}).keys()):
            # Find matching body — the prefix is stripped during attach
            # e.g. "pickup_object/Tissue_Box_1" → body name is "Tissue_Box_1"
            # but attached with prefix "pickup_object/" so full name is "pickup_object/Tissue_Box_1"
            # Check various name patterns
            for body_name, bp in body_positions.items():
                if body_name == obj_name or body_name.endswith("/" + obj_name.split("/")[-1]):
                    new_pose = bp["pos"] + bp["quat"]
                    old_pose = sm["object_poses"][obj_name]
                    if new_pose != old_pose:
                        print(f"  {obj_name}:")
                        print(f"    old: pos={[round(x, 3) for x in old_pose[:3]]}")
                        print(f"    new: pos={[round(x, 3) for x in new_pose[:3]]}")
                        sm["object_poses"][obj_name] = new_pose
                        changed = True

                        # Also update task fields that reference this object
                        if task.get("pickup_obj_name") == obj_name:
                            task["pickup_obj_start_pose"] = new_pose
                            goal = list(new_pose)
                            goal[2] += 0.05  # goal is 5cm above start
                            task["pickup_obj_goal_pose"] = goal
                        if task.get("place_receptacle_name") == obj_name:
                            task["place_receptacle_start_pose"] = new_pose
                    break

    if changed:
        with open(BENCHMARK_JSON, "w") as f:
            json.dump(episodes, f, indent=2)
        print(f"\nUpdated {BENCHMARK_JSON}")
    else:
        print(f"\nNo position changes detected in benchmark objects.")

    print("\nNext steps:")
    print("  1. Update furniture positions in create_book_pencil_benchmark.py")
    print("  2. Update furniture positions in view_benchmark_scene.py")
    print("  3. Update furniture positions in run_benchmark_with_viewer.py")
    print("  4. Run: python scripts/create_book_pencil_benchmark.py")
    print("  5. Verify: python scripts/view_benchmark_scene.py --preview")


def main():
    parser = argparse.ArgumentParser(description="Scene editor export/import helper")
    parser.add_argument("command", choices=["export", "import"],
                        help="export: build editable XML; import: read positions back")
    args = parser.parse_args()

    if args.command == "export":
        export_scene()
    elif args.command == "import":
        import_scene()


if __name__ == "__main__":
    main()
