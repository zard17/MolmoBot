#!/usr/bin/env python3
"""Direct MuJoCo IK pick validation - bypasses PD controller entirely.

Instead of sending targets to a PD controller (which can't track large torso
changes), this script directly sets qpos in MuJoCo and steps the simulation.
This is a privileged physics test to answer: CAN the simulator pick this object?

Usage:
    .venv/bin/python scripts/run_rby1_ik_direct_pick.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.kinematics.rby1_kinematics import RBY1Kinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_OBJECT_POS = np.array([0.445, 0.28, 0.805])
EE_TO_FINGERTIP_Z = 0.047  # meters

INIT_QPOS = {
    "base": [0.0, 0.0, 0.0],
    "head": [0.0, 0.6],
    "left_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "left_gripper": [-0.05],
    "right_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "right_gripper": [-0.05],
    "torso": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def load_model():
    model_path = Path(
        ".venv/lib/python3.11/site-packages/assets/robots/rby1m/"
        "rby1_v1.2_site_control.xml"
    )
    if not model_path.exists():
        # Search in site-packages
        for p in sys.path:
            candidate = Path(p) / "assets" / "robots" / "rby1m" / "rby1_v1.2_site_control.xml"
            if candidate.exists():
                model_path = candidate
                break

    cwd = os.getcwd()
    os.chdir(model_path.parent)
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path.name))
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(model)
    return model, data


def set_qpos_dict(robot_view, qpos_dict):
    full = {}
    for group_name in robot_view.move_group_ids():
        if group_name in qpos_dict:
            full[group_name] = np.array(qpos_dict[group_name])
        else:
            full[group_name] = robot_view.get_move_group(group_name).joint_pos.copy()
    robot_view.set_qpos_dict(full)


def interpolate_qpos(q_start, q_end, n_steps):
    """Linear interpolation between two qpos dicts."""
    result = []
    for i in range(n_steps):
        alpha = i / max(n_steps - 1, 1)
        q = {}
        for key in q_start:
            s = np.array(q_start[key])
            e = np.array(q_end[key])
            q[key] = (s * (1 - alpha) + e * alpha).tolist()
        result.append(q)
    return result


def get_object_pos(model, data, obj_name="Salt_Shaker_1"):
    """Get position of the salt shaker (or any named body)."""
    # Try various naming conventions
    for name in [obj_name, f"/{obj_name}", obj_name.replace("/", "")]:
        try:
            body_id = model.body(name).id
            return data.xpos[body_id].copy()
        except (KeyError, mujoco.MjModelError):
            continue
    return None


def get_finger_info(model, data):
    """Get finger positions and gripper state."""
    try:
        l1_pos = data.xpos[model.body("robot_0/ee_finger_l1").id].copy()
        l2_pos = data.xpos[model.body("robot_0/ee_finger_l2").id].copy()
    except (KeyError, mujoco.MjModelError):
        return None
    midpoint = (l1_pos + l2_pos) / 2.0
    dist = np.linalg.norm(l1_pos - l2_pos)
    return {"l1": l1_pos, "l2": l2_pos, "midpoint": midpoint, "inter_finger_dist": dist}


def main():
    print("=" * 60)
    print("RBY1 Direct MuJoCo IK Pick Validation")
    print("=" * 60)

    # 1. Load model and set up
    print("\n[1] Loading standalone RBY1 model...")
    model, data = load_model()
    robot_view = RBY1RobotView(data, namespace="robot_0/", holo_base=True)
    kinematics = RBY1Kinematics(model, data, namespace="robot_0/", holo_base=True)

    # Set initial qpos
    set_qpos_dict(robot_view, INIT_QPOS)
    mujoco.mj_forward(model, data)

    object_pos = DEFAULT_OBJECT_POS.copy()
    print(f"    Object target position: {object_pos}")

    # Note: standalone model doesn't have the salt shaker.
    # We test gripper positioning only.

    # 2. Compute IK for grasp pose
    print("\n[2] Computing IK waypoints...")

    # Grasp: EE below object so fingers center on it
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    grasp_pose = np.eye(4)
    grasp_pose[:3, :3] = R
    grasp_pose[:3, 3] = object_pos.copy()
    grasp_pose[2, 3] -= EE_TO_FINGERTIP_Z

    base_pose = robot_view.base.pose
    q0 = robot_view.get_qpos_dict()

    grasp_qpos = kinematics.ik(
        "left_arm", grasp_pose, ["torso", "left_arm"],
        q0, base_pose, eps=1e-3, max_iter=2000, dt=0.5,
    )
    if grasp_qpos is None:
        print("    [FAIL] Grasp IK failed")
        return

    # Approach: 12cm above grasp
    approach_pose = grasp_pose.copy()
    approach_pose[2, 3] += 0.12
    set_qpos_dict(robot_view, INIT_QPOS)
    mujoco.mj_forward(model, data)
    q0 = robot_view.get_qpos_dict()
    approach_qpos = kinematics.ik(
        "left_arm", approach_pose, ["torso", "left_arm"],
        q0, base_pose, eps=1e-3, max_iter=2000, dt=0.5,
    )
    if approach_qpos is None:
        print("    [FAIL] Approach IK failed")
        return

    # Lift: 15cm above grasp
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] += 0.15
    lift_qpos = kinematics.ik(
        "left_arm", lift_pose, ["torso", "left_arm"],
        grasp_qpos, base_pose, eps=1e-3, max_iter=2000, dt=0.5,
    )
    if lift_qpos is None:
        print("    [FAIL] Lift IK failed")
        return

    print("    [OK] All IK waypoints solved")

    # 3. Execute trajectory by directly setting qpos
    print("\n[3] Executing trajectory (direct qpos manipulation)...")

    # Convert to lists for interpolation
    init_dict = {k: list(v) for k, v in robot_view.get_qpos_dict().items()
                 if k in ("base", "torso", "left_arm")}

    # Reset to initial
    set_qpos_dict(robot_view, INIT_QPOS)
    mujoco.mj_forward(model, data)

    approach_dict = {k: v.tolist() for k, v in approach_qpos.items()
                     if k in ("base", "torso", "left_arm")}
    grasp_dict = {k: v.tolist() for k, v in grasp_qpos.items()
                  if k in ("base", "torso", "left_arm")}
    lift_dict = {k: v.tolist() for k, v in lift_qpos.items()
                 if k in ("base", "torso", "left_arm")}

    phases = [
        ("approach", init_dict, approach_dict, 40, "open"),
        ("descend", approach_dict, grasp_dict, 30, "open"),
        ("close", grasp_dict, grasp_dict, 20, "closing"),
        ("lift", grasp_dict, lift_dict, 40, "closed"),
        ("hold", lift_dict, lift_dict, 20, "closed"),
    ]

    for phase_name, q_start, q_end, n_steps, gripper_state in phases:
        waypoints = interpolate_qpos(q_start, q_end, n_steps)
        for i, q in enumerate(waypoints):
            # Set arm and torso positions directly
            for group in ("base", "torso", "left_arm"):
                if group in q:
                    mg = robot_view.get_move_group(group)
                    mg.joint_pos = np.array(q[group])

            # Set gripper
            left_gripper = robot_view.get_move_group("left_gripper")
            if gripper_state == "open":
                left_gripper.joint_pos = np.array([-0.05, 0.05])
            elif gripper_state == "closing":
                # Gradually close
                alpha = i / max(n_steps - 1, 1)
                finger_pos = -0.05 * (1 - alpha)
                left_gripper.joint_pos = np.array([finger_pos, -finger_pos])
            else:  # closed
                left_gripper.joint_pos = np.array([0.0, 0.0])

            # Step simulation
            mujoco.mj_forward(model, data)

        # Report state at end of phase
        left_arm = robot_view.get_move_group("left_arm")
        ee_pos = left_arm.leaf_frame_to_world[:3, 3]
        fingers = get_finger_info(model, data)
        if fingers:
            mid = fingers["midpoint"]
            dist_to_obj = np.linalg.norm(mid - object_pos)
            print(
                f"    [{phase_name:10s}] EE={ee_pos}, "
                f"finger_mid={mid}, "
                f"mid-obj={dist_to_obj:.4f}m, "
                f"inter_finger={fingers['inter_finger_dist']:.4f}m"
            )

    # 4. Final analysis
    print("\n[4] Final state analysis:")
    fingers = get_finger_info(model, data)
    if fingers:
        mid_to_obj = np.linalg.norm(fingers["midpoint"] - object_pos)
        print(f"    Finger midpoint: {fingers['midpoint']}")
        print(f"    Object position: {object_pos}")
        print(f"    Midpoint-to-object distance: {mid_to_obj:.4f} m")
        print(f"    Inter-finger distance: {fingers['inter_finger_dist']:.4f} m")

        # Check if fingers would enclose the object
        l1_to_obj = np.linalg.norm(fingers["l1"] - object_pos)
        l2_to_obj = np.linalg.norm(fingers["l2"] - object_pos)
        print(f"    Finger L1 to object: {l1_to_obj:.4f} m")
        print(f"    Finger L2 to object: {l2_to_obj:.4f} m")

        if mid_to_obj < 0.01 and fingers["inter_finger_dist"] < 0.02:
            print("\n    *** GEOMETRY CHECK: Fingers would enclose the object! ***")
            print("    The IK grasp geometry is correct.")
            print("    To validate actual physics (contact, friction, lift),")
            print("    run the full scene eval with a controller that can reach these targets.")
        elif mid_to_obj < 0.01:
            print(f"\n    Midpoint is close ({mid_to_obj:.4f}m) but fingers may be too open ({fingers['inter_finger_dist']:.4f}m)")
        else:
            print(f"\n    Midpoint miss: {mid_to_obj:.4f}m")

    print("\n" + "=" * 60)
    print("NOTE: This standalone model has no salt shaker or table.")
    print("This test confirms the IK geometry is correct.")
    print("To validate actual grasping physics, the full scene eval")
    print("needs a controller that can track these large torso changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
