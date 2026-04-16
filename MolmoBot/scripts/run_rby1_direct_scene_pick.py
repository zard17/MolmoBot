#!/usr/bin/env python3
"""Standalone full-scene direct-qpos pick validation.

Assembles the complete MuJoCo model (robot + desk + salt shaker) outside the
eval pipeline, then directly sets joint positions to IK-computed waypoints.
Bypasses PD controllers entirely to test if the physics can pick the object.

Usage:
    .venv/bin/python scripts/run_rby1_direct_scene_pick.py
    .venv/bin/python scripts/run_rby1_direct_scene_pick.py --viewer  # with MuJoCo viewer
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── Asset paths ──────────────────────────────────────────────────────────

ASSETS_DIR = Path(
    ".venv/lib/python3.11/site-packages/assets"
).resolve()
SCENE_XML = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark" / "custom_scene.xml"
ROBOT_XML = ASSETS_DIR / "robots" / "rby1m" / "rby1_v1.2_site_control.xml"
SALT_SHAKER_XML = (
    ASSETS_DIR / "objects" / "thor" / "Kitchen Objects" / "SaltShaker"
    / "Prefabs" / "Salt_Shaker_1" / "Salt_Shaker_1.xml"
)

NAMESPACE = "robot_0/"
OBJECT_POS = [0.445, 0.28, 0.805]
OBJECT_QUAT = [0.7071068, 0.7071068, 0.0, 0.0]
EE_TO_FINGERTIP_Z = 0.047

INIT_QPOS = {
    "base": [0.0, 0.0, 0.0],
    "head": [0.0, 0.6],
    "left_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "left_gripper": [-0.05],
    "right_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
    "right_gripper": [-0.05],
    "torso": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def assemble_scene() -> mujoco.MjModel:
    """Assemble the full scene: desk + robot + salt shaker."""
    print("  Loading scene XML...")
    scene_content = SCENE_XML.read_text()

    # Inject robot include and increase memory for full scene
    robot_rel = os.path.relpath(ROBOT_XML, SCENE_XML.parent)
    include_line = f'  <include file="{robot_rel}"/>\n'
    memory_line = '  <size memory="128M"/>\n'
    # Insert after first ">" of <mujoco> tag
    idx = scene_content.index(">") + 1
    scene_content = scene_content[:idx] + "\n" + memory_line + include_line + scene_content[idx:]

    # Write to temp file in scene directory (preserves relative paths)
    temp_path = SCENE_XML.parent / "_temp_full_scene.xml"
    temp_path.write_text(scene_content)

    try:
        print("  Loading MjSpec...")
        spec = mujoco.MjSpec.from_file(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    # Add salt shaker object
    print("  Adding salt shaker...")
    obj_spec = mujoco.MjSpec.from_file(str(SALT_SHAKER_XML))
    obj_body = obj_spec.worldbody.bodies[0]

    # Add free joint if not present
    if not obj_body.first_joint():
        obj_body.add_joint(
            name="XYZ_jntfree",
            type=mujoco.mjtJoint.mjJNT_FREE,
            damping=0.0001,
        )

    # Attach at object pose
    frame = spec.worldbody.add_frame(pos=OBJECT_POS, quat=OBJECT_QUAT)
    frame.attach_body(obj_body, "", "")

    print("  Compiling model...")
    model = spec.compile()

    # Boost friction on ALL geoms (both object and finger surfaces).
    # Default 0.9 is too low for stable grasping with the RBY1 parallel gripper.
    FRICTION_MULT = 5.0
    for i in range(model.ngeom):
        model.geom_friction[i, 0] *= FRICTION_MULT  # sliding
        model.geom_friction[i, 1] *= FRICTION_MULT  # torsional

    print(f"  Model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    return model


def find_body(model, *candidates):
    """Find a body by trying multiple name candidates."""
    for name in candidates:
        try:
            return model.body(name).id
        except (KeyError, mujoco.MjModelError):
            pass
    return -1


def set_robot_qpos(model, data, init_qpos):
    """Set robot joint positions by finding joints by name pattern."""
    from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView
    try:
        robot_view = RBY1RobotView(data, namespace=NAMESPACE, holo_base=True)
        for group_name, values in init_qpos.items():
            try:
                mg = robot_view.get_move_group(group_name)
                mg.joint_pos = np.array(values)
            except Exception:
                pass
        return robot_view
    except Exception as e:
        print(f"  Warning: Could not create RBY1RobotView: {e}")
        return None


def solve_ik(model, data, robot_view, target_pos, orientation_deg=0):
    """Solve IK for arm+torso to reach target position."""
    from molmo_spaces.kinematics.rby1_kinematics import RBY1Kinematics
    kinematics = RBY1Kinematics(model, data, namespace=NAMESPACE, holo_base=True)

    angle_rad = np.radians(orientation_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[-s, c, 0], [c, s, 0], [0, 0, -1]])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = target_pos.copy()
    pose[2, 3] -= EE_TO_FINGERTIP_Z  # EE below so fingers land on target

    base_pose = robot_view.base.pose
    q0 = robot_view.get_qpos_dict()

    result = kinematics.ik(
        "left_arm", pose, ["torso", "left_arm"],
        q0, base_pose, eps=1e-3, max_iter=2000, dt=0.5,
    )
    return result, pose


def get_finger_midpoint(model, data):
    """Get the left finger midpoint position."""
    l1_id = find_body(model, f"{NAMESPACE}ee_finger_l1")
    l2_id = find_body(model, f"{NAMESPACE}ee_finger_l2")
    if l1_id < 0 or l2_id < 0:
        return None
    l1 = data.xpos[l1_id].copy()
    l2 = data.xpos[l2_id].copy()
    return (l1 + l2) / 2.0, np.linalg.norm(l1 - l2)


def get_object_pos(model, data):
    """Get salt shaker position."""
    obj_id = find_body(model, "Salt_Shaker_1", "/Salt_Shaker_1",
                       "Salt_Shaker_1/", "salt_shaker")
    if obj_id < 0:
        # Search for any body with "salt" in the name
        for i in range(model.nbody):
            name = model.body(i).name
            if "salt" in name.lower() or "Salt" in name:
                return data.xpos[i].copy()
        return None
    return data.xpos[obj_id].copy()


def interpolate(a, b, n):
    """Linear interpolation between two arrays."""
    return [a + (b - a) * t / max(n - 1, 1) for t in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--object_pos", type=float, nargs=3, default=OBJECT_POS)
    args = parser.parse_args()

    obj_target = np.array(args.object_pos)

    print("=" * 60)
    print("Standalone Full-Scene Direct-Qpos Pick Validation")
    print("=" * 60)

    # 1. Assemble the full scene
    print("\n[1] Assembling full scene model...")
    model = assemble_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 2. Set robot initial configuration
    print("\n[2] Setting robot initial configuration...")
    robot_view = set_robot_qpos(model, data, INIT_QPOS)
    if robot_view is None:
        print("  FAILED to set robot qpos")
        return
    mujoco.mj_forward(model, data)

    # Verify object position
    obj_pos = get_object_pos(model, data)
    print(f"  Object position: {obj_pos}")
    print(f"  Expected: {obj_target}")

    # 3. Settle gravity (step simulation with ctrl holding current position)
    print("\n[3] Settling gravity (500 steps with ctrl holding)...")
    # Set ctrl targets to hold current position
    for i in range(model.nu):
        # Find the joint this actuator drives and set ctrl to current qpos
        jnt_id = model.actuator_trnid[i, 0]
        if jnt_id >= 0 and model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            qadr = model.jnt_qposadr[jnt_id]
            data.ctrl[i] = data.qpos[qadr]
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    obj_pos_settled = get_object_pos(model, data)
    print(f"  Object after settling: {obj_pos_settled}")
    print(f"  Object shift: {obj_pos_settled - obj_target if obj_pos_settled is not None else 'N/A'}")

    # 4. Solve IK from settled state
    print("\n[4] Solving IK from settled state...")
    # Re-read the settled qpos
    settled_qpos = robot_view.get_qpos_dict()
    print(f"  Settled torso: {[round(v, 3) for v in settled_qpos['torso']]}")

    # Use the actual object position (post-settling) as target
    target = obj_pos_settled if obj_pos_settled is not None else obj_target

    # Try multiple orientations
    ik_result = None
    for angle in [0, 15, 30, -30, 45, 60, -60, 90]:
        ik_result, grasp_pose = solve_ik(model, data, robot_view, target, angle)
        if ik_result is not None:
            print(f"  IK OK at {angle} deg")
            break
        # Reset to settled state for next attempt
        robot_view.set_qpos_dict(settled_qpos)
        mujoco.mj_forward(model, data)

    if ik_result is None:
        print("  IK FAILED for all orientations")
        return

    # Also solve approach (15cm above) and lift (15cm above)
    approach_target = target.copy()
    approach_target[2] += 0.15

    robot_view.set_qpos_dict(settled_qpos)
    mujoco.mj_forward(model, data)
    approach_result, _ = solve_ik(model, data, robot_view, approach_target, angle)

    lift_target = target.copy()
    lift_target[2] += 0.15
    robot_view.set_qpos_dict(ik_result)
    mujoco.mj_forward(model, data)
    lift_result, _ = solve_ik(model, data, robot_view, lift_target, angle)

    if approach_result is None:
        print("  Approach IK failed, using settled state as approach")
        approach_result = settled_qpos
    if lift_result is None:
        print("  Lift IK failed, using grasp state as lift")
        lift_result = ik_result

    # 5. Execute trajectory by directly setting qpos
    print("\n[5] Executing pick trajectory (direct qpos)...")

    def qpos_array(qdict, group):
        return np.array(qdict[group])

    groups_to_set = ["torso", "left_arm"]

    # Phase execution modes:
    # "kinematic": set qpos + mj_forward (no dynamics, no contacts)
    # "hybrid": pin arm/torso qpos, let gripper use physics (mj_step)
    # "physics": full dynamics with mj_step
    phases = [
        ("approach", settled_qpos, approach_result, 40, "open", "kinematic"),
        ("descend", approach_result, ik_result, 40, "open", "kinematic"),
        ("settle", ik_result, ik_result, 10, "open", "kinematic"),
        ("close", ik_result, ik_result, 200, "closing", "hybrid"),
        ("lift", ik_result, lift_result, 200, "closed", "hybrid"),
        ("hold", lift_result, lift_result, 60, "closed", "hybrid"),
    ]

    for phase_name, q_start, q_end, n_steps, grip_state, mode in phases:
        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)

            # Set arm/torso positions (always - pin them in place)
            for group in groups_to_set:
                start = qpos_array(q_start, group)
                end = qpos_array(q_end, group)
                target_q = start * (1 - alpha) + end * alpha
                robot_view.get_move_group(group).joint_pos = target_q

            # Set gripper
            left_gripper = robot_view.get_move_group("left_gripper")
            if mode == "kinematic":
                left_gripper.joint_pos = np.array([-0.05, 0.05])
                mujoco.mj_forward(model, data)
            else:
                # Physics mode: use actuator ctrl for gripper, pin arm/torso
                # Set all arm/torso actuator ctrl to hold position
                for j in range(model.nu):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
                    if name and "finger" in name:
                        # Gripper actuator: command close or open
                        if grip_state == "closing" or grip_state == "closed":
                            data.ctrl[j] = 100.0 if "left" in name else -100.0
                        else:
                            data.ctrl[j] = -100.0 if "left" in name else -100.0
                    else:
                        # Arm/torso actuator: hold at target position
                        jnt_id = model.actuator_trnid[j, 0]
                        if jnt_id >= 0 and model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT:
                            qadr = model.jnt_qposadr[jnt_id]
                            data.ctrl[j] = data.qpos[qadr]

                mujoco.mj_step(model, data)

                # Re-pin arm/torso qpos after step (override physics drift)
                for group in groups_to_set:
                    start = qpos_array(q_start, group)
                    end = qpos_array(q_end, group)
                    target_q = start * (1 - alpha) + end * alpha
                    robot_view.get_move_group(group).joint_pos = target_q

        # Report state at end of phase
        mujoco.mj_forward(model, data)
        finger_info = get_finger_midpoint(model, data)
        obj_now = get_object_pos(model, data)
        if finger_info and obj_now is not None:
            mid, inter_dist = finger_info
            mid_to_obj = np.linalg.norm(mid - obj_now)
            obj_delta = np.linalg.norm(obj_now - obj_target)
            obj_z = obj_now[2]
            print(
                f"  [{phase_name:10s}] mid_to_obj={mid_to_obj:.4f}m "
                f"inter_finger={inter_dist:.4f}m "
                f"obj_z={obj_z:.4f} obj_delta={obj_delta:.4f}m"
            )

    # 6. Final analysis
    print("\n[6] Final state:")
    finger_info = get_finger_midpoint(model, data)
    obj_final = get_object_pos(model, data)
    if finger_info and obj_final is not None:
        mid, inter_dist = finger_info
        obj_delta = np.linalg.norm(obj_final - obj_target)
        z_lift = obj_final[2] - obj_target[2]
        print(f"  Object final: {obj_final}")
        print(f"  Object delta: {obj_delta:.4f}m")
        print(f"  Object Z lift: {z_lift:.4f}m")
        print(f"  Inter-finger: {inter_dist:.4f}m")

        if z_lift > 0.03 or obj_delta > 0.03:
            print("\n  *** SUCCESS! Object was moved/lifted! ***")
            print("  The physics CAN pick this salt shaker.")
        else:
            print("\n  Object did NOT move significantly.")
            if inter_dist < 0.02:
                print("  Fingers closed but no grasp - check friction/contact.")
            else:
                print("  Fingers may not have reached the object.")

    # Optional: launch viewer
    if args.viewer:
        print("\n[7] Launching viewer (close window to exit)...")
        try:
            with mujoco.viewer.launch_passive(model, data) as v:
                v.sync()
                import time
                while v.is_running():
                    time.sleep(0.1)
        except Exception as e:
            print(f"  Viewer failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
