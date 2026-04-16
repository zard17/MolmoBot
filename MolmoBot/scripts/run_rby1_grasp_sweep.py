#!/usr/bin/env python3
"""Sweep grasp orientations and friction to find a successful pick.

Builds on run_rby1_direct_scene_pick.py - runs multiple orientation/friction
combinations to find a stable grasp.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rby1_direct_scene_pick import (
    ASSETS_DIR, SCENE_XML, ROBOT_XML, SALT_SHAKER_XML,
    NAMESPACE, OBJECT_POS, OBJECT_QUAT, EE_TO_FINGERTIP_Z, INIT_QPOS,
    find_body, set_robot_qpos, get_finger_midpoint, get_object_pos,
)
from molmo_spaces.kinematics.rby1_kinematics import RBY1Kinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView


def assemble_scene(friction_multiplier: float = 1.0) -> mujoco.MjModel:
    """Assemble full scene with optional friction boost."""
    scene_content = SCENE_XML.read_text()
    robot_rel = os.path.relpath(ROBOT_XML, SCENE_XML.parent)
    include_line = f'  <include file="{robot_rel}"/>\n'
    memory_line = '  <size memory="128M"/>\n'
    idx = scene_content.index(">") + 1
    scene_content = scene_content[:idx] + "\n" + memory_line + include_line + scene_content[idx:]

    temp_path = SCENE_XML.parent / "_temp_sweep_scene.xml"
    temp_path.write_text(scene_content)
    try:
        spec = mujoco.MjSpec.from_file(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    # Add salt shaker
    obj_spec = mujoco.MjSpec.from_file(str(SALT_SHAKER_XML))
    obj_body = obj_spec.worldbody.bodies[0]
    if not obj_body.first_joint():
        obj_body.add_joint(name="XYZ_jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.0001)

    frame = spec.worldbody.add_frame(pos=OBJECT_POS, quat=OBJECT_QUAT)
    frame.attach_body(obj_body, "", "")

    model = spec.compile()

    # Boost friction on ALL geoms if requested
    if friction_multiplier != 1.0:
        for i in range(model.ngeom):
            model.geom_friction[i, 0] *= friction_multiplier  # sliding
            model.geom_friction[i, 1] *= friction_multiplier  # torsional

    return model


def solve_ik_at_orientation(model, data, robot_view, target_pos, angle_deg):
    """Solve IK at a specific orientation. Returns qpos dict or None."""
    kinematics = RBY1Kinematics(model, data, namespace=NAMESPACE, holo_base=True)

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[-s, c, 0], [c, s, 0], [0, 0, -1]])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = target_pos.copy()
    pose[2, 3] -= EE_TO_FINGERTIP_Z

    q0 = robot_view.get_qpos_dict()
    base_pose = robot_view.base.pose
    return kinematics.ik("left_arm", pose, ["torso", "left_arm"], q0, base_pose,
                         eps=1e-3, max_iter=2000, dt=0.5)


def run_single_pick(model, data, robot_view, ik_grasp, ik_approach, ik_lift, n_close=200, n_lift=200):
    """Execute a single pick attempt. Returns (obj_delta, z_lift, details)."""
    groups = ["torso", "left_arm"]

    def qpos_array(qdict, g):
        return np.array(qdict[g])

    obj_start = get_object_pos(model, data)

    # Kinematic phases: approach, descend, settle
    for q_start, q_end, n in [
        (robot_view.get_qpos_dict(), ik_approach, 40),
        (ik_approach, ik_grasp, 40),
        (ik_grasp, ik_grasp, 10),
    ]:
        for i in range(n):
            alpha = i / max(n - 1, 1)
            for g in groups:
                s, e = qpos_array(q_start, g), qpos_array(q_end, g)
                robot_view.get_move_group(g).joint_pos = s * (1 - alpha) + e * alpha
            robot_view.get_move_group("left_gripper").joint_pos = np.array([-0.05, 0.05])
            mujoco.mj_forward(model, data)

    # Record pre-close finger midpoint
    fi = get_finger_midpoint(model, data)
    obj_pre = get_object_pos(model, data)
    pre_mid_dist = np.linalg.norm(fi[0] - obj_pre) if fi and obj_pre is not None else 999

    # Physics phase: close gripper with arm pinned
    for i in range(n_close):
        # Pin arm/torso
        for g in groups:
            robot_view.get_move_group(g).joint_pos = qpos_array(ik_grasp, g)
        # Gripper via actuator
        for j in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            if name and "left_finger" in name:
                data.ctrl[j] = 100.0  # close
            elif name and "right_finger" in name:
                data.ctrl[j] = -100.0  # keep right open
            else:
                jnt_id = model.actuator_trnid[j, 0]
                if jnt_id >= 0 and model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT:
                    data.ctrl[j] = data.qpos[model.jnt_qposadr[jnt_id]]
        mujoco.mj_step(model, data)
        for g in groups:
            robot_view.get_move_group(g).joint_pos = qpos_array(ik_grasp, g)

    mujoco.mj_forward(model, data)
    obj_after_close = get_object_pos(model, data)
    fi_close = get_finger_midpoint(model, data)
    close_inter = fi_close[1] if fi_close else 999

    # Physics phase: lift with arm pinned
    for i in range(n_lift):
        alpha = i / max(n_lift - 1, 1)
        for g in groups:
            s, e = qpos_array(ik_grasp, g), qpos_array(ik_lift, g)
            robot_view.get_move_group(g).joint_pos = s * (1 - alpha) + e * alpha
        for j in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
            if name and "left_finger" in name:
                data.ctrl[j] = 100.0
            elif name and "right_finger" in name:
                data.ctrl[j] = -100.0
            else:
                jnt_id = model.actuator_trnid[j, 0]
                if jnt_id >= 0 and model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT:
                    data.ctrl[j] = data.qpos[model.jnt_qposadr[jnt_id]]
        mujoco.mj_step(model, data)
        for g in groups:
            s, e = qpos_array(ik_grasp, g), qpos_array(ik_lift, g)
            robot_view.get_move_group(g).joint_pos = s * (1 - alpha) + e * alpha

    mujoco.mj_forward(model, data)
    obj_final = get_object_pos(model, data)

    if obj_final is not None and obj_start is not None:
        obj_delta = np.linalg.norm(obj_final - obj_start)
        z_lift = obj_final[2] - obj_start[2]
    else:
        obj_delta, z_lift = 0.0, 0.0

    return obj_delta, z_lift, {
        "pre_mid_dist": pre_mid_dist,
        "close_inter_finger": close_inter,
        "obj_after_close": obj_after_close.tolist() if obj_after_close is not None else None,
        "obj_final": obj_final.tolist() if obj_final is not None else None,
    }


def main():
    orientations = list(range(0, 180, 15))
    friction_mults = [1.0, 2.0, 5.0]

    print("=" * 70)
    print("RBY1 Grasp Orientation + Friction Sweep")
    print("=" * 70)

    results = []

    for fric_mult in friction_mults:
        print(f"\n{'='*70}")
        print(f"Friction multiplier: {fric_mult}x")
        print(f"{'='*70}")

        model = assemble_scene(friction_multiplier=fric_mult)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        robot_view = set_robot_qpos(model, data, INIT_QPOS)
        mujoco.mj_forward(model, data)

        # Settle
        for j in range(model.nu):
            jnt_id = model.actuator_trnid[j, 0]
            if jnt_id >= 0 and model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT:
                data.ctrl[j] = data.qpos[model.jnt_qposadr[jnt_id]]
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        obj_settled = get_object_pos(model, data)
        settled_qpos = robot_view.get_qpos_dict()
        print(f"  Object settled at: {obj_settled}")

        for angle in orientations:
            # Reset to settled state
            robot_view.set_qpos_dict(settled_qpos)
            mujoco.mj_forward(model, data)

            # Solve IK
            ik_grasp = solve_ik_at_orientation(model, data, robot_view, obj_settled, angle)
            if ik_grasp is None:
                continue

            robot_view.set_qpos_dict(settled_qpos)
            mujoco.mj_forward(model, data)

            approach_target = obj_settled.copy()
            approach_target[2] += 0.15
            ik_approach = solve_ik_at_orientation(model, data, robot_view, approach_target, angle)
            if ik_approach is None:
                ik_approach = settled_qpos

            robot_view.set_qpos_dict(ik_grasp)
            mujoco.mj_forward(model, data)
            lift_target = obj_settled.copy()
            lift_target[2] += 0.15
            ik_lift = solve_ik_at_orientation(model, data, robot_view, lift_target, angle)
            if ik_lift is None:
                ik_lift = ik_grasp

            # Reset fully for pick attempt
            data2 = mujoco.MjData(model)
            mujoco.mj_forward(model, data2)
            robot_view2 = set_robot_qpos(model, data2, INIT_QPOS)
            mujoco.mj_forward(model, data2)
            for j in range(model.nu):
                jnt_id = model.actuator_trnid[j, 0]
                if jnt_id >= 0 and model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT:
                    data2.ctrl[j] = data2.qpos[model.jnt_qposadr[jnt_id]]
            for _ in range(500):
                mujoco.mj_step(model, data2)
            mujoco.mj_forward(model, data2)

            obj_delta, z_lift, details = run_single_pick(
                model, data2, robot_view2, ik_grasp, ik_approach, ik_lift
            )

            success = z_lift > 0.02 or (obj_delta > 0.01 and z_lift > 0.005)
            marker = " *** LIFT! ***" if z_lift > 0.02 else (" *push*" if obj_delta > 0.01 else "")
            print(
                f"  {angle:4d}deg: delta={obj_delta:.4f}m z_lift={z_lift:+.4f}m "
                f"pre_mid={details['pre_mid_dist']:.4f}m "
                f"close_finger={details['close_inter_finger']:.4f}m{marker}"
            )
            results.append({
                "friction": fric_mult,
                "angle": angle,
                "obj_delta": obj_delta,
                "z_lift": z_lift,
                **details,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Best results per friction level")
    print("=" * 70)
    for fric in friction_mults:
        fric_results = [r for r in results if r["friction"] == fric]
        if not fric_results:
            print(f"  friction={fric}x: no IK solutions")
            continue
        best = max(fric_results, key=lambda r: r["z_lift"])
        print(
            f"  friction={fric}x: best angle={best['angle']}deg "
            f"z_lift={best['z_lift']:+.4f}m delta={best['obj_delta']:.4f}m "
            f"close_finger={best['close_inter_finger']:.4f}m"
        )

    any_lift = any(r["z_lift"] > 0.02 for r in results)
    if any_lift:
        print("\n  *** AT LEAST ONE CONFIGURATION ACHIEVED LIFT! ***")
    else:
        any_push = any(r["obj_delta"] > 0.01 for r in results)
        if any_push:
            print("\n  Object was pushed but not lifted. Contact works, grasp quality needs work.")
        else:
            print("\n  No significant object interaction found.")


if __name__ == "__main__":
    main()
