# ACT Bridge: MolmoBot ↔ Isaac Sim Integration

Reference skeleton for bridging the MolmoBot RBY1 policy server with an ACT (Isaac Sim) evaluation environment via ROS 2.

## Architecture

```
Isaac Sim (RBY1 scene)
  ├─ /rby1/head_camera/image_raw          ─┐
  ├─ /rby1/left_wrist_camera/image_raw     ├─ ROS 2 ─→ BridgeNode ─→ ws://host:8000 ─→ MolmoBot Server
  ├─ /rby1/right_wrist_camera/image_raw    │                                                   │
  └─ /rby1/joint_states                   ─┘                                                   │
                                                                                                │
  /rby1/joint_commands  ←──────────────── ROS 2 ←── BridgeNode ←── ws://host:8000 ←────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All mapping constants (camera topics, joint names, gripper ranges) |
| `bridge_node.py` | ROS 2 node skeleton — subscribes, calls MolmoBot, publishes commands |
| `isaac_sim_helpers.py` | Scene setup stubs (URDF load, cameras, joint publisher) |

## Prerequisites

- MolmoBot server running (`serve_molmobot_rby1_multitask.py`)
- Isaac Sim with ROS 2 bridge enabled
- RBY1 URDF loaded into Isaac Sim
- ROS 2 Humble (or compatible)
- `olmo.eval.websocket_client` importable (from MolmoBot repo)

## Quick Start

```bash
# 1. Start MolmoBot server (on GPU machine)
cd MolmoBot
python launch_scripts/serve_molmobot_rby1_multitask.py \
    --hf-repo allenai/MolmoBot-RBY1Multitask \
    --task_type pick_pnp --port 8000

# 2. Test client connection (no ROS needed)
python -m olmo.eval.test_websocket_client --host <gpu-host> --port 8000

# 3. Launch bridge node (in ROS 2 environment)
# TODO: ros2 run act_bridge bridge_node
```

## What needs to be done

1. **Confirm joint names** — Load RBY1 URDF into Isaac Sim, print joint names, update `config.py`
2. **Fill in `bridge_node.py`** — Uncomment ROS 2 imports and subscriber/publisher/timer code
3. **Fill in `isaac_sim_helpers.py`** — Add Isaac Sim API calls for URDF loading and camera setup
4. **Package as ROS 2 node** — Create `package.xml`, `setup.py`, `launch/bridge.launch.py`
