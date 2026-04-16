# RBY1 Pick Training Data Generation Guide

## Prerequisites

- Ubuntu machine with NVIDIA GPU (RTX 4080 16GB sufficient)
- molmospaces repo checked out with CuRobo installed:
  ```bash
  cd ~/simul/molmospaces
  uv sync --extra curobo
  ```

## Step 1: Verify CuRobo

```bash
cd ~/simul/molmospaces
python -c "from curobo.types.math import Pose; print('CuRobo OK')"
```

## Step 2: Verify Data Generation Config

```bash
python -c "
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import RBY1PickDataGenConfig
config = RBY1PickDataGenConfig()
print(f'Policy config ready: {config.policy_config is not None}')
print(f'Task horizon: {config.task_horizon}')
print(f'Output dir: {config.output_dir}')
"
```

## Step 3: Generate Training Data

### Small test run (1 house, 2 episodes)

```bash
python -c "
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import RBY1PickDataGenConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from pathlib import Path

config = RBY1PickDataGenConfig()
config.num_workers = 1
config.task_sampler_config.samples_per_house = 2
config.task_sampler_config.house_inds = [0]
config.output_dir = Path('/mnt/data/rby1_pick_test')
config.filter_for_successful_trajectories = True
config.seed = 42

runner = ParallelRolloutRunner(config)
success, total = runner.run()
print(f'{success}/{total} successful episodes')
"
```

### Full data generation (~100 trajectories)

```bash
python -c "
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import RBY1PickDataGenConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from pathlib import Path

config = RBY1PickDataGenConfig()
config.num_workers = 1                                 # single GPU
config.task_sampler_config.samples_per_house = 10      # episodes per scene
config.task_sampler_config.house_inds = list(range(10)) # 10 scenes
config.output_dir = Path('/mnt/data/rby1_pick_training_data')
config.filter_for_successful_trajectories = True
config.seed = 42

runner = ParallelRolloutRunner(config)
success, total = runner.run()
print(f'{success}/{total} successful episodes')
"
```

Estimated time: 5-30 seconds per episode, ~3-20 hours total for 100 trajectories.

### Adjust output directory

Change `/mnt/data/...` to wherever you have disk space. Each trajectory includes
MP4 video files (~100-500MB per house), so plan for ~10-50GB total.

## Step 4: Verify Torso Movement in Generated Data

This is critical — if CuRobo keeps the torso at zero in the generated data,
the fine-tuning won't help.

```bash
python -c "
import h5py, json, numpy as np, glob

h5_files = sorted(glob.glob('/mnt/data/rby1_pick_training_data/house_*/trajectories_batch_*.h5'))
print(f'Found {len(h5_files)} H5 files')

for h5_path in h5_files[:3]:
    print(f'\n{h5_path}:')
    f = h5py.File(h5_path, 'r')
    for key in list(f.keys())[:3]:
        traj = f[key]
        n = traj['obs/agent/qpos'].shape[0]
        # Check torso at start and midpoint
        for step in [0, n//2, n-1]:
            raw = bytes(traj['obs/agent/qpos'][step].tolist()).split(b'\x00')[0]
            if raw:
                qpos = json.loads(raw.decode('utf-8'))
                torso = [round(v, 3) for v in qpos.get('torso', [])]
                print(f'  {key} step {step}/{n}: torso={torso}')
    f.close()
"
```

**Expected:** Torso values should vary across the trajectory (not all zeros).
If torso stays at zero, the CuRobo planner isn't using the torso for pick tasks
and we'll need the IK-based data generation fallback.

## Step 5: Post-Process for Training

```bash
cd ~/simul/molmobot-phase1/MolmoBot
python scripts/validate_trajectories.py /mnt/data/rby1_pick_training_data
```

This creates `valid_trajectory_index.json` required by the training pipeline.

## Step 6: Compute Normalization Stats

```bash
python -c "
from olmo.train_init_utils import compute_state_action_normalization_stats
from olmo.data.robot_processing import RobotProcessorConfig

camera_names = ['wrist_camera_r', 'head_camera', 'wrist_camera_l']
action_move_groups = ['base', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper', 'torso']
action_spec = {'base': 3, 'left_arm': 7, 'left_gripper': 1, 'right_arm': 7, 'right_gripper': 1, 'torso': 1}
action_keys = {
    'base': 'joint_pos_rel', 'left_arm': 'joint_pos_rel',
    'left_gripper': 'joint_pos', 'right_arm': 'joint_pos_rel',
    'right_gripper': 'joint_pos', 'torso': 'joint_pos',
}

stats, action_norm, state_norm = compute_state_action_normalization_stats(
    data_paths=['/mnt/data/rby1_pick_training_data'],
    camera_names=camera_names,
    action_move_group_names=action_move_groups,
    action_spec=action_spec,
    action_keys=action_keys,
    action_horizon=16,
    input_window_size=1,
    num_workers=2,
)

proc_cfg = RobotProcessorConfig.from_stats(
    stats_by_repo={'synthmanip': stats},
    default_repo_id='synthmanip',
    action_norm_mode=action_norm,
    state_norm_mode=state_norm,
)
proc_cfg.save('/mnt/data/rby1_pick_training_data/norm_stats.yaml')
print('Saved normalization stats')
"
```

## Output Structure

```
/mnt/data/rby1_pick_training_data/
├── house_0/
│   ├── trajectories_batch_1_of_1.h5
│   ├── episode_00000000_head_camera.mp4
│   ├── episode_00000000_wrist_camera_l.mp4
│   ├── episode_00000000_wrist_camera_r.mp4
│   └── ...
├── house_1/
│   └── ...
├── valid_trajectory_index.json  (from step 5)
└── norm_stats.yaml              (from step 6)
```

## Troubleshooting

### CuRobo import fails
```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall
cd ~/simul/molmospaces
uv pip install --force-reinstall "nvidia-curobo @ git+https://github.com/allenai/curobo.git@417c995647fcb173a2bc094d1284b2a4f4b000ad"
```

### Policy config is None
GPU not detected during config init. Check `nvidia-smi` and CUDA setup.

### Asset download failures
```bash
export MLSPACES_USE_HF=1  # Use HuggingFace for asset downloads
```

### Out of GPU memory
Reduce `num_workers` to 1 and `samples_per_house` to 1 for testing.
