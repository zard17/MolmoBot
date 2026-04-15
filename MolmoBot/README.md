<div align="center">
  <img src="../assets/MolmoBot.png" alt="MolmoBot Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/MolmoBot/blob/main/MolmoBot/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://allenai.org/blog/molmobot">
    <img alt="Blog Post" src="https://img.shields.io/badge/Molmobot-blog-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2603.16861">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2603.16861-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmobot-models">
    <img alt="Molmobot Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmobot-data">
    <img alt="Molmobot Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-yellow">
  </a>
</p>

# Installation

Clone this repo and install the MolmoBot package:
```bash
git clone https://github.com/allenai/MolmoBot.git <code_path>
cd <code_path>/MolmoBot/MolmoBot
uv sync --extra eval
```

# Franka Static Manipulation

## Running the Franka Policy

To eval our official MolmoBot-DROID checkpoint, simply run
```bash
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID
```

## Demo notebook

Try playing around with MolmoBot-DROID! Use this [demo notebook](demo_policy.ipynb) to create a scene, load a policy, and roll out an episode.
Modify the scene and task to see how the policy behaves!

## Running the Sim Eval for Franka
To run simulation evals, use:
```bash
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu  # optional, used to reduce MolmoSpaces VRAM usage
python launch_scripts/run_eval.py \
  --checkpoint_path <model_path> \
  --benchmark_path <molmo_space_benchmark> \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
  --task_horizon 600  # 600 for Pick and place
```
`<molmo_space_benchmark>` is the path to the MolmoSpaces benchmark to evaluate on. Furthermore, `--use_filament` should be set to true or false depending on if the optional filement renderer is installed.
For further information, see the MolmoSpaces documentation.

To evaluate the official MolmoBot-DROID model, download it and provide to checkpoint_path.

## Training Franka Molmobot with init Molmo2-4B

MolmoBot-Data can be downloaded from Huggingface with [this script](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/bulk_download.py).

To download the dataset used to train MolmoBot-DROID:

```bash
python /path/to/bulk_download.py --config FrankaPickAndPlaceOmniCamConfig --part 1 --split all <data_root>
python /path/to/bulk_download.py --config FrankaPickOmniCamConfig --part 0 --split all <data_root>
python /path/to/bulk_download.py --config FrankaPickAndPlaceOmniCamConfig --part 0 --split all <data_root>
python /path/to/bulk_download.py --config FrankaPickAndPlaceNextToOmniCamConfig --part 2 --split all <data_root>
python /path/to/bulk_download.py --config FrankaPickAndPlaceColorOmniCamConfig --part 0 --split all <data_root>
```

The following command can be used to replicate the training process for MolmoBot-DROID.

```bash
# Install the train dependencies
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
uv sync --extra train

# Select a location for the Molmo2 model. We use the place holder <model_path>
cd <model_path>
wget https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar
tar -xvf Molmo2-4B.tar

# By default the weights are sharded. Unshard them using -
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
python launch_scripts/convert_to_unsharded.py <model_path>/Molmo2-4B <model_path>/Molmo2-4B-unsharded

# Set for stable training
export OLMO_SHARED_FS=1 
export OMP_NUM_THREADS=8 
export LOG_FILTER_TYPE="rank0_only" 
export TORCH_LOGS_RANK0="recompiles,graph_breaks" 
export OLMO_NUM_THREADS_ENV_VAR=8 
export NCCL_TIMEOUT_MINUTES=60 

# Train with the unsharded Molmo2 weights -
torchrun --nnodes=1 --nproc-per-node=8 --master_port=29401 \
  launch_scripts/train_molmobot.py <model_path>/Molmo2-4B-unsharded \
  --stats_path=<stats_path> \
  --action_preset franka_joint \
  --camera_preset franka_one_random_then_wrist \
  --wandb.name=<wandb_name> \
  --wandb.entity=<wandb_entity> \
  --wandb.project=<wandb_project> \
  --seq_len=528 \
  --max_duration=200000 \
  --device_batch_size=32 \
  --global_batch_size=1024 \
  --log_interval=20 \
  --save_folder=<save_folder> \
  --model.mm_preprocessor.image.crop_mode=resize \
  --model.mm_preprocessor.max_frames=1 \
  --weighted_sampling \
  --randomize_prompts \
  --ft_embedding=ae \
  --model.mm_preprocessor.image.max_images=2 \
  --model.num_flow_timestamps=8 \
  --ft_llm=True \
  --img_aug \
  --furthest_camera_prob=0.5 \
  --data_paths <data_root>/FrankaPickAndPlaceOmniCamConfig/part1/train <data_root>/FrankaPickOmniCamConfig/part0/train <data_root>/FrankaPickAndPlaceOmniCamConfig/part0/train <data_root>/FrankaPickAndPlaceNextToOmniCamConfig/part2/train <data_root>/FrankaPickAndPlaceColorOmniCamConfig/part0/train \
  --dataset_sample_rates 0.35 0.2 0.1 0.2 0.15

```

Note that the above training run was used with torchrun to train on 8-64 H100s, so you will need to scale as necessary.

The code base supports micro-batching. To train on smaller GPUs, reduce the device_batch_size and the model will handle it.

## Fine-tuning Molmobot-Img to use 2 frames

```bash
# If you have a trained Molmobot model locally
export IMG_MODEL_PATH=<molmobot_ckpt>

# OR if you want to train the provide Molmobot-Img-DROID model - Get the model from hugging face
hf download allenai/MolmoBot-Img-DROID --local-dir <hf_droid_img_model_path>
export IMG_MODEL_PATH=<hf_droid_img_model_path>

cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate

# Set for stable training
export OLMO_SHARED_FS=1 
export OMP_NUM_THREADS=8 
export LOG_FILTER_TYPE="rank0_only" 
export TORCH_LOGS_RANK0="recompiles,graph_breaks" 
export OLMO_NUM_THREADS_ENV_VAR=8 
export NCCL_TIMEOUT_MINUTES=60 

# Train with the unsharded Molmo2 weights -
torchrun --nnodes=1 --nproc-per-node=8 --master_port=29401 \
  launch_scripts/train_molmobot.py $IMG_MODEL_PATH \
  --model.mm_preprocessor.image.max_images=4 \
  --seq_len=928 \
  --n_obs_steps=2 \
  --obs_step_delta=8 \
  --model.mm_preprocessor.image.single_frame=False \
   --reset_trainer_state \
   --reset_optimizer_state \
  --stats_path=<stats_path> \
  --action_preset franka_joint \
  --camera_preset franka_one_random_then_wrist \
  --wandb.name=<wandb_name> \
  --wandb.entity=<wandb_entity> \
  --wandb.project=<wandb_project> \
  --max_duration=50000 \
  --device_batch_size=32 \
  --global_batch_size=1024 \
  --log_interval=20 \
  --save_folder=<save_folder> \
  --model.mm_preprocessor.image.crop_mode=resize \
  --model.mm_preprocessor.max_frames=1 \
  --weighted_sampling \
  --randomize_prompts \
  --ft_embedding=ae \
  --model.num_flow_timestamps=8 \
  --ft_llm=True \
  --img_aug \
  --furthest_camera_prob=0.5 \
  --data_paths <data_root>/FrankaPickAndPlaceOmniCamConfig/part1/train <data_root>/FrankaPickOmniCamConfig/part0/train <data_root>/FrankaPickAndPlaceOmniCamConfig/part0/train <data_root>/FrankaPickAndPlaceNextToOmniCamConfig/part2/train <data_root>/FrankaPickAndPlaceColorOmniCamConfig/part0/train \
  --dataset_sample_rates 0.35 0.2 0.1 0.2 0.15

```


# RBY1 Mobile Manipulation

To eval our RBY1 checkpoints on robot, run:
```bash
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
uv sync --extra eval

python launch_scripts/serve_molmobot_rby1_multitask.py \
  --task_type door_plus_open \
  --hf_repo allenai/MolmoBot-RBY1Multitask
```

To train RBY1 mobile manipulation models, use `launch_scripts/train_molmobot.py` with:
- `--action_preset RBY1_door_opening` for door opening
- `--action_preset RBY1_multitask` for multitask
- `--camera_preset RBY1_full_with_head_gopro`

MolmoBot-Data can be downloaded from Huggingface with [this script](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/bulk_download.py). Relevant RBY1 configs include `DoorOpeningDataGenConfig` (door opening expert), `RBY1OpenDataGenConfig` (general opening tasks), `RBY1PickDataGenConfig`, and `RBY1PickAndPlaceDataGenConfig`.

Downloaded trajectories should be post-processed with `validate_trajectories.py` to create `valid_trajectory_index.json` before training.

For example:
```bash
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
uv sync --extra train

torchrun --nnodes=1 --nproc-per-node=8 --master_port=29401 \
  launch_scripts/train_molmobot.py ckpts/molmobot/MolmoBot-RBY1Multitask \
  --stats_path=<stats_path> \
  --action_preset RBY1_multitask \
  --camera_preset RBY1_full_with_head_gopro \
  --wandb.name=<wandb_name> \
  --wandb.entity=<wandb_entity> \
  --wandb.project=<wandb_project> \
  --seq_len=1024 \
  --max_duration=100000 \
  --device_batch_size=16 \
  --global_batch_size=256 \
  --save_folder=<save_folder> \
  --data_paths <data_root>/RBY1OpenDataGenConfig/part0
```

To run simulation evals for RBY1, use:
```bash
cd <code_path>/MolmoBot/MolmoBot
. .venv/bin/activate
uv sync --extra eval

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
python launch_scripts/run_eval.py \
  --checkpoint_path <model_path> \
  --benchmark_path <benchmark_dir> \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorEvalConfig
```

Other released RBY1 eval configs are:
- `olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig`
- `olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig`

## RBY1 Frozen Base Test

Compare RBY1 pick performance with the mobile base active (default) vs frozen
(base actions zeroed). The generated benchmark uses a controlled custom desk
scene and the Thor `Salt_Shaker_1` object, which is the same pickup object used
in the working Franka batch evaluation path.

The default script now runs one paired episode by default:

- `default`: mobile base active
- `frozen_base`: model still predicts base actions, but they are zeroed before execution
- `task_horizon`: 400 steps
- `num_episodes`: 1

### Quick start (RunPod)

```bash
bash scripts/run_rby1_freeze_test.sh
```

The script handles asset downloads, benchmark generation, checkpoint discovery,
and runs both conditions automatically. It uses OSMesa headless rendering by
default, so `--use_filament` is not required for the standard headless pod run.

### Options

```bash
# Short smoke test
bash scripts/run_rby1_freeze_test.sh --task_horizon 1 --num_episodes 1 --only default

# Shorter full episodes
bash scripts/run_rby1_freeze_test.sh --task_horizon 200

# Run only one condition
bash scripts/run_rby1_freeze_test.sh --only default
bash scripts/run_rby1_freeze_test.sh --only frozen

# Custom checkpoint
bash scripts/run_rby1_freeze_test.sh --checkpoint_path /path/to/checkpoint
```

### Benchmark Details

`generate_rby1_pickpnp_benchmark.py` creates
`benchmarks/rby1_pickpnp_benchmark/benchmark.json` with:

- `scene_dataset`: `rby1-custom`
- `house_index`: `0`
- pickup object: `/Salt_Shaker_1`
- pickup poses: a small salt-shaker reach grid near the RBY1 left-arm approach
  path, currently centered around table coordinates `x=0.45..0.52`,
  `y=0.25..0.35`, `z=0.79`
- goal poses: same `x/y`, with `z=0.99`
- debug exocentric camera: `exo_camera_1`

The custom scene XML is mirrored under
`$MLSPACES_ASSETS_DIR/scenes/rby1-custom/custom_scene.xml` at runtime so the
MolmoSpaces scene loader accepts it as a normal scene asset.

`exo_camera_1` is a fixed front-left third-person recording camera for visual
debugging. It is not included in the RBY1 policy `camera_names`, so it does not
change the model inputs.

### Gripper Mapping

RBY1 pick/pnp evals clamp model gripper output before sending it to the
simulator. The model action is trained in a `[0, 255]`-style convention, while
the RBY1 simulator command uses large signed motor targets that saturate at the
joint limits. Thresholding at `5.0` maps model-open values to `-100.0` and
model-close values to `100.0`, which correspond to the simulator's open and
closed directions respectively. Pick/pnp does not apply an extra gripper
inversion.

### Latest Included Results

The branch includes result videos from a completed one-episode 400-step run
after the RBY1 pick/pnp gripper mapping fix:

- `eval_output/rby1_freeze_test/default/MolmoBotRBY1PickPnPEvalConfig/20260415_001457/house_0/`
- `eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_002027/house_0/`

Both conditions ran to 400 steps and saved videos. Both reported `fail`. The
saved trajectories confirm the left gripper command now closes the simulated
fingers from about `0.10 m` inter-finger distance to near `0.0 m`, but the
grasp-state sensor still reports no touching or holding. Video inspection shows
the arm reaches near the desk, but the salt shaker remains on the table.

The branch also includes a six-pose frozen-base diagnostic run:

- `eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_011321/house_0/`

Summarize saved trajectories with:

```bash
python scripts/summarize_rby1_eval.py \
  eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_011321
```

That run reached closest at pose `[0.45, 0.25, 0.79]`, with minimum
TCP-object distance `0.0953 m`; none of the six episodes touched, held, or moved
the salt shaker.

### Eval Configs

- `MolmoBotRBY1PickPnPEvalConfig`: default, base active
- `MolmoBotRBY1PickPnPFrozenBaseEvalConfig`: frozen base, base actions zeroed

# Batch Evaluation (Franka)

Run batch evaluation across multiple object combinations and scene types for Franka:

```bash
# Generate default config
python scripts/run_batch_eval.py --generate-config

# Run on RunPod
bash scripts/run_batch.sh

# Or run directly
python scripts/run_batch_eval.py --checkpoint_path <path>

# Quick test (10 steps per episode)
python scripts/run_batch_eval.py --checkpoint_path <path> --task_horizon_override 10
```

See `scripts/EVAL_GUIDE.md` for detailed setup instructions.
