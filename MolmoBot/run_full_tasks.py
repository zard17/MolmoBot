"""Full multi-task runs — 100 steps each, ~20 min per task."""
import os
import sys

if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
from mujoco import MjModel, MjData, MjSpec
from scipy.spatial.transform import Rotation as R
import numpy as np

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path, install_uid

print("Loading scene...")
houses = get_procthor_10k_houses(split="val")
house_xml_path = houses["val"][0]["base"]
install_scene_with_objects_and_grasps_from_path(house_xml_path)

spec = MjSpec.from_file(house_xml_path)

robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
robot_file_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
robot_spec = MjSpec.from_file(str(robot_file_path))

FrankaRobot.add_robot_to_scene(
    robot_config, spec, robot_spec,
    prefix=robot_config.robot_namespace,
    pos=[6.8, 9.75],
    quat=R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
)

spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [640, 360]
spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
    pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
    fovy=71.0, resolution=[640, 360], name="robot_0/exo_camera_1",
)

bowl_xml_path = install_uid("Bowl_3")
bowl_spec = MjSpec.from_file(str(bowl_xml_path))
receptacle_frame = spec.worldbody.add_frame(
    pos=[7.1, 10.2, 1.01],
    quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
)
receptacle_frame.attach_body(bowl_spec.worldbody.first_body(), prefix="place_receptacle/")

mug_xml_path = install_uid("Mug_1")
mug_spec = MjSpec.from_file(str(mug_xml_path))
mug_frame = spec.worldbody.add_frame(
    pos=[6.9, 10.35, 1.01],
    quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
)
mug_frame.attach_body(mug_spec.worldbody.first_body(), prefix="mug_obj/")

bottle_xml_path = install_uid("Bottle_1")
bottle_spec = MjSpec.from_file(str(bottle_xml_path))
bottle_frame = spec.worldbody.add_frame(
    pos=[7.2, 10.35, 1.01],
    quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
)
bottle_frame.attach_body(bottle_spec.worldbody.first_body(), prefix="bottle_obj/")

print("Compiling model...")
model: MjModel = spec.compile()
data = MjData(model)
view = FrankaDroidRobotView(data, robot_config.robot_namespace)

view.set_qpos_dict(robot_config.init_qpos)
mujoco.mj_forward(model, data)
for mg_id in view.move_group_ids():
    mg = view.get_move_group(mg_id)
    mg.ctrl = mg.noop_ctrl
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 360, 640)
scene_option = mujoco.MjvOption()
scene_option.sitegroup = 0

def render():
    renderer.update_scene(data, camera="robot_0/exo_camera_1", scene_option=scene_option)
    exo_img = renderer.render()
    renderer.update_scene(data, camera="robot_0/gripper/wrist_camera", scene_option=scene_option)
    cam_img = renderer.render()
    return {"exo_camera_1": exo_img, "wrist_camera": cam_img}

print("Scene ready.")

# --- Load Model ---
print("Loading MolmoBot-DROID model...")
from huggingface_hub import snapshot_download
from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

ckpt_path = snapshot_download("allenai/MolmoBot-DROID")

class MockConfig:
    def __init__(self, policy_config):
        self.policy_config = policy_config

policy_config = RealRobotVLAPolicyConfig()
policy_config.checkpoint_path = ckpt_path
policy_config.action_type = "joint_pos"
policy_config.action_keys["arm"] = "joint_pos"
mock_config = MockConfig(policy_config)
policy = RealRobotVLAPolicy(config=mock_config, task_type="manipulation")
print("Model loaded.")

# --- run_task ---
import time
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def run_task(task, episode_dur=6.6, policy_dt_ms=66, video_filename=None):
    if video_filename is None:
        video_filename = task.replace(" ", "_")[:40] + ".mp4"

    mujoco.mj_resetData(model, data)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)
    policy.reset()

    total_steps = round(episode_dur * 1000 / policy_dt_ms)
    print(f"\n{'='*60}")
    print(f"Task: \"{task}\"")
    print(f"Steps: {total_steps} | episode_dur: {episode_dur}s")
    print(f"{'='*60}")

    frames = []
    t0 = time.time()

    for step in range(total_steps):
        step_t0 = time.time()

        jp = view.get_move_group("arm").joint_pos
        gripper_input = view.get_move_group("gripper").joint_pos
        obs = {
            "task": task,
            "qpos": {"arm": jp, "gripper": gripper_input},
            **render()
        }

        frame = np.hstack([obs["exo_camera_1"], obs["wrist_camera"]])
        frames.append(frame)

        is_inference_step = (step % 8 == 0) or len(policy.action_buffer) == 0 or policy.buffer_index >= len(policy.action_buffer)
        if is_inference_step:
            print(f"  Step {step+1}/{total_steps} | {time.time()-t0:.1f}s elapsed | running inference (~80s on CPU)...", end="", flush=True)

        action = policy.get_action(obs)

        if is_inference_step:
            print(f" done ({time.time()-step_t0:.1f}s)")
        else:
            gripper_val = action.get("gripper", [0])[0]
            print(f"  Step {step+1}/{total_steps} | {time.time()-t0:.1f}s elapsed | gripper: {gripper_val:.0f} | step: {time.time()-step_t0:.1f}s")

        for mg_id in action.keys():
            view.get_move_group(mg_id).ctrl = action[mg_id]

        mujoco.mj_step(model, data, nstep=policy_dt_ms // round(model.opt.timestep * 1000))

    elapsed = time.time() - t0
    print(f"Done! {total_steps} steps in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    video = ImageSequenceClip(frames, fps=15)
    video.write_videofile(video_filename, audio=False, logger=None)
    print(f"Saved: {video_filename}")

# --- Full Runs ---
print("\n" + "#"*60)
print("FULL RUNS — 100 steps each, ~20 min per task")
print("#"*60)

run_task("put the salt shaker in the bowl", episode_dur=6.6, video_filename="task1_pick_and_place.mp4")
run_task("pick up the mug", episode_dur=6.6, video_filename="task2_pick.mp4")
run_task("place the bottle next to the bowl", episode_dur=6.6, video_filename="task3_place_next_to.mp4")

print("\n" + "#"*60)
print("ALL FULL RUNS COMPLETE")
print("#"*60)
