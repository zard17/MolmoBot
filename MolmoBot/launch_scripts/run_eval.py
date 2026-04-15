"""
Run SynthVLA evaluation using molmo_spaces.

This script is the entry point invoked by gantry for evaluation jobs.
It calls run_evaluation from molmo_spaces with the provided arguments.

Usage:
    python launch_scripts/run_eval.py \
        --checkpoint_path /path/to/checkpoint \
        --benchmark_path /path/to/benchmark \
        --eval_config_cls olmo.eval.configure_molmo_spaces:SynthVLAFrankaBenchmarkEvalConfig
"""

import argparse
import importlib
import json
import shutil
from pathlib import Path


def _register_local_rby1_custom_scene(benchmark_dir: Path) -> None:
    """Register the repo-local RBY1 custom desk scene for JSON eval.

    MolmoSpaces scene loading expects scene XMLs to live under
    ``MLSPACES_ASSETS_DIR/scenes`` and to be discoverable through
    ``molmo_spaces_constants.get_scenes``. The RBY1 freeze benchmark is a small
    local scene used only for this repo, so we register it at runtime instead of
    requiring a packaged MolmoSpaces resource archive.
    """
    benchmark_json = benchmark_dir / "benchmark.json"
    if not benchmark_json.is_file():
        return

    try:
        episodes = json.loads(benchmark_json.read_text())
    except json.JSONDecodeError:
        return

    if not any(ep.get("scene_dataset") == "rby1-custom" for ep in episodes):
        return

    import molmo_spaces.molmo_spaces_constants as msc
    import molmo_spaces.utils.lazy_loading_utils as lazy_utils

    repo_root = Path(__file__).resolve().parents[1]
    source_scene = benchmark_dir / "custom_scene.xml"
    source_metadata = benchmark_dir / "custom_scene_metadata.json"
    if not source_scene.is_file():
        source_scene = repo_root / "benchmarks" / "rby1_pickpnp_benchmark" / "custom_scene.xml"
    if not source_metadata.is_file():
        source_metadata = (
            repo_root / "benchmarks" / "rby1_pickpnp_benchmark" / "custom_scene_metadata.json"
        )

    scene_dir = msc.ASSETS_DIR / "scenes" / "rby1-custom"
    scene_dir.mkdir(parents=True, exist_ok=True)
    target_scene = scene_dir / "custom_scene.xml"
    target_metadata = scene_dir / "custom_scene_metadata.json"
    shutil.copyfile(source_scene, target_scene)
    if source_metadata.is_file():
        shutil.copyfile(source_metadata, target_metadata)
    elif not target_metadata.exists():
        target_metadata.write_text('{"objects": {}}\n')

    original_get_scenes = msc.get_scenes

    def get_scenes(dataset_name: str, split: str = "train", return_version: bool = False):
        if dataset_name == "rby1-custom":
            scene_map = {split: {0: {"base": str(target_scene), "ceiling": str(target_scene)}}}
            if return_version:
                return scene_map, None
            return scene_map
        return original_get_scenes(dataset_name, split=split, return_version=return_version)

    msc.get_scenes = get_scenes

    original_install_scene_from_path = lazy_utils.install_scene_from_path

    def install_scene_from_path(xml_path):
        try:
            rel_parts = Path(xml_path).relative_to(msc.get_scenes_root()).parts
        except ValueError:
            rel_parts = ()
        if rel_parts and rel_parts[0] == "rby1-custom":
            return {}
        return original_install_scene_from_path(xml_path)

    lazy_utils.install_scene_from_path = install_scene_from_path


def main():
    parser = argparse.ArgumentParser(
        description="Run SynthVLA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to the benchmark directory",
    )
    parser.add_argument(
        "--eval_config_cls",
        type=str,
        required=True,
        help="Evaluation config class (module:ClassName)",
    )
    parser.add_argument(
        "--task_horizon",
        type=int,
        default=600,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for eval results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel eval workers",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mjthor-online-eval",
        help="WandB project name",
    )
    parser.add_argument(
        "--use_filament",
        action="store_true",
        help="Use filament renderer instead of legacy OpenGL",
    )
    parser.add_argument(
        "--environment_light_intensity",
        type=float,
        default=None,
        help="Intensity of default environmental light (filament only)",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_path)
    _register_local_rby1_custom_scene(benchmark_dir)

    from molmo_spaces.evaluation.eval_main import run_evaluation

    # Resolve module:ClassName string to actual class so mujoco-thor uses __name__
    # (not the full "module:ClassName" string) when constructing the output directory.
    eval_config_cls = args.eval_config_cls
    if isinstance(eval_config_cls, str) and ":" in eval_config_cls:
        module_path, class_name = eval_config_cls.split(":")
        eval_config_cls = getattr(importlib.import_module(module_path), class_name)

    results = run_evaluation(
        eval_config_cls=eval_config_cls,
        benchmark_dir=benchmark_dir,
        checkpoint_path=Path(args.checkpoint_path),
        task_horizon_steps=args.task_horizon,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_filament=args.use_filament,
        environment_light_intensity=args.environment_light_intensity,
    )

    print(f"Success rate: {results.success_rate:.1%}")
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")


if __name__ == "__main__":
    main()

