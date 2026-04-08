#!/usr/bin/env python
"""
Wrapper script to run evaluation with proper GL context initialization and MuJoCo viewer.
MuJoCo requires an OpenGL context to be created before it can initialize rendering.

This script optionally integrates MuJoCo viewer for real-time simulation visualization.
"""

import sys
import os
import argparse
import threading
import time
from typing import Optional, Tuple
from pathlib import Path

import glfw

# Try to import mujoco viewer
try:
    import mujoco
    from mujoco import MjModel, MjData
    import mujoco.viewer
    MUJOCO_VIEWER_AVAILABLE = True
except ImportError:
    print("Warning: MuJoCo viewer not available. Running without visualization.")
    MUJOCO_VIEWER_AVAILABLE = False


class ViewerManager:
    """Manages MuJoCo viewer lifecycle and synchronization."""
    
    def __init__(self):
        self.model: Optional[MjModel] = None
        self.data: Optional[MjData] = None
        self.viewer = None
        self.running = False
        self._lock = threading.Lock()
        
    def set_model_data(self, model: MjModel, data: MjData):
        """Update model and data references for viewer synchronization."""
        with self._lock:
            self.model = model
            self.data = data
            
    def launch(self) -> bool:
        """Launch the MuJoCo viewer in passive mode."""
        if not MUJOCO_VIEWER_AVAILABLE:
            return False
        if self.model is None or self.data is None:
            print("Viewer cannot launch: model/data not set")
            return False
            
        try:
            print("✓ Launching MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.running = True
            return True
        except Exception as e:
            print(f"Failed to launch viewer: {e}")
            return False
            
    def sync(self):
        """Synchronize viewer with current simulation state."""
        with self._lock:
            if self.viewer is not None and self.viewer.is_running():
                try:
                    self.viewer.sync()
                except Exception as e:
                    pass  # Silently ignore sync errors
                    
    def is_running(self) -> bool:
        """Check if viewer is still running."""
        if self.viewer is not None:
            return self.viewer.is_running()
        return False
        
    def close(self):
        """Close the viewer."""
        self.running = False
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None


# Global viewer manager instance
viewer_manager = ViewerManager()


def viewer_sync_loop(sync_interval: float = 0.02):
    """Background thread for continuous viewer synchronization."""
    while viewer_manager.running:
        viewer_manager.sync()
        time.sleep(sync_interval)


def setup_glfw():
    """Initialize glfw and create a hidden window."""
    if not glfw.init():
        print("Failed to initialize glfw")
        return None
        
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, 'Hidden', None, None)
    if not window:
        print("Failed to create glfw window")
        glfw.terminate()
        return None
        
    glfw.make_context_current(window)
    print("✓ Initialized GL context via glfw")
    return window


def patch_simulator_for_viewer():
    """
    Patch molmo_spaces simulator to expose model/data for viewer integration.
    Returns a function that can be called to set up the viewer.
    """
    if not MUJOCO_VIEWER_AVAILABLE:
        return None
        
    try:
        # Attempt to patch the simulator - try multiple possible import paths
        simulator_modules = [
            "molmo_spaces.simulation.simulator",
            "molmo_spaces.simulator",
            "mjthor.simulation.simulator",
            "mjthor.simulator",
        ]
        
        for module_path in simulator_modules:
            try:
                parts = module_path.split(".")
                module = __import__(parts[0], fromlist=parts[1:])
                for part in parts[1:]:
                    module = getattr(module, part)
                
                if hasattr(module, 'Simulator'):
                    original_init = module.Simulator.__init__
                    
                    def make_patched_init(orig):
                        def patched_init(self, *args, **kwargs):
                            # Call original init
                            orig(self, *args, **kwargs)
                            # Store model/data references
                            if hasattr(self, 'model') and hasattr(self, 'data'):
                                viewer_manager.set_model_data(self.model, self.data)
                                # Launch viewer if not already running
                                if viewer_manager.viewer is None:
                                    viewer_manager.launch()
                        return patched_init
                    
                    module.Simulator.__init__ = make_patched_init(original_init)
                    print(f"✓ Patched simulator at {module_path} for viewer integration")
                    return True
            except (ImportError, AttributeError):
                continue
        
        print("Note: Could not find simulator module to patch. Viewer integration may not work automatically.")
        return False
    except Exception as e:
        print(f"Note: Could not auto-patch simulator ({e}). Viewer will be launched manually if model/data available.")
        return False


def try_launch_viewer_from_env():
    """
    Try to launch viewer from environment variables or global state.
    This is a fallback for when simulator patching doesn't work.
    """
    if not MUJOCO_VIEWER_AVAILABLE:
        return False
        
    # Check for environment variables that might contain model/data
    import os
    model_path = os.environ.get('MUJOCO_MODEL_PATH')
    if model_path and os.path.exists(model_path):
        try:
            model = MjModel.from_xml_path(model_path)
            data = MjData(model)
            viewer_manager.set_model_data(model, data)
            return viewer_manager.launch()
        except Exception as e:
            print(f"Failed to load model from env path: {e}")
    return False


def main():
    """Main evaluation entry point with optional MuJoCo viewer integration."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation with MuJoCo viewer integration",
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
    parser.add_argument(
        "--enable_viewer",
        action="store_true",
        help="Enable MuJoCo viewer for real-time visualization (uses molmo_spaces use_passive_viewer)",
    )
    parser.add_argument(
        "--viewer_sync_interval",
        type=float,
        default=0.02,
        help="Viewer sync interval in seconds",
    )
    
    args = parser.parse_args()
    
    # Resolve module:ClassName
    import importlib
    eval_config_cls = args.eval_config_cls
    if isinstance(eval_config_cls, str) and ":" in eval_config_cls:
        module_path, class_name = eval_config_cls.split(":")
        eval_config_cls = getattr(importlib.import_module(module_path), class_name)
    
    # Setup glfw
    window = setup_glfw()
    if window is None:
        sys.exit(1)
    
    # Setup viewer if enabled - patch the config class to enable passive viewer
    viewer_thread = None
    if args.enable_viewer:
        # Patch the config class to enable use_passive_viewer
        # This is the recommended way to enable viewer in molmo_spaces
        try:
            # Create a temporary instance to check if use_passive_viewer exists
            # We need to patch the class before it's instantiated
            original_init = eval_config_cls.__init__ if hasattr(eval_config_cls, '__init__') else None
            
            # The config is created by run_evaluation, so we patch the class attribute directly
            # This is a workaround since we can't modify the external library
            print("✓ MuJoCo viewer enabled via use_passive_viewer config")
            print("  Note: The viewer will be launched by molmo_spaces when the simulator initializes.")
            print("  Close the viewer window to stop the evaluation.")
        except Exception as e:
            print(f"Note: Could not patch config for viewer ({e}).")
    elif args.enable_viewer and not MUJOCO_VIEWER_AVAILABLE:
        print("⚠ Viewer enabled but mujoco viewer not available. Running without visualization.")
    
    try:
        # Import and run evaluation directly
        from molmo_spaces.evaluation.eval_main import run_evaluation
        
        print("✓ Running evaluation...")
        
        # Import run_evaluation to access its internal functions
        from molmo_spaces.evaluation.eval_main import run_evaluation, create_eval_config, JsonEvalRunner
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
        import datetime
        import os
        
        # Validate benchmark and create output directory
        benchmark_dir = Path(args.benchmark_path).resolve()
        checkpoint_path = Path(args.checkpoint_path).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        
        # Load episodes
        episodes = load_all_episodes(benchmark_dir)
        if not episodes:
            raise ValueError(f"No episodes found in benchmark at {benchmark_dir}")
        
        # Create timestamp and output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = eval_config_cls.__name__
        if output_dir is not None:
            resolved_output_dir = output_dir / config_name / timestamp
        else:
            resolved_output_dir = Path("eval_output") / config_name / timestamp
        os.makedirs(resolved_output_dir, exist_ok=True)
        
        # Create experiment config
        exp_config = create_eval_config(
            eval_config_cls=eval_config_cls,
            benchmark_dir=benchmark_dir,
            output_dir=resolved_output_dir,
            checkpoint_path=checkpoint_path,
            task_horizon=args.task_horizon,
            num_workers=args.num_workers,
            camera_config_override=None,
        )
        
        # Enable passive viewer if requested
        if args.enable_viewer:
            exp_config.use_passive_viewer = True
            print("✓ MuJoCo passive viewer enabled")
            print("  Note: The viewer will be launched by molmo_spaces when the simulator initializes.")
            print("  Close the viewer window to stop the evaluation.")
        
        # Apply filament settings
        exp_config.use_filament |= args.use_filament
        if args.environment_light_intensity is not None:
            exp_config.environment_light_intensity = args.environment_light_intensity
        
        # Patch config with evaluation-specific runtime parameters
        exp_config = JsonEvalRunner.patch_config(
            exp_config=exp_config,
            episode_idx=None,
            add_custom_object=False,
            custom_object_path=None,
            custom_object_name=None,
        )
        JsonEvalRunner.adjust_robot(exp_config)
        
        # Run evaluation with the modified config
        from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner
        from molmo_spaces.policy.base_policy import BasePolicy
        
        # Create runner and run evaluation
        runner = JsonEvalRunner(exp_config, benchmark_dir)
        results = runner.run(preloaded_policy=None)
        
        print("✓ Evaluation completed successfully")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up viewer
        viewer_manager.close()
        if viewer_thread is not None:
            viewer_thread.join(timeout=1.0)
            
        # Clean up glfw
        glfw.destroy_window(window)
        glfw.terminate()
        print("✓ Cleaned up glfw resources")


def run_eval_with_manual_viewer():
    """
    Alternative entry point for manual viewer integration.
    This function provides a template for integrating viewer with custom simulator code.
    """
    if not MUJOCO_VIEWER_AVAILABLE:
        print("MuJoCo viewer not available")
        return
        
    print("=== Manual Viewer Integration Template ===")
    print("This template shows how to integrate MuJoCo viewer with custom simulator code.")
    print()
    print("Example usage:")
    print("  from mujoco import MjModel, MjData")
    print("  import mujoco.viewer")
    print()
    print("  # Load your model and data")
    print("  model = MjModel.from_xml_path('your_scene.xml')")
    print("  data = MjData(model)")
    print()
    print("  # Launch viewer")
    print("  with mujoco.viewer.launch_passive(model, data) as viewer:")
    print("      while viewer.is_running():")
    print("          # Step your simulation")
    print("          mujoco.mj_step(model, data)")
    print("          # Sync viewer")
    print("          viewer.sync()")


if __name__ == "__main__":
    main()