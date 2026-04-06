#!/usr/bin/env python
"""
Wrapper script to run evaluation with proper GL context initialization.
MuJoCo requires an OpenGL context to be created before it can initialize rendering.
"""

import sys
import os
import glfw

# Initialize glfw and create a hidden window BEFORE importing anything that might use mujoco
if not glfw.init():
    print("Failed to initialize glfw")
    sys.exit(1)

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(640, 480, 'Hidden', None, None)
if not window:
    print("Failed to create glfw window")
    glfw.terminate()
    sys.exit(1)

glfw.make_context_current(window)
print("✓ Initialized GL context via glfw")

try:
    # Import the evaluation script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'launch_scripts'))
    from run_eval import main as eval_main
    
    print("✓ Running evaluation...")
    # Run the evaluation
    eval_main()
    print("✓ Evaluation completed successfully")
    
except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Clean up glfw
    glfw.destroy_window(window)
    glfw.terminate()
    print("✓ Cleaned up glfw resources")