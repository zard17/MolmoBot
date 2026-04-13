#!/usr/bin/env python3
"""Wrapper for serve_molmo.py on R2-blocked environments (e.g., SPACE).

Patches molmo_spaces to use HuggingFace before any imports trigger R2 access.

Usage (from MolmoBot/MolmoBot/):
    python scripts/serve_molmo_hf.py --local-path <path>
    python scripts/serve_molmo_hf.py --hf-repo allenai/MolmoBot-DROID
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch molmo_spaces BEFORE any other imports trigger get_resource_manager()
import molmo_spaces.molmo_spaces_constants as _msc
_msc.USE_HUGGING_FACE = True
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["robots"].pop("franka_cap", None)
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"].pop("rby1_pnp", None)
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["franka_pick"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["franka_pick_and_place"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rby1_door_opening"] = "20250107"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rum_open_close"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rum_pick"] = "20260202"

from launch_scripts.serve_molmo import main

main()
