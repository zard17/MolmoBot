"""Patch molmo_spaces to use HuggingFace instead of R2 for asset downloads.

Import this module BEFORE any other molmo_spaces imports:

    import scripts.patch_molmo_spaces_hf  # noqa: F401 — side-effect import
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    ...

Or set MLSPACES_USE_HF=1 and call apply_hf_patch() conditionally.

Background: molmo_spaces hardcodes USE_HUGGING_FACE = False and calls
get_resource_manager() at module level (via texture.py import chain),
which contacts R2. On environments where R2 is blocked (e.g., SPACE),
this causes a 403 error even when only serving a local checkpoint.
"""

import molmo_spaces.molmo_spaces_constants as _msc

# Some assets/versions on R2 don't exist on HuggingFace — downgrade to versions that do
_msc.USE_HUGGING_FACE = True
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["robots"].pop("franka_cap", None)
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"].pop("rby1_pnp", None)
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["franka_pick"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["franka_pick_and_place"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rby1_door_opening"] = "20250107"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rum_open_close"] = "20260202"
_msc.DATA_TYPE_TO_SOURCE_TO_VERSION["test_data"]["rum_pick"] = "20260202"
