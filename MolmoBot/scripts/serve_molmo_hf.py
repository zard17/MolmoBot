#!/usr/bin/env python3
"""Wrapper for serve_molmo.py on R2-blocked environments (e.g., SPACE).

Patches molmo_spaces to use HuggingFace before any imports trigger R2 access.

Usage:
    python scripts/serve_molmo_hf.py --local-path <path>
    python scripts/serve_molmo_hf.py --hf-repo allenai/MolmoBot-DROID
"""

import scripts.patch_molmo_spaces_hf  # noqa: F401 — must be first

from launch_scripts.serve_molmo import main

main()
