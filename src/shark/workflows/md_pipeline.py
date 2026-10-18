"""GROMACS Molecular Dynamics pipeline orchestration module.

Provides high-level orchestration bridging PoliScreen virtual screening sessions
with classical molecular dynamics simulations using the standardized GROMACS pipeline.
"""

from __future__ import annotations

from .md_bridge import (
    MD_PRESETS,
    find_gromacs_pipeline,
    gromacs_production_hashes,
    resolve_md_preset,
    resume_md_from_checkpoint,
    setup_and_launch_md,
    run_md_from_session,
    validate_gromacs_production,
    MDRunResult,
)

__all__ = [
    "MD_PRESETS",
    "find_gromacs_pipeline",
    "gromacs_production_hashes",
    "resolve_md_preset",
    "resume_md_from_checkpoint",
    "setup_and_launch_md",
    "run_md_from_session",
    "validate_gromacs_production",
    "MDRunResult",
]
