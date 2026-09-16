"""GROMACS Molecular Dynamics pipeline orchestration module.

Provides high-level orchestration bridging PoliScreen virtual screening sessions
with classical molecular dynamics simulations using the standardized GROMACS pipeline.
"""

from __future__ import annotations

from .md_bridge import (
    find_gromacs_pipeline,
    setup_and_launch_md,
    run_md_from_session,
    MDRunResult,
)

__all__ = [
    "find_gromacs_pipeline",
    "setup_and_launch_md",
    "run_md_from_session",
    "MDRunResult",
]
