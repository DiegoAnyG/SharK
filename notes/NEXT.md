# Next session — start here

Written 2026-09-06. Delete sections as they are done; this file is a handover, not a record.
The record is `LOG.md`.

## State at handover

| Component | State |
|---|---|
| Project root | `/home/diego/SharK` |
| Working notes | Complete setup in `notes/` (`RULES.md`, `README.md`, `OBJECTIVES.md`, etc.) |
| CLAUDE.md | Present at root enforcing the 7 unbreakable rules |
| Package structure | Initialized (`pyproject.toml`, `src/shark/core/`, `src/shark/analysis/`, `src/shark/reports/`) |
| Parser & Thermo | Fully functional and unit-tested (`tests/test_parser_and_thermo.py`, 4 passed) |
| Visualizer & Reports | Generates 300 DPI Boltzmann & IR plots, CSV summary, Markdown & HTML reports |
| Benchmark run | Verified on benzofuroxan tautomers: `reports/benzofuroxan_tautomers/` |

---

## 1. Implement automated Tautomer Enumeration & Input Generator

Refactor the preparation stage (`prepare_dft_tautomers.py`) into a clean module in `src/shark/workflows/tautomers.py`:
- Input: arbitrary SMILES or 2D molecule.
- Tautomer generation: RDKit `TautomerEnumerator`.
- 3D conformer ensemble generation with ETKDGv3 and MMFF94s energy ranking (and random seeding for reproducibility).
- Automatic generation of validated ORCA input files (`.inp`) with presets (`fast-screening`, `publication-quality`, custom).
- Zero hardcoded personal paths (strict compliance with `RULES.md` Rule 2 & 3).

## 2. Implement GPU-Accelerated Pre-Filtering (RTX 5070 Ti)

Build an optional acceleration module using Machine Learning Potentials (e.g. AIMNet2 via PyTorch CUDA):
- Rapidly optimize conformer ensembles (100+ conformers in seconds on the RTX 5070 Ti).
- Filter out duplicate conformers (RMSD clustering).
- Select only the top 2–3 unique minima to feed into ORCA for final DFT + analytical Hessian on the 8 CPU cores.
- Keep automatic CPU fallback for systems without a CUDA GPU.

## 3. Implement Subprocess Execution Manager

Build `src/shark/core/runner.py`:
- Spawns ORCA with proper process group isolation (`os.setsid`).
- Handles interrupts (`Ctrl+C`, SIGINT) cleanly to terminate MPI and child binaries (`orca_opt`, `orca_gstep`).
- Manages scratch directory cleanup and honors `SHARK_ORCA`, `SHARK_SCRATCH`, `SHARK_NPROCS`, `SHARK_MAXCORE`.
