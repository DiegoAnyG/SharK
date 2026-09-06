# Changes

Narrative record of changes by session: what was altered, why, and what was verified.

---

## 2026-09-06 (session 4) — FMO $\pi$-nodal plane resolution & FTIR layout overhaul

- **Orbital Projection Fix (`src/shark/reports/orbitals.py`)**: Fixed the root cause of the missing lobes in Tautomer 3. Benzofuroxan HOMO/LUMO are $\pi/\pi^*$ orbitals where the molecular plane is an antisymmetric nodal plane ($\psi(x, y, -z) = -\psi(x, y, z)$). The previous $Z$-midplane averaging integrated odd functions across the nodal plane, causing severe destructive cancellation for planar geometries ($z \approx 0$). Slicing at $+0.70\text{ \AA}$ (the radial maximum of $2p_z$ lobes) completely eliminates cancellation, restoring rich, balanced phase lobes ($\psi \in [-0.18, +0.12]$) across both tautomers.
- **FTIR Layout Overhaul (`src/shark/reports/visualizer.py`)**: Expanded $y$-axis range to $[0, 115]\%$, moved functional group band annotations ("O-H / N-H stretch", "C=O / N=O stretch", "Fingerprint Region") into the top header band ($y = 107\%$), and relocated the legend to `loc="lower center"` ($x \approx 2200-2600\text{ cm}^{-1}$), completely eliminating visual overlap with spectra peaks and labels.
- **Test Suite Expansion (`tests/test_visualizers.py`)**: Added unit tests for FTIR transmittance generation, Boltzmann equilibrium plotting, and FMO panel rendering. All 7 unit tests passing.

---

## 2026-09-06 (session 3) — Frontier molecular orbitals & electron density imaging

- **FTIR Transmittance conversion**: Updated `src/shark/reports/visualizer.py` to support `mode='transmittance'` as default. Displays standard FTIR format (baseline at 100%, absorption peaks as downward dips, inverted wavenumber axis).
- **Volumetric Cube Engine (`src/shark/reports/orbitals.py`)**: Implemented `parse_cube_file` capable of reading Gaussian `.cube` files emitted by `orca_plot` (molecular orbitals and densities).
- **Frontier Molecular Orbitals (FMO) visualization**: Implemented `plot_frontier_orbitals_panel` generating a 4-panel publication-grade figure (HOMO/LUMO for 1-oxide and 3-oxide) displaying wavefunction phase coloring (\(+0.03\) in blue, \(-0.03\) in red) overlaid on the 2D molecular skeleton with energy eigenvalues.
- **Total Electron Density Map**: Implemented `plot_electron_density_comparison` generating logarithmic isodensity contours (\(\rho(\mathbf{r})\)) showing the van der Waals envelope.
- **Automated report update**: Re-compiled `examples/demo_tautomer_report.py`, producing `frontier_orbitals_comparison.png` and `electron_density_comparison.png`, and updated `report.html` and `report.md`.

---

## 2026-09-06 (session 2) — Core parser, thermo & visual reporting engine

- **Packaging setup**: Initialized `pyproject.toml` (`shark-qc` v0.1.0) with standard layout in `src/shark/` and configured pytest path discovery.
- **ORCA 6 output parser (`src/shark/core/parser.py`)**: Implemented `parse_orca_results` reading `.property.txt` and `.out` files. Robustly extracts electronic energy (\(E_{el}\)), zero-point energy (\(ZPE\)), enthalpy (\(H\)), Gibbs free energy (\(G\)), entropy (\(S\)), frequencies, IR intensities, dipole moment vector/magnitude, frontier orbital eigenvalues (HOMO, LUMO, gap), and flags imaginary frequencies.
- **Thermochemistry & Boltzmann analysis (`src/shark/analysis/thermo.py`)**: Implemented relative energy calculations (\(\Delta E, \Delta H, \Delta G\) in kcal/mol) and Boltzmann population weighting (\(P_i\)) at specified temperatures (default 298.15 K).
- **Spectroscopy convolution (`src/shark/analysis/spectra.py`)**: Added Lorentzian lineshape convolution for simulated infrared (IR) spectra with auto-alignment for translational/rotational modes.
- **Visualizer & Reporting (`src/shark/reports/visualizer.py`)**: Built publication-quality (300 DPI) plotting functions for Boltzmann distribution and overlaid IR spectra (inverted wavenumber axis, shaded characteristic bands), alongside native markdown and interactive HTML report generators.
- **End-to-end verification (`examples/demo_tautomer_report.py` and `tests/test_parser_and_thermo.py`)**:
  - Validated against benzofuroxan benchmark: confirmed 3-oxide is favored by \(\Delta G = 0.755\text{ kcal/mol}\) (78.15% vs 21.85% population).
  - All 4 unit tests passing in both `unittest` and `pytest`.

---

## 2026-09-06 (session 1) — Project initialization & working notes architecture

- **Working notes established**: Created `notes/` directory mirroring the structure of `poliscreen_notes` (`README.md`, `RULES.md`, `OBJECTIVES.md`, `DECISIONS.md`, `CHANGES.md`, `LOG.md`, `NEXT.md`) and project root `CLAUDE.md`.
- **Unbreakable rules copied and contextualized**: Preserved the 7 inviolable principles with domain adaptations for quantum chemistry.
- **Initial DFT test completed**: Verified execution of `dft_benzofuroxan/prepare_dft_tautomers.py` and convergence of ORCA 6.0 calculations (`tautomer_1_oxide` and `tautomer_3_oxide`).
