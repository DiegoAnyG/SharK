# Log

Ledger of specific technical changes, and **Lessons** learned.

## Lessons — Read this first

1. **ORCA processes do not always terminate on Python parent exit.** Spawning ORCA via `subprocess` or shells requires explicit process group management (`os.setsid` / `preexec_fn`) so that SIGINT/SIGTERM kills the MPI ring and child executables (`orca_opt`, `orca_gstep`). Otherwise, orphaned binaries peg CPUs in the background.
2. **Scratch files accumulate rapidly in DFT.** An optimization + frequency calculation can generate hundreds of MBs to GBs of intermediate files (`.tmp`, `.densities`, `.cpcm_corr`). High-throughput workflows must run inside an isolated scratch directory and keep only essential deliverables (`.out`, `.xyz`, `.property.txt`, `.gbw` if requested).
3. **Never trust single-point energies for tautomer preference.** Thermal contributions (\(ZPE\), entropy, thermal enthalpy) and solvation free energy can flip energetic ordering compared to gas-phase electronic energy alone. Always calculate full \(\Delta G_{solv}\) with frequencies to verify stationary states.
4. **Vibrational mode offsets in ORCA outputs.** In ORCA, `&FREQ` in `.property.txt` reports all \(3N\) degrees of freedom (including the 5 or 6 translational/rotational modes with zero frequency), whereas the `IR SPECTRUM` table in `.out` reports only the \(3N-6\) actual vibrational transitions. A robust spectrum simulator must map the frequencies and intensities using the parsed IR table or slice off the low translational/rotational modes.
5. **Keep reporting dependencies zero-external.** Report tables in Markdown should use native formatters rather than depending on optional packages like `tabulate`, ensuring reports generate seamlessly across any environment.
6. **Pi orbital volumetric projection across planar geometries.** When slicing or projecting 3D volumetric wavefunctions ($\psi$) for planar conjugated systems onto 2D, never use arithmetic averaging across the molecular plane. A $\pi$ orbital has an antisymmetric nodal plane ($\psi = 0$) coinciding with the ring plane. Integrating across the nodal plane leads to exact destructive cancellation ($+A + (-A) = 0$). Instead, slice at the peak lobe amplitude above the plane ($z \approx z_{plane} + 0.70\text{ \AA}$ for $2p_z$) to preserve the true sign and spatial structure of all lobes.

---

## 2026-09-06 — FMO $\pi$-Nodal Plane Resolution & FTIR Layout Overhaul

- Fixed orbital lobe cancellation in `src/shark/reports/orbitals.py` by sampling at $+0.70\text{ \AA}$ above the atomic plane rather than averaging across $z_{mid}$.
- Overhauled `plot_ir_comparison` in `src/shark/reports/visualizer.py`: expanded $y \in [0, 115]\%$, moved functional group band labels into top banner ($y = 107\%$), placed legend at `loc="lower center"`.
- Added `tests/test_visualizers.py` testing IR transmittance, Boltzmann plots, and orbital panel generation. All 7 unit tests passing.

---

## 2026-09-06 — Core Parser, Thermo & Visual Reporting Engine

- Created package structure `shark-qc` v0.1.0 with `src/shark/core/`, `src/shark/analysis/`, and `src/shark/reports/`.
- Implemented `parse_orca_results`: extracts \(E_{el}\), \(ZPE\), \(H\), \(G\), \(S\), frequencies, IR intensities, dipole vector/magnitude, HOMO, LUMO, and gap.
- Implemented `calculate_relative_thermo`: calculates \(\Delta E_{el}\), \(\Delta H\), \(\Delta G\) (kcal/mol) and Boltzmann populations (\(P_i\)).
- Implemented `simulate_ir_spectrum`: Lorentzian broadening over harmonic vibrational transitions with automatic mode alignment.
- Implemented `visualizer.py`: High-DPI publication plots for Boltzmann distribution and overlaid IR spectra (inverted wavenumber axis, key band shading), along with Markdown and HTML report generation.
- Validated with `examples/demo_tautomer_report.py` and 4 passing tests in `tests/test_parser_and_thermo.py`.

---

## 2026-09-06 — Project Setup & Initial Benzofuroxan Calculations

- Formatted and generated standard working notes (`README.md`, `RULES.md`, `OBJECTIVES.md`, `DECISIONS.md`, `CHANGES.md`, `LOG.md`, `NEXT.md`).
- Confirmed results for pilot benchmark in `dft_benzofuroxan/`:
  - `tautomer_1_oxide`: Final \(E_{el} = -678.61671677\text{ Eh}\), Gibbs Free Energy \(G = -678.54021185\text{ Eh}\).
  - `tautomer_3_oxide`: Final \(E_{el} = -678.61791720\text{ Eh}\), Gibbs Free Energy \(G = -678.54141500\text{ Eh}\).
  - Relative stability: \(\Delta E_{el} = -0.753\text{ kcal/mol}\), \(\Delta G = -0.755\text{ kcal/mol}\) (tautomer 3-oxide favored).
  - Verified stationary points: Both converged with zero imaginary frequencies.
