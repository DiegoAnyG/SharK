<p align="center">
  <img src="assets/shark_logo.jpg" alt="SharK Logo" width="380"/>
</p>

<h1 align="center">SharK</h1>

<p align="center">
  <strong>Democratizing Computational Chemistry — Science Without Barriers</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#license">Philosophy</a>
</p>

---

## Overview

**SharK** is an open-source, reproducible Python orchestrator and analysis engine for quantum chemistry, designed as an accessible alternative to proprietary workflows like WEASEL. 

SharK bridges high-throughput virtual screening (such as [**PoliScreen**](https://github.com/DiegoAnyG/PoliScreen)) with rigorous quantum mechanics (DFT via ORCA 6.x). It automates the extraction of electronic structures, thermochemical equilibria, simulated infrared (FTIR) spectra, and frontier orbital visualizations into publication-grade reports.

---

## Key Features

- **Robust ORCA 6 Parser**: Instant extraction of electronic energies ($E_{el}$), thermochemical corrections ($ZPE$, $H$, $G$, $S$), dipole moments, Loewdin/Mulliken atomic charges, and vibrational modes from `.property.txt` and `.out` files.
- **Boltzmann Thermodynamics**: Calculates relative free energies ($\Delta G, \Delta H, \Delta E$) and equilibrium Boltzmann population distributions ($P_i$) at specified temperatures.
- **Conceptual DFT & Warhead Reactivity**: Calculates chemical hardness ($\eta$), chemical potential ($\mu$), global electrophilicity index ($\omega$), chemical softness ($S$), and condensed Fukui functions ($f_k^+, \omega_k$) to automatically rank electrophilic warhead centers.
- **Covalent Near-Attack Conformation (NAC) Matching**: Autonomous 3D spatial scanning of catalytic pocket nucleophiles (Cys, Ser, Thr, Lys, His, Tyr) against ligand warheads ($d \le 3.5\text{ \AA}$) to assess covalent reaction feasibility without prior manual residue specification.
- **GROMACS MD Pipeline Bridge**: Automated orchestration of nanosecond classical molecular dynamics from PoliScreen screening sessions, strictly selecting raw receptor structures for accurate force-field topology generation (`pdb2gmx`).
- **FTIR Transmittance Simulator**: Convolutes harmonic transitions with Lorentzian line-shapes into standardized FTIR transmittance spectra ($\%T$, 100% baseline, downward absorption dips) with shaded functional-group zones.
- **Frontier Molecular Orbitals (FMO)**: Signed 3D HOMO/LUMO surfaces from ORCA CUBE fields, with offline interaction, source hashes and configurable exports.
- **Offline Reporting**: One-command generation of 300-DPI publication figures, GitHub-flavored Markdown summaries, and interactive standalone HTML dossiers.

---

## Quickstart

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/DiegoAnyG/SharK.git
cd SharK
pip install -e .
```

### Python API

```python
from shark.core.parser import parse_orca_results
from shark.analysis.thermo import calculate_relative_thermo
from shark.reports.visualizer import plot_boltzmann_equilibrium, plot_ir_comparison

# 1. Parse ORCA calculation results
res1 = parse_orca_results("calculations/state_a", name="state_a")
res3 = parse_orca_results("calculations/state_b", name="state_b")

# 2. Compute relative thermochemistry and Boltzmann populations
df = calculate_relative_thermo([res1, res3], temperature=298.15)
print(df[["Name", "dG (kcal/mol)", "Population (%)"]])

# 3. Generate publication-ready figures
plot_boltzmann_equilibrium(df, "boltzmann_distribution.png")
plot_ir_comparison([res1, res3], "ir_spectra.png", mode="transmittance")
```

Run tests to verify installation:

```bash
pytest -v
```

---

## PoliScreen to isolated-ligand DFT

```bash
# Prepare one unique candidate, excluding crystallographic controls.
shark run-qm --session screening.poliscreen --work-dir ../qm_jobs

# Optimize in CPCM water, calculate frequencies, and export 3D frontier orbitals.
shark run-qm --session screening.poliscreen --work-dir ../qm_results \
  --execute --frequencies --nprocs 2 --maxcore 1024

# Regenerate geometry from recorded SMILES, selecting a target and Pareto leaders.
shark run-qm --session screening.poliscreen --target target_ready~Pk1 \
  --pareto --top 3 --geometry smiles --work-dir ../selected_jobs
```

`--dft --session ...` is an alias for `run-qm`. Preparation never runs ORCA;
`--execute` launches it. Output directories must be new or empty. By default,
outputs use a unique directory under `SHARK_SCRATCH` or the system temporary directory.
Choose `--work-dir` to retain jobs outside temporary storage.

The default method is `r2SCAN-3c`, geometry optimization, CPCM water and singlet spin.
These are starting settings, not a validated protocol for every compound. Override with
`--theory`, `--solvent` (`gas` disables CPCM), `--multiplicity`, `--single-point`,
and `--frequencies`. `--charge` checks the charge against the input molecular structure;
it does not change protonation. CPU and memory defaults honor `SHARK_NPROCS` and
`SHARK_MAXCORE` (MB per process). `SHARK_ORCA` accepts an executable or installation
directory, with `orca` on PATH as fallback. `--timeout` limits each external process.
ORCA is installed separately and keeps its own license.

Selection uses exact compound and target identities, including pocket suffixes. Use
`--compound NAME` (repeatable) for explicit selection or `--include-controls` to include
reference compounds. Across pockets, a ligand is prepared once; target associations
remain in its record. Without explicit names, `--top` selects by recorded docking score;
this is a prioritization criterion, not evidence of affinity.

Geometry comes from the input SDF/MOL/MOL2 when valid 3D coordinates are available;
otherwise RDKit ETKDGv3 generates a seeded conformer. If there is no input structure,
recorded SMILES can supply the chemistry. Invalid inputs or disagreement with the
recorded SMILES stop preparation. `--geometry smiles` explicitly regenerates the geometry
from SMILES, useful for unsupported MOL2 representations. PDB/PDBQT coordinates alone
are not used to infer chemistry. This workflow evaluates the isolated input ligand,
not its docking pose or a protein cluster. It does not enumerate protonation states,
tautomers or conformer ensembles.

Each `job-NNN/` contains `geometry.xyz`, `calculation.inp` and `job.json`. Execution adds
raw ORCA output, the GBW wavefunction and calculated results. Records include source and
input hashes, parameters, seed, software versions, execution status, output hashes and
failures. Changed inputs and already-executed jobs require fresh preparation. Raw engine
logs may contain local runtime paths; keep calculation directories private.

The offline `dossier.html` distinguishes prepared, completed, failed and interrupted jobs.
Only normally terminated calculations with a final energy and orbital table provide
results; optimization also requires ORCA's convergence marker. Frequencies are required
to assess a stationary minimum. Electronic energies and orbital gaps are not binding free
energies or evidence of biological activity.

Completed runs export frontier CUBE fields using the actual final orbital indices and
spin channels, then generate an offline `frontier_orbitals.html` viewer per job.
`--orbital-grid` controls resolution (default 60 points per axis); a single grid does not
establish spatial convergence. `--no-orbital-plots` retains eigenvalues without exporting
surfaces. Export failures are recorded separately from completed DFT calculations.

Existing CUBE fields can also be rendered with `shark-orbitals <calculation-directory>
--output <report-directory> --formats html`. For static PNG/PDF exports, install
`pip install -e '.[export]'` and Chrome/Chromium, then use `--formats html png pdf`.
Use `--config <saved-settings.json>` to reproduce a saved view.

Input syntax, parallel invocation and orbital export follow the official
[ORCA input documentation](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/input.html),
[parallel execution guide](https://www.faccts.de/docs/orca/6.1/tutorials/first_steps/parallel.html),
and [orca_plot documentation](https://www.faccts.de/docs/orca/6.1/manual/contents/utilitiesvisualization/utilities.html).

The GROMACS MD bridge is configured via `SHARK_GROMACS_PIPELINE` or `--pipeline-dir`.
The orchestrator extracts raw receptor PDB files from PoliScreen sessions to ensure clean
`pdb2gmx` topology builds, prepares simulation directories, and launches production runs.
Covalent Near-Attack Conformations can be evaluated via `--covalent` across poses or MD trajectory frames.
Private notes, calculations and reports are excluded from version control. Tests using
private benchmark data can locate it through `SHARK_TEST_DATA` (or `TOPICS_TEST_DATA`)
and skip when it is unavailable. The session and DFT workflow tests use synthetic inputs.

---

## Philosophy

Scientific progress should not be locked behind financial paywalls or proprietary software suites. **SharK** is developed under the premise that rigorous computational chemistry and biophysical discovery tools should be freely accessible to researchers, students, and institutions worldwide.
