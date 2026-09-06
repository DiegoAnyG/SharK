# Objectives

## What SharK is

**SharK** is an open, modular quantum chemical workflow engine and report suite inspired by **WEASEL** (FACCTs). Built in Python, it automates quantum mechanical calculations with **ORCA 6** (and complementary chemoinformatics/GPU machine-learning potentials like RDKit and AIMNet2/xTB).

Its mission is twofold:
1. **Automated End-to-End Orchestration**: Seamless workflows from structure input to publication-ready research reports (figures, spectra, tables, interactive HTML).
2. **Dual-Audience Flexibility**: Effortless zero-configuration presets for non-specialists/beginners, alongside deep, uncompromised customization for computational chemistry experts.

## Who runs it, and how

- **Beginners / Fast Screening**: 1-click execution using validated presets (e.g. `"fast-screening"`, `"publication-quality"`, solvent selection) without manual ORCA input syntax.
- **Experts**: Granular parameterization of functionals (`B3LYP`, `wB97X-D4`, `PBE0`, `M06-2X`, composite DFT `r2SCAN-3c`), basis sets (`def2-SVP`, `def2-TZVP`, `ma-def2-...`), dispersion (`D3BJ`, `D4`), implicit solvents (`CPCM`, `SMD`), grids, and property modules.
- **Integration with PoliScreen**: Seamlessly ingest hit compounds from PoliScreen to resolve physiological tautomeric ratios, binding-strain penalties (\(\Delta G_{taut}\)), or enrich docking/ADMET scoring with quantum descriptors.

## Compute & Hardware Strategy

- **ORCA 6.1.1 (Electronic Structure Core)**: Runs on multi-threaded/distributed **CPU via OpenMPI** across all host cores (e.g. 8 CPU cores). ORCA electronic structure algorithms do not natively utilize CUDA in standard releases.
- **NVIDIA GPU Acceleration (e.g. RTX 5070 Ti 16 GB)**: Leveraged for high-throughput conformer ensemble relaxation and pre-screening using GPU-accelerated Neural Network Potentials (MLIPs like AIMNet2 / TorchANI). This collapses a 100-conformer relaxation step from ~20 hours on CPU to **< 30 seconds on GPU**, forwarding only the top 2–3 unique minima to ORCA for final DFT + analytical Hessian calculation. (With automatic CPU fallback to MMFF94s / xTB).

## Roadmap of the ~25 Quantum Chemical Analyses (WEASEL counterpart)

### 1. Tautomerism & Conformers (Milestone 1 — Active)
- **A1. Tautomer Enumeration & Energetics**: RDKit enumeration, DFT optimization, Gibbs free energy \(\Delta G_{solv}\), and Boltzmann populations \(P_i\).
- **A2. Conformer Ensemble Screening**: Hierarchical sampling (ETKDGv3 → GPU/ML-FF or MMFF → DFT refinement).
- **A3. Rotational Barrier Scans**: Relaxed dihedral angle potential energy surface scans.

### 2. Thermochemistry & Solvation
- **A4. Full Thermochemical Breakdown**: \(E_{el}\), \(ZPE\), \(H\), \(G\), \(S\), and stationary point verification (\(N_{imag} = 0\)).
- **A5. Solvation Free Energies (\(\Delta G_{solv}\))**: Gas-phase vs implicit solvent (CPCM/SMD) thermodynamic cycles.
- **A6. Temperature & Pressure Dependencies**: Thermochemical corrections across arbitrary \(T\) ranges.

### 3. Spectroscopy & Optical Properties
- **A7. IR Vibrational Spectra**: Lorentzian-convoluted simulated infrared absorption spectra.
- **A8. Raman Spectra**: Vibrational polarizability derivatives and Raman intensities.
- **A9. UV-Vis / Electronic Absorption**: TD-DFT singlet/triplet excitation energies, oscillator strengths, and spectrum convolution.
- **A10. NMR Chemical Shifts**: Gauge-Independent Atomic Orbital (GIAO) isotropic shieldings and TMS-referenced \({}^1\text{H}/{}^{13}\text{C}\) shifts.
- **A11. Circular Dichroism (ECD)**: Electronic circular dichroism rotatory strengths for chiral conformers.

### 4. Reactivity & Electronic Descriptors
- **A12. Frontier Molecular Orbitals (FMO)**: HOMO, LUMO, energy gap, chemical hardness (\(\eta\)), softness (\(S\)), and electrophilicity index (\(\omega\)).
- **A13. Conceptual DFT / Fukui Functions**: Nucleophilic (\(f^+\)), electrophilic (\(f^-\)), and radical (\(f^0\)) local reactivity indices.
- **A14. Molecular Electrostatic Potential (MEP)**: Extreme surface values (\(V_{S,max}\), \(V_{S,min}\)) for electrophilic/nucleophilic site identification.
- **A15. Atomic Charges & Population Analyses**: Mulliken, Löwdin, CHELPG electrostatic potential charges, and Mayer bond orders.
- **A16. Bond Dissociation Energies (BDE)**: Homolytic bond cleavage thermodynamics for metabolic stability or radical scavenging.

### 5. Physical Chemistry & Thermodynamics
- **A17. \(pK_a\) Estimation**: Deprotonation/protonation thermodynamic cycle with implicit/explicit solvent corrections.
- **A18. Redox Potentials**: One-electron oxidation/reduction standard electrode potentials (\(E^\circ\)).
- **A19. Dipole Moments & Polarizabilities**: Static dipole vectors and polarizability tensors.

### 6. Non-Covalent Interactions & Dimer Binding
- **A20. Interaction / Binding Energy**: Supramolecular dimer binding with Counterpoise (CP / BSSE) correction.
- **A21. SAPT / Energy Decomposition Analysis (EDA)**: Electrostatic, exchange, induction, and dispersion partitioning.

### 7. Reaction Mechanisms & Kinetics
- **A22. Transition State Optimization**: Eigenvector-following and Berny TS searches.
- **A23. Intrinsic Reaction Coordinate (IRC)**: Minimum energy pathway tracing connecting reactants, TS, and products.
- **A24. Reaction Free Energy Profiles**: Activation barriers \(\Delta G^\ddagger\) and reaction thermodynamics \(\Delta G_{rxn}\).
- **A25. Eyring Rate Constants**: Transition state theory (TST) reaction rate calculations at varying temperatures.

## Deliverables & Reporting

- **Interactive HTML & Static PDF/PNG Reports**:
  - Publication-grade figures (Altair / Matplotlib / Seaborn) with automatic margin protection and clear labeling.
  - Molecular 2D diagrams with RDKit and 3D coordinate viewer embeds.
  - Tabular exports (CSV, JSON, pandas DataFrame).

## Non-negotiables

- **Reproducibility.** Fixed conformer seeds, strict tracking of ORCA versions, exact input templates, and unambiguous parameter preservation.
- **Process Safety & Clean Exit.** Clean process group management (`os.setsid`) to prevent orphaned MPI or `orca_opt` processes on interruption (`Ctrl+C`).
- **Resource Respect.** Memory (`maxcore`) and CPU (`nprocs`) bounded strictly to host limits.
- **Graceful Failure.** Convergence errors and imaginary frequencies clearly flagged and explained.
