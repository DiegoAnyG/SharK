# Decisions

Newest first. Each entry: what was decided, why, and what it cost. Reversing one is allowed —
say so out loud and add a new entry with the new reason. Do not quietly do the opposite.

---

## 2026-09-06 — Compute Allocation: CPU OpenMPI for ORCA core, GPU for ensemble pre-filtering

**Decided:** Use the host 8 CPU cores with OpenMPI for executing ORCA 6.1.1 electronic structure calculations. Reserve the NVIDIA GPU (RTX 5070 Ti) for optional rapid pre-screening of large conformer ensembles using Machine Learning Potentials (AIMNet2 / TorchANI), with a CPU fallback (MMFF94s / xTB).

**Why:** Standard ORCA releases (up to 6.1.1) do not have native CUDA/GPU acceleration for DFT electronic structure methods; they are vectorized for x86 CPUs with OpenMPI. However, relaxing 100 conformers with DFT on CPU takes hours/days, whereas an ML potential on the RTX 5070 Ti can relax 100 conformers in ~20 seconds, allowing SharK to pass only the top 2–3 unique conformers to ORCA.

**Cost:** Adds an optional PyTorch/MLIP dependency for users wanting GPU acceleration; pure CPU users continue using standard force fields or semiempirical xTB.

---

## 2026-09-06 — Project Inception: SharK as an open WEASEL counterpart

**Decided:** Build SharK as a modular, open-source Python framework for automated quantum chemistry workflows, driving ORCA for tasks like automated tautomer exploration, conformer ensembles, and thermochemical analysis.

**Why:** WEASEL (FACCTs) is a commercial, closed-source workflow driver. An open, scriptable Python framework allows full integration with chemoinformatics (RDKit), virtual screening pipelines (such as PoliScreen), and custom analysis workflows without licensing constraints or black-box restrictions.

**Cost:** We must implement robust subprocess orchestration, process signal handling (preventing orphaned ORCA/MPI workers), and comprehensive output parsers.

---

## 2026-09-06 — Adopt the proven notes and rules structure

**Decided:** Adopt the 7-file working notes system (`README.md`, `RULES.md`, `OBJECTIVES.md`, `DECISIONS.md`, `CHANGES.md`, `LOG.md`, `NEXT.md`) inside `notes/` and `CLAUDE.md` at the project root.

**Why:** It eliminates repeated mistakes across sessions, establishes unbreakable scientific and privacy constraints, and provides a clear audit trail of why design choices were made.

**Cost:** Keeping notes synchronized requires discipline at the end of each session.

---

## 2026-09-06 — Reference benchmark: Benzofuroxan tautomer pair

**Decided:** Use the benzofuroxan-5-carboxylic acid tautomeric equilibrium (1-oxide vs 3-oxide) as the initial pilot benchmark for the tautomerism workflow.

**Why:** It is a clinically relevant scaffold with well-documented subtle thermodynamics. The initial test run (`B3LYP/def2-SVP D4 RIJCOSX CPCM(Water) Opt Freq`) converged cleanly, showing the 3-oxide to be thermodynamically favored by \(\Delta G \approx -0.75 \text{ kcal/mol}\).

**Cost:** Validates a closed heterocyclic N-oxide rearrangement; further test sets with proton-transfer tautomers (keto-enol, amide-imidic, azole NH-shifts) must follow.
