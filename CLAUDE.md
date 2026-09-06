# CLAUDE.md

Guidance for AI assistants when working in this repository. These instructions override default
behaviour; follow them exactly.

## What SharK is

**SharK** is an open, reproducible quantum chemical workflow framework inspired by **WEASEL** (FACCTs).
It automates complex computational chemistry tasks using **ORCA** and chemoinformatics toolkits (RDKit, semiempirical methods):
**Input (SMILES/3D) → Tautomer & conformer generation → ORCA calculation orchestration → Output extraction & thermochemical analysis**.

Author: Diego Cesar Anaya Guerrero.

## Working notes

Longer-lived context — objectives, decisions already made and their cost, and a log of what
changed and why — is kept in the `notes/` directory within this repository:
- `notes/NEXT.md`: Current handover state and immediate tasks. Read first every session.
- `notes/RULES.md`: Inviolable rules.
- `notes/OBJECTIVES.md`: Scope and definition of "done".
- `notes/DECISIONS.md`: Architectural choices and rationale.
- `notes/LOG.md`: Change ledger and accumulated Lessons.
- `notes/CHANGES.md`: Session-by-session narrative.

Read these before proposing work, and whenever a question feels like one that has already been
settled. If out of date, update there first, then act.

## Hard rules

1. **Never bend the code toward a result.** Never write code so that a chosen answer comes out.
   No tautomer forced to be most stable, no energy fitted to expectations. Methodological improvements
   are encouraged; results must remain an unmanipulated physical output. If it would be embarrassing
   to describe in a Methods section, it is the wrong change.
2. **Never leak anything personal.** No absolute paths from a personal machine, no usernames, no
   machine names, no home directories, no credentials in code, comments, tests, docs, or commit messages.
   (Exception: author name for citation and authorship).
3. **Never hardcode.** No hardcoded paths or lists. Discover from disk; honour `SHARK_ORCA`,
   `SHARK_SCRATCH`, `SHARK_NPROCS`, `SHARK_MAXCORE`. A path existing only on the author's machine
   must not fail with a crash — it should cleanly report missing tools/data.
4. **English, always.** Code, comments, docstrings, identifiers, on-disk names, tests, docs, notes,
   and commit messages.
5. **Keep the workspace clean.** Announce every new directory: where, what for, and whether it can
   be deleted. Temporary calculation files and heavy scratch go to an isolated scratch folder.
6. **Backward compatibility is mandatory.** Older calculations, configs, and output archives must
   keep parsing cleanly.
7. **Verify what only shows up at runtime.** Subprocess lifecycle, ORCA signal handling (no zombie
   MPI processes), memory bounds, and cross-platform path handling get explicit tests.

## Environment and running

- Python 3.10+ / Conda environment (e.g., `cribado`).
- Key dependencies: `rdkit`, `numpy`, `pandas`, `scipy`.
- External tools: `orca` (quantum chemistry engine). Check with `which orca` or `$SHARK_ORCA`.

## Architecture Vision

- `shark/core/` — Core engine:
  - `orca.py`: ORCA input builder, execution manager, signal/process cleanup.
  - `parser.py`: Robust parser for ORCA `.out` and `.property.txt` (energies, frequencies, dipole, orbitals).
  - `converters.py`: Format translation (SMILES, RDKit Mol, XYZ, PDB).
- `shark/workflows/` — High-level automated workflows:
  - `tautomers.py`: Automated enumeration, conformer sampling, DFT minimization, Boltzmann population.
  - `conformers.py`: Conformational ensemble generation, pruning, and hierarchical refinement.
  - `solvation.py`: Gas-phase vs implicit solvent comparisons (\(\Delta G_{solv}\)).
- `shark/cli.py` — Scriptable command-line interface.

