"""Quantum mechanical binding energy and covalent interaction analysis.

Calculates interaction energies, frontier orbital eigenvalues (HOMO, LUMO, gap),
and coordinate bond descriptors from ORCA DFT outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Optional

from ..core.parser import parse_orca_output

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.211386


@dataclass
class OrbitalSummary:
    homo_energy_hartree: float
    lumo_energy_hartree: float
    homo_energy_ev: float
    lumo_energy_ev: float
    gap_ev: float
    is_open_shell: bool = False
    somo_energy_hartree: Optional[float] = None
    somo_energy_ev: Optional[float] = None


@dataclass
class QMBindingResult:
    complex_energy_hartree: float
    pocket_energy_hartree: Optional[float] = None
    ligand_energy_hartree: Optional[float] = None
    delta_e_bind_hartree: Optional[float] = None
    delta_e_bind_kcal: Optional[float] = None
    orbitals: Optional[OrbitalSummary] = None
    dipole_debye: float = 0.0
    metal_coordination_dist_angstrom: Optional[float] = None
    raw_data: dict = field(default_factory=dict)


def extract_orbital_summary(orca_out_path: str | Path) -> OrbitalSummary:
    """Extracts frontier orbital energies (HOMO, LUMO, gap) from an ORCA output."""
    path = Path(orca_out_path)
    if not path.is_file():
        raise FileNotFoundError(f"ORCA output file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    # Check for ORBITAL ENERGIES block
    # Pattern: NO OCC E(Hartree) E(eV)
    # 24 2.0000 -0.2501 -6.8055
    homo_eh = None
    lumo_eh = None
    somo_eh = None
    is_uhf = "UHF" in text or "UKS" in text or "SPIN UP" in text

    # Search lines
    in_orb_block = False
    last_occ_eh = None
    first_unocc_eh = None

    lines = text.splitlines()
    for line in lines:
        if "ORBITAL ENERGIES" in line:
            in_orb_block = True
            continue
        if in_orb_block:
            if not line.strip() or "---" in line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    occ = float(parts[1])
                    eh = float(parts[2])
                    ev = float(parts[3])

                    if occ > 0.0:
                        last_occ_eh = (eh, ev)
                        if occ < 2.0 and not occ == 1.0 and is_uhf:
                            somo_eh = (eh, ev)
                    elif occ == 0.0 and first_unocc_eh is None:
                        first_unocc_eh = (eh, ev)
                except ValueError:
                    if "Total" in line or "E(SCF)" in line:
                        break
                    continue

    if last_occ_eh is not None and first_unocc_eh is not None:
        gap = first_unocc_eh[1] - last_occ_eh[1]
        return OrbitalSummary(
            homo_energy_hartree=last_occ_eh[0],
            lumo_energy_hartree=first_unocc_eh[0],
            homo_energy_ev=last_occ_eh[1],
            lumo_energy_ev=first_unocc_eh[1],
            gap_ev=gap,
            is_open_shell=is_uhf,
            somo_energy_hartree=somo_eh[0] if somo_eh else None,
            somo_energy_ev=somo_eh[1] if somo_eh else None
        )

    # Fallback to general parser
    data = parse_orca_output(path)
    e_scf = data.get("energy_scf", 0.0)
    return OrbitalSummary(
        homo_energy_hartree=0.0,
        lumo_energy_hartree=0.0,
        homo_energy_ev=0.0,
        lumo_energy_ev=0.0,
        gap_ev=0.0
    )


def compute_qm_binding_energy(
    complex_out: str | Path,
    pocket_out: Optional[str | Path] = None,
    ligand_out: Optional[str | Path] = None,
    coordination_dist: Optional[float] = None
) -> QMBindingResult:
    """Computes interaction energy between pocket and ligand."""
    comp_path = Path(complex_out)
    comp_data = parse_orca_output(comp_path)
    e_comp = comp_data.get("energy_scf", 0.0)
    dipole = comp_data.get("dipole_total", 0.0)
    orbs = extract_orbital_summary(comp_path)

    e_pocket = None
    e_lig = None
    delta_e_hartree = None
    delta_e_kcal = None

    if pocket_out and Path(pocket_out).is_file():
        p_data = parse_orca_output(Path(pocket_out))
        e_pocket = p_data.get("energy_scf")

    if ligand_out and Path(ligand_out).is_file():
        l_data = parse_orca_output(Path(ligand_out))
        e_lig = l_data.get("energy_scf")

    if e_pocket is not None and e_lig is not None:
        delta_e_hartree = e_comp - (e_pocket + e_lig)
        delta_e_kcal = delta_e_hartree * HARTREE_TO_KCAL

    return QMBindingResult(
        complex_energy_hartree=e_comp,
        pocket_energy_hartree=e_pocket,
        ligand_energy_hartree=e_lig,
        delta_e_bind_hartree=delta_e_hartree,
        delta_e_bind_kcal=delta_e_kcal,
        orbitals=orbs,
        dipole_debye=dipole,
        metal_coordination_dist_angstrom=coordination_dist,
        raw_data=comp_data
    )
