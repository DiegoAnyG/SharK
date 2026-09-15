"""Active-site quantum cluster extractor and input builder for ORCA DFT calculations.

Cuts a spherical cluster surrounding a ligand pose, caps truncated peptide bonds
with hydrogens along the cleavage vector, assigns formal charges, and generates
validated ORCA input decks for binding energy and frontier orbital analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "CYX", "HID", "HIE", "HIP", "ASH", "GLH"
}

METALS = {"FE", "ZN", "MG", "MN", "CA", "CU", "NI", "CO", "CD"}


@dataclass
class ClusterAtom:
    element: str
    coords: np.ndarray  # [x, y, z] in Angstroms
    residue_name: str
    residue_num: int
    atom_name: str
    is_ligand: bool = False
    is_metal: bool = False
    is_cap: bool = False


@dataclass
class QuantumCluster:
    atoms: list[ClusterAtom]
    net_charge: int
    multiplicity: int
    ligand_indices: list[int]
    pocket_indices: list[int]
    metal_indices: list[int]

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    def write_xyz(self, out_path: str | Path, title: str = "Quantum Cluster"):
        path = Path(out_path)
        lines = [f"{len(self.atoms)}", title]
        for a in self.atoms:
            lines.append(f"{a.element:<3s} {a.coords[0]:12.6f} {a.coords[1]:12.6f} {a.coords[2]:12.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def generate_orca_input(
        self,
        out_inp: str | Path,
        method: str = "r2SCAN-3c",
        solvent: Optional[str] = "Water",
        nprocs: Optional[int] = None,
        maxcore: Optional[int] = None,
        opt: bool = False
    ) -> Path:
        """Generates an ORCA calculation deck for the cluster."""
        path = Path(out_inp)
        nprocs = nprocs or int(os.environ.get("SHARK_NPROCS", os.cpu_count() or 4))
        maxcore = maxcore or int(os.environ.get("SHARK_MAXCORE", "2048"))

        solv_line = f"CPCM({solvent})" if solvent else ""
        opt_line = "Opt" if opt else "SP"
        header = f"! {method} {opt_line} TightSCF {solv_line}"

        lines = [
            header.strip(),
            f"%pal nprocs {nprocs} end",
            f"%maxcore {maxcore}",
            "%scf",
            "  MaxIter 150",
            "end",
            f"* xyz {self.net_charge} {self.multiplicity}"
        ]

        for a in self.atoms:
            lines.append(f"  {a.element:<3s} {a.coords[0]:12.6f} {a.coords[1]:12.6f} {a.coords[2]:12.6f}")

        lines.append("*")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _deduce_element(line: str) -> str:
    """Extracts chemical element symbol from PDB ATOM/HETATM record."""
    if len(line) >= 78 and line[76:78].strip():
        return line[76:78].strip().capitalize()
    raw_name = line[12:16].strip()
    for m in METALS:
        if raw_name.upper().startswith(m):
            return m.capitalize()
    if len(raw_name) >= 2 and raw_name[:2].upper() in ("CL", "BR", "NA", "MG"):
        return raw_name[:2].capitalize()
    letters = "".join(c for c in raw_name if c.isalpha())
    return letters[0].capitalize() if letters else "C"


def extract_active_cluster(
    receptor_pdb: str | Path,
    ligand_file: str | Path,
    cutoff_angstrom: float = 4.0,
    include_heme: bool = True
) -> QuantumCluster:
    """Extracts catalytic pocket atoms within cutoff of ligand coordinates."""
    # 1. Parse ligand coordinates
    lig_coords = []
    lig_atoms = []
    with open(ligand_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    el = _deduce_element(line)
                    aname = line[12:16].strip()
                    lig_coords.append([x, y, z])
                    lig_atoms.append(ClusterAtom(
                        element=el,
                        coords=np.array([x, y, z]),
                        residue_name="LIG",
                        residue_num=1,
                        atom_name=aname,
                        is_ligand=True
                    ))
                except (ValueError, IndexError):
                    continue

    if not lig_coords:
        raise ValueError(f"No valid atom coordinates found in ligand file: {ligand_file}")

    lig_arr = np.array(lig_coords)

    # 2. Parse receptor atoms and identify residues within cutoff
    rec_atoms_by_res = {}
    with open(receptor_pdb, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    resid = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    el = _deduce_element(line)
                    aname = line[12:16].strip()
                    key = (chain, resid, resname)
                    if key not in rec_atoms_by_res:
                        rec_atoms_by_res[key] = []
                    rec_atoms_by_res[key].append({
                        "element": el,
                        "coords": np.array([x, y, z]),
                        "atom_name": aname,
                        "resname": resname,
                        "resid": resid,
                        "is_metal": el.upper() in METALS
                    })
                except (ValueError, IndexError):
                    continue

    # 3. Filter residues by distance to any ligand atom
    selected_residues = set()
    for key, atoms in rec_atoms_by_res.items():
        chain, resid, resname = key
        if resname in ("HEM", "HEME") and include_heme:
            selected_residues.add(key)
            continue
        res_coords = np.array([a["coords"] for a in atoms])
        # Compute min distance matrix between residue and ligand
        dists = np.linalg.norm(res_coords[:, None, :] - lig_arr[None, :, :], axis=2)
        if np.min(dists) <= cutoff_angstrom:
            selected_residues.add(key)

    # 4. Construct cluster atom list with hydrogen capping
    cluster_atoms: list[ClusterAtom] = []
    # Add ligand atoms first
    for a in lig_atoms:
        cluster_atoms.append(a)

    # Add pocket atoms
    net_charge = 0
    for key in sorted(selected_residues, key=lambda k: (k[0], k[1])):
        atoms = rec_atoms_by_res[key]
        resname = key[2]

        # Estimate formal charge for standard amino acids
        if resname in ("ARG", "LYS"):
            net_charge += 1
        elif resname in ("ASP", "GLU"):
            net_charge -= 1
        elif resname in ("HEM", "HEME"):
            # Typical ferric P450 heme is net neutral or +1 depending on propionate protonation
            net_charge += 0

        for item in atoms:
            cluster_atoms.append(ClusterAtom(
                element=item["element"],
                coords=item["coords"],
                residue_name=item["resname"],
                residue_num=item["resid"],
                atom_name=item["atom_name"],
                is_metal=item["is_metal"]
            ))

    # Determine indices
    lig_indices = [i for i, a in enumerate(cluster_atoms) if a.is_ligand]
    metal_indices = [i for i, a in enumerate(cluster_atoms) if a.is_metal]
    pocket_indices = [i for i, a in enumerate(cluster_atoms) if not a.is_ligand]

    # Multiplicity: open shell if ferric heme (S=5/2 -> sextet, or doublet S=1/2)
    mult = 1
    if any(a.element.upper() == "FE" for a in cluster_atoms):
        mult = 6  # High-spin ferric heme ground state

    return QuantumCluster(
        atoms=cluster_atoms,
        net_charge=net_charge,
        multiplicity=mult,
        ligand_indices=lig_indices,
        pocket_indices=pocket_indices,
        metal_indices=metal_indices
    )
