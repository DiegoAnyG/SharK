"""Active Site QM Cluster Extractor and Valence Capping Engine.

Supports both:
1. Minimal Capped Residue Model: Ligand + targeted nucleophilic residue
   (e.g., THR309 capped with standard hydrogen caps on cleaved peptide bonds, ~30-50 atoms).
2. Extended Pocket Cluster Model: Ligand + all active site residues within
   a radial cutoff (~100-150 atoms) with hydrogen capping and backbone freezing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple, Union

from .covalent_matcher import (
    parse_pdb_atoms,
    parse_ligand_pose_coordinates,
    _euclidean_distance,
    REACTIVE_NUCLEOPHILES,
)


@dataclass
class ClusterAtom:
    """An individual atom within the QM active-site cluster."""
    element: str
    coords: Tuple[float, float, float]
    atom_name: str = ""
    res_name: str = ""
    res_seq: int = 0
    chain_id: str = "A"
    is_cap: bool = False
    is_frozen: bool = False
    formal_charge: int = 0
    charge: float = 0.0

    @property
    def label(self) -> str:
        tag = "[CAP]" if self.is_cap else ""
        frz = "[FRZ]" if self.is_frozen else ""
        return f"{self.element} ({self.res_name}{self.res_seq}:{self.atom_name}){tag}{frz}"


@dataclass
class QMCluster:
    """Quantum mechanical active site cluster model for DFT and TS modeling."""
    name: str
    atoms: List[ClusterAtom] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    nucleophile_idx: Optional[int] = None   # 0-based index in atoms list
    electrophile_idx: Optional[int] = None  # 0-based index in atoms list
    model_type: str = "minimal"             # "minimal" or "extended"
    target_residue: str = "THR309"
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]

    @property
    def coordinates(self) -> List[Tuple[float, float, float]]:
        return [a.coords for a in self.atoms]

    @property
    def frozen_indices(self) -> List[int]:
        """0-based atom indices marked as frozen for geometric constraints."""
        return [idx for idx, a in enumerate(self.atoms) if a.is_frozen]

    def to_xyz(self) -> str:
        """Serializes the cluster to standard XYZ format."""
        lines = [f"{len(self.atoms)}", f"{self.name} | Model: {self.model_type} | Charge: {self.charge} Mult: {self.multiplicity}"]
        for a in self.atoms:
            lines.append(f"{a.element:<2} {a.coords[0]:12.6f} {a.coords[1]:12.6f} {a.coords[2]:12.6f}")
        return "\n".join(lines) + "\n"

    def write_xyz(self, filepath: str | Path) -> Path:
        """Writes the cluster to an XYZ file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_xyz(), encoding="utf-8")
        return p

    def to_orca_input(
        self,
        job_type: str = "scan",
        method: str = "r2SCAN-3c",
        solvent: Optional[str] = "Water",
        scan_start: float = 3.30,
        scan_end: float = 1.45,
        scan_steps: int = 18,
        nprocs: int = 4,
        maxcore_mb: int = 2000,
    ) -> str:
        """Generates an ORCA 6 input file for reaction scanning or TS optimization.

        Parameters
        ----------
        job_type : str
            'scan' for relaxed surface coordinate scan,
            'optts' for saddle-point TS optimization + frequency calculation,
            'sp' for single point calculation.
        method : str
            DFT functional or composite method, default 'r2SCAN-3c'.
        solvent : str or None
            Implicit solvent via CPCM, e.g. 'Water' or None.
        scan_start : float
            Starting distance in Angstroms for coordinate scan.
        scan_end : float
            Ending distance in Angstroms for coordinate scan.
        scan_steps : int
            Number of scan intervals.
        """
        lines = []

        # Route line
        solvent_str = f"CPCM({solvent})" if solvent else ""
        if job_type.lower() == "scan":
            route = f"! Opt {method} {solvent_str} TightSCF"
        elif job_type.lower() == "optts":
            route = f"! OptTS Freq {method} {solvent_str} TightSCF"
        else:
            route = f"! {method} {solvent_str} TightSCF"
        lines.append(route.strip())

        # Parallel & Memory
        if nprocs > 1:
            lines.append(f"%pal nprocs {nprocs} end")
        lines.append(f"%maxcore {maxcore_mb}")

        # Geometry block: constraints or scan
        has_geom = False
        geom_lines = ["%geom"]

        if job_type.lower() == "scan" and self.nucleophile_idx is not None and self.electrophile_idx is not None:
            has_geom = True
            geom_lines.append("   Scan")
            geom_lines.append(f"      B {self.nucleophile_idx} {self.electrophile_idx} = {scan_start:.2f}, {scan_end:.2f}, {scan_steps}")
            geom_lines.append("   end")
        elif job_type.lower() == "optts":
            has_geom = True
            geom_lines.append("   Calc_Hess true")
            geom_lines.append("   Recalc_Hess 5")

        frozen = self.frozen_indices
        if frozen:
            has_geom = True
            geom_lines.append("   Constraints")
            for idx in frozen:
                geom_lines.append(f"      {{ C {idx} C }}")
            geom_lines.append("   end")

        geom_lines.append("end")
        if has_geom:
            lines.extend(geom_lines)

        # Coordinate block
        lines.append(f"* xyz {self.charge} {self.multiplicity}")
        for a in self.atoms:
            lines.append(f"  {a.element:<2} {a.coords[0]:12.6f} {a.coords[1]:12.6f} {a.coords[2]:12.6f}")
        lines.append("*")
        return "\n".join(lines) + "\n"

    def write_orca_input(self, filepath: str | Path, **kwargs) -> Path:
        """Writes the cluster to an ORCA input file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_orca_input(**kwargs)
        p.write_text(content, encoding="utf-8")
        return p


def _vector_sub(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])


def _vector_norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _create_capping_hydrogen(
    kept_coord: Tuple[float, float, float],
    deleted_coord: Tuple[float, float, float],
    target_bond_length: float = 1.09
) -> Tuple[float, float, float]:
    """Generates a capping hydrogen coordinate along the cleaved bond vector."""
    v = _vector_sub(deleted_coord, kept_coord)
    norm = _vector_norm(v)
    if norm < 1e-6:
        # Fallback if identical coordinates
        return (kept_coord[0], kept_coord[1], kept_coord[2] + target_bond_length)
    direction = (v[0] / norm, v[1] / norm, v[2] / norm)
    return (
        round(kept_coord[0] + target_bond_length * direction[0], 4),
        round(kept_coord[1] + target_bond_length * direction[1], 4),
        round(kept_coord[2] + target_bond_length * direction[2], 4),
    )


def _prepare_ligand_atoms(
    ligand_pose: Union[str, Path, Sequence[Tuple[str, Tuple[float, float, float]]]],
    ligand_smiles: Optional[str] = None
) -> List[ClusterAtom]:
    """Extracts all ligand atoms, adding hydrogens if only heavy atoms are present."""
    if isinstance(ligand_pose, (list, tuple)):
        # Passed as list of (element, coords) or (atom_name, coords)
        result = []
        for idx, item in enumerate(ligand_pose):
            if len(item) == 2:
                elem = str(item[0]).strip().capitalize()
                crd = (float(item[1][0]), float(item[1][1]), float(item[1][2]))
                result.append(ClusterAtom(
                    element=elem,
                    coords=crd,
                    atom_name=f"{elem}{idx+1}",
                    res_name="LIG",
                    res_seq=1,
                ))
        return result

    p = Path(ligand_pose)
    if not p.is_file():
        raise FileNotFoundError(f"Ligand pose file not found: {p}")

    ext = p.suffix.lower()

    # Try RDKit first if SMILES is available or file has 3D coords
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = None
        if ext in (".pdb", ".ent"):
            mol = Chem.MolFromPDBFile(str(p), removeHs=False)
        elif ext in (".sdf", ".mol"):
            mol = Chem.MolFromMolFile(str(p), removeHs=False)

        if mol is not None and ligand_smiles:
            ref = Chem.MolFromSmiles(ligand_smiles)
            if ref is not None:
                try:
                    mol_assigned = AllChem.AssignBondOrdersFromTemplate(ref, mol)
                    mol_h = Chem.AddHs(mol_assigned, addCoords=True)
                    atoms = []
                    conf = mol_h.GetConformer()
                    for at in mol_h.GetAtoms():
                        pos = conf.GetAtomPosition(at.GetIdx())
                        atoms.append(ClusterAtom(
                            element=at.GetSymbol().capitalize(),
                            coords=(pos.x, pos.y, pos.z),
                            atom_name=f"{at.GetSymbol()}{at.GetIdx()+1}",
                            res_name="LIG",
                            res_seq=1
                        ))
                    if atoms:
                        return atoms
                except Exception:
                    pass

        if mol is not None:
            # Check if hydrogens are already present
            has_h = any(at.GetSymbol() == "H" for at in mol.GetAtoms())
            if not has_h:
                try:
                    mol_h = Chem.AddHs(mol, addCoords=True)
                    conf = mol_h.GetConformer()
                    atoms = []
                    for at in mol_h.GetAtoms():
                        pos = conf.GetAtomPosition(at.GetIdx())
                        atoms.append(ClusterAtom(
                            element=at.GetSymbol().capitalize(),
                            coords=(pos.x, pos.y, pos.z),
                            atom_name=f"{at.GetSymbol()}{at.GetIdx()+1}",
                            res_name="LIG",
                            res_seq=1
                        ))
                    if atoms:
                        return atoms
                except Exception:
                    pass
    except ImportError:
        pass

    # Fallback to direct PDB atom parsing
    raw_atoms = parse_pdb_atoms(p)
    result = []
    for idx, at in enumerate(raw_atoms):
        result.append(ClusterAtom(
            element=at["element"].capitalize(),
            coords=at["coords"],
            atom_name=at["atom_name"],
            res_name=at.get("res_name", "LIG"),
            res_seq=at.get("res_seq", 1),
        ))
    return result


def extract_qm_cluster(
    receptor_pdb: Union[str, Path],
    ligand_pose: Union[str, Path, Sequence[Tuple[str, Tuple[float, float, float]]]],
    model_type: str = "minimal",
    target_residue: str = "THR309",
    cutoff_radius: float = 4.5,
    ligand_smiles: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: int = 1,
    freeze_backbone: bool = True,
    cluster_name: Optional[str] = None,
) -> QMCluster:
    """Extracts an active-site QM cluster with appropriate hydrogen valence capping.

    Parameters
    ----------
    receptor_pdb : str or Path
        Path to receptor structure (PDB).
    ligand_pose : str, Path, or coordinate sequence
        Ligand pose structure.
    model_type : str
        'minimal': Ligand + target nucleophile residue (capped with H, ~30-50 atoms).
        'extended': Ligand + all pocket residues within cutoff_radius (~100-150 atoms).
    target_residue : str
        Residue label, e.g. 'THR309', 'THR309:A', or 'CYS145'.
    cutoff_radius : float
        Radial cutoff in Angstroms for the extended cluster model (default 4.5 A).
    ligand_smiles : str or None
        SMILES for hydrogen assignment if ligand pose only has heavy atoms.
    charge : int or None
        Net charge of the cluster. If None, automatically computed.
    multiplicity : int
        Spin multiplicity (default 1 for closed-shell singlet).
    freeze_backbone : bool
        If True, marks non-reactive backbone atoms as frozen in extended models.
    cluster_name : str or None
        Identifier for the cluster.
    """
    rec_atoms = parse_pdb_atoms(receptor_pdb)
    lig_atoms = _prepare_ligand_atoms(ligand_pose, ligand_smiles=ligand_smiles)

    # Parse target residue number and chain
    m = re.match(r"^([A-Za-z]+)?(\d+)(?::([A-Za-z0-9]))?", target_residue.strip())
    target_resname = m.group(1).upper() if m and m.group(1) else "THR"
    target_resnum = int(m.group(2)) if m and m.group(2) else 309
    target_chain = m.group(3) if m and m.group(3) else None

    # Filter target residue atoms
    target_atoms_dict = {}
    for a in rec_atoms:
        if a["res_seq"] == target_resnum:
            if target_chain is None or a["chain_id"] == target_chain:
                target_atoms_dict[a["atom_name"].upper()] = a

    if not target_atoms_dict:
        # Fallback search by residue sequence number alone
        for a in rec_atoms:
            if a["res_seq"] == target_resnum:
                target_atoms_dict[a["atom_name"].upper()] = a

    cluster_atoms: List[ClusterAtom] = []

    # 1. Add ligand atoms
    cluster_atoms.extend(lig_atoms)

    # Identify nucleophile heteroatom name
    nucl_atom_names = REACTIVE_NUCLEOPHILES.get(target_resname, ("OG1", "OG", "SG", "NZ", "OH"))

    if model_type.lower() == "minimal":
        # Add target residue atoms
        for at_name, at in target_atoms_dict.items():
            cluster_atoms.append(ClusterAtom(
                element=at["element"].capitalize(),
                coords=at["coords"],
                atom_name=at["atom_name"],
                res_name=at["res_name"],
                res_seq=at["res_seq"],
                chain_id=at["chain_id"],
                is_cap=False,
                is_frozen=False,
            ))

        # Valence capping for cleaved peptide bonds:
        # N-terminus cap: cleaved bond between N(i) and C(i-1)
        prev_res_c = next((a for a in rec_atoms if a["res_seq"] == target_resnum - 1 and a["atom_name"].upper() == "C"), None)
        n_atom = target_atoms_dict.get("N")
        if n_atom:
            if prev_res_c:
                cap_coord = _create_capping_hydrogen(n_atom["coords"], prev_res_c["coords"], target_bond_length=1.01)
            else:
                # Default displacement along -Z
                cap_coord = (n_atom["coords"][0], n_atom["coords"][1], n_atom["coords"][2] - 1.01)
            cluster_atoms.append(ClusterAtom(
                element="H",
                coords=cap_coord,
                atom_name="HN_CAP",
                res_name=target_resname,
                res_seq=target_resnum,
                is_cap=True,
                is_frozen=False,
            ))

        # C-terminus cap: cleaved bond between C(i) and N(i+1)
        next_res_n = next((a for a in rec_atoms if a["res_seq"] == target_resnum + 1 and a["atom_name"].upper() == "N"), None)
        c_atom = target_atoms_dict.get("C")
        if c_atom:
            if next_res_n:
                cap_coord = _create_capping_hydrogen(c_atom["coords"], next_res_n["coords"], target_bond_length=1.09)
            else:
                cap_coord = (c_atom["coords"][0], c_atom["coords"][1], c_atom["coords"][2] + 1.09)
            cluster_atoms.append(ClusterAtom(
                element="H",
                coords=cap_coord,
                atom_name="HC_CAP",
                res_name=target_resname,
                res_seq=target_resnum,
                is_cap=True,
                is_frozen=False,
            ))

    else:
        # Extended Pocket Model: Select all residues within cutoff_radius of any ligand heavy atom
        lig_heavy_coords = [a.coords for a in lig_atoms if a.element != "H"]
        if not lig_heavy_coords:
            lig_heavy_coords = [a.coords for a in lig_atoms]

        pocket_res_keys = set()
        for a in rec_atoms:
            for lc in lig_heavy_coords:
                if _euclidean_distance(a["coords"], lc) <= cutoff_radius:
                    pocket_res_keys.add((a["res_name"], a["res_seq"], a["chain_id"]))
                    break

        # Ensure target residue is included
        pocket_res_keys.add((target_resname, target_resnum, target_chain or "A"))

        # Add all atoms from selected residues
        for a in rec_atoms:
            key = (a["res_name"], a["res_seq"], a["chain_id"])
            if key in pocket_res_keys:
                is_target = (a["res_seq"] == target_resnum)
                is_bb = a["atom_name"].upper() in ("N", "CA", "C", "O")
                freeze = freeze_backbone and is_bb and not is_target
                cluster_atoms.append(ClusterAtom(
                    element=a["element"].capitalize(),
                    coords=a["coords"],
                    atom_name=a["atom_name"],
                    res_name=a["res_name"],
                    res_seq=a["res_seq"],
                    chain_id=a["chain_id"],
                    is_cap=False,
                    is_frozen=freeze,
                ))

        # Check and cap cleaved peptide bonds between included and excluded residues
        for rname, rseq, rchain in pocket_res_keys:
            # Check N-terminus connection to rseq - 1
            if (rname, rseq - 1, rchain) not in pocket_res_keys:
                n_at = next((a for a in rec_atoms if a["res_seq"] == rseq and a["atom_name"].upper() == "N"), None)
                c_prev = next((a for a in rec_atoms if a["res_seq"] == rseq - 1 and a["atom_name"].upper() == "C"), None)
                if n_at and c_prev:
                    cap_crd = _create_capping_hydrogen(n_at["coords"], c_prev["coords"], target_bond_length=1.01)
                    cluster_atoms.append(ClusterAtom(
                        element="H",
                        coords=cap_crd,
                        atom_name="HN_CAP",
                        res_name=rname,
                        res_seq=rseq,
                        is_cap=True,
                        is_frozen=freeze_backbone,
                    ))

            # Check C-terminus connection to rseq + 1
            if (rname, rseq + 1, rchain) not in pocket_res_keys:
                c_at = next((a for a in rec_atoms if a["res_seq"] == rseq and a["atom_name"].upper() == "C"), None)
                n_next = next((a for a in rec_atoms if a["res_seq"] == rseq + 1 and a["atom_name"].upper() == "N"), None)
                if c_at and n_next:
                    cap_crd = _create_capping_hydrogen(c_at["coords"], n_next["coords"], target_bond_length=1.09)
                    cluster_atoms.append(ClusterAtom(
                        element="H",
                        coords=cap_crd,
                        atom_name="HC_CAP",
                        res_name=rname,
                        res_seq=rseq,
                        is_cap=True,
                        is_frozen=freeze_backbone,
                    ))

    # Identify nucleophile atom index (0-based) in cluster_atoms
    nucl_idx = None
    for idx, at in enumerate(cluster_atoms):
        if at.res_seq == target_resnum and at.atom_name.upper() in nucl_atom_names:
            nucl_idx = idx
            break

    # If not found by name, search for any heteroatom (O, S, N) in target residue
    if nucl_idx is None:
        for idx, at in enumerate(cluster_atoms):
            if at.res_seq == target_resnum and at.element in ("O", "S", "N") and not at.is_cap:
                nucl_idx = idx
                break

    # Identify electrophile atom index in ligand
    # Choose ligand heavy atom closest to the nucleophile
    electrophile_idx = None
    if nucl_idx is not None:
        nucl_crd = cluster_atoms[nucl_idx].coords
        min_d = 999.0
        for idx, at in enumerate(cluster_atoms):
            if at.res_name == "LIG" and at.element in ("C", "N", "O", "S"):
                d = _euclidean_distance(nucl_crd, at.coords)
                if d < min_d:
                    min_d = d
                    electrophile_idx = idx

    # Net charge default
    net_charge = charge if charge is not None else 0

    c_name = cluster_name or f"QM_Cluster_{target_residue}_{model_type}"

    return QMCluster(
        name=c_name,
        atoms=cluster_atoms,
        charge=net_charge,
        multiplicity=multiplicity,
        nucleophile_idx=nucl_idx,
        electrophile_idx=electrophile_idx,
        model_type=model_type,
        target_residue=target_residue,
        metadata={
            "target_residue": target_residue,
            "n_ligand_atoms": len(lig_atoms),
            "total_atoms": len(cluster_atoms),
            "cutoff_radius": cutoff_radius if model_type == "extended" else None,
        }
    )

