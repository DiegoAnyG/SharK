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


AMINO_ACID_PKA = {
    'ASP': 3.9,
    'GLU': 4.2,
    'HIS': 6.5,
    'CYS': 8.3,
    'TYR': 10.0,
    'LYS': 10.5,
    'ARG': 12.5,
    'THR': 13.5,
    'SER': 13.5,
}

Z_MAP = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19, 'CA': 20, 'BR': 35, 'I': 53
}


def _normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-6 else (0.0, 0.0, 1.0)


def _place_tetrahedral_hydrogen(
    central_coord: Tuple[float, float, float],
    neighbor_coords: Sequence[Tuple[float, float, float]],
    target_bond_length: float = 1.09,
) -> Tuple[float, float, float]:
    """Places a hydrogen opposite to given neighbors in tetrahedral geometry."""
    if not neighbor_coords:
        return (central_coord[0], central_coord[1], central_coord[2] + target_bond_length)
    sum_dirs = [0.0, 0.0, 0.0]
    for nc in neighbor_coords:
        v = (nc[0] - central_coord[0], nc[1] - central_coord[1], nc[2] - central_coord[2])
        u = _normalize_vector(v)
        sum_dirs[0] += u[0]
        sum_dirs[1] += u[1]
        sum_dirs[2] += u[2]
    inv_dir = _normalize_vector((-sum_dirs[0], -sum_dirs[1], -sum_dirs[2]))
    return (
        round(central_coord[0] + target_bond_length * inv_dir[0], 4),
        round(central_coord[1] + target_bond_length * inv_dir[1], 4),
        round(central_coord[2] + target_bond_length * inv_dir[2], 4),
    )


def _place_methyl_hydrogens(
    carbon_coord: Tuple[float, float, float],
    parent_coord: Tuple[float, float, float],
    target_bond_length: float = 1.09,
) -> List[Tuple[float, float, float]]:
    """Places 3 methyl hydrogens in a tripod around the parent-carbon bond vector."""
    axis = _normalize_vector((carbon_coord[0] - parent_coord[0], carbon_coord[1] - parent_coord[1], carbon_coord[2] - parent_coord[2]))
    if abs(axis[0]) < 0.9:
        perp1 = _normalize_vector((0.0, -axis[2], axis[1]))
    else:
        perp1 = _normalize_vector((-axis[1], axis[0], 0.0))
    perp2 = _normalize_vector((
        axis[1] * perp1[2] - axis[2] * perp1[1],
        axis[2] * perp1[0] - axis[0] * perp1[2],
        axis[0] * perp1[1] - axis[1] * perp1[0],
    ))
    cos_t = 1.0 / 3.0
    sin_t = math.sqrt(8.0) / 3.0
    coords = []
    for phi_deg in (0.0, 120.0, 240.0):
        phi_rad = math.radians(phi_deg)
        cp = math.cos(phi_rad)
        sp = math.sin(phi_rad)
        dr = (
            axis[0] * cos_t + (perp1[0] * cp + perp2[0] * sp) * sin_t,
            axis[1] * cos_t + (perp1[1] * cp + perp2[1] * sp) * sin_t,
            axis[2] * cos_t + (perp1[2] * cp + perp2[2] * sp) * sin_t,
        )
        u = _normalize_vector(dr)
        coords.append((
            round(carbon_coord[0] + target_bond_length * u[0], 4),
            round(carbon_coord[1] + target_bond_length * u[1], 4),
            round(carbon_coord[2] + target_bond_length * u[2], 4),
        ))
    return coords


def _place_hydroxyl_hydrogen(
    oxygen_coord: Tuple[float, float, float],
    carbon_coord: Tuple[float, float, float],
    reference_coord: Optional[Tuple[float, float, float]] = None,
    target_bond_length: float = 0.96,
) -> Tuple[float, float, float]:
    """Places a hydroxyl hydrogen with ~105 deg bent geometry."""
    axis = _normalize_vector((oxygen_coord[0] - carbon_coord[0], oxygen_coord[1] - carbon_coord[1], oxygen_coord[2] - carbon_coord[2]))
    if reference_coord:
        ref_v = _normalize_vector((reference_coord[0] - oxygen_coord[0], reference_coord[1] - oxygen_coord[1], reference_coord[2] - oxygen_coord[2]))
        dot = ref_v[0] * axis[0] + ref_v[1] * axis[1] + ref_v[2] * axis[2]
        perp = _normalize_vector((ref_v[0] - dot * axis[0], ref_v[1] - dot * axis[1], ref_v[2] - dot * axis[2]))
    else:
        if abs(axis[0]) < 0.9:
            perp = _normalize_vector((0.0, -axis[2], axis[1]))
        else:
            perp = _normalize_vector((-axis[1], axis[0], 0.0))
    cos_a = math.cos(math.radians(105.0))
    sin_a = math.sin(math.radians(105.0))
    dr = (
        axis[0] * cos_a + perp[0] * sin_a,
        axis[1] * cos_a + perp[1] * sin_a,
        axis[2] * cos_a + perp[2] * sin_a,
    )
    u = _normalize_vector(dr)
    return (
        round(oxygen_coord[0] + target_bond_length * u[0], 4),
        round(oxygen_coord[1] + target_bond_length * u[1], 4),
        round(oxygen_coord[2] + target_bond_length * u[2], 4),
    )


def _reconstruct_target_residue_hydrogens(
    target_atoms_dict: dict[str, dict],
    res_name: str,
    res_num: int,
    chain_id: str = "A",
    ph: float = 7.4,
) -> Tuple[List[ClusterAtom], int]:
    """Reconstructs missing sidechain and backbone hydrogens based on pH and residue chemistry.
    
    Returns:
        (new_hydrogen_atoms, formal_charge_delta)
    """
    res = res_name.upper()
    existing = {k.upper() for k in target_atoms_dict.keys()}
    has_h = any(at.get("element", "").upper() == "H" for at in target_atoms_dict.values())
    if has_h:
        return [], 0

    new_atoms = []
    charge = 0

    ca = target_atoms_dict.get("CA")
    n = target_atoms_dict.get("N")
    c = target_atoms_dict.get("C")
    cb = target_atoms_dict.get("CB")

    # 1. Alpha-hydrogen (HA)
    if ca and "HA" not in existing and "H_A" not in existing:
        neighbors = [a["coords"] for a in (n, c, cb) if a]
        if neighbors:
            crd = _place_tetrahedral_hydrogen(ca["coords"], neighbors, target_bond_length=1.09)
            new_atoms.append(ClusterAtom(
                element="H", coords=crd, atom_name="HA",
                res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
            ))

    # 2. Sidechain hydrogens depending on residue type
    if res == "THR":
        cg2 = target_atoms_dict.get("CG2")
        og1 = target_atoms_dict.get("OG1")
        if cb and "HB" not in existing:
            neighbors = [a["coords"] for a in (ca, cg2, og1) if a]
            if neighbors:
                crd = _place_tetrahedral_hydrogen(cb["coords"], neighbors, target_bond_length=1.09)
                new_atoms.append(ClusterAtom(
                    element="H", coords=crd, atom_name="HB",
                    res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
                ))
        if cg2 and cb and not any(k.startswith("HG2") for k in existing):
            m_coords = _place_methyl_hydrogens(cg2["coords"], cb["coords"])
            for idx, mc in enumerate(m_coords, 1):
                new_atoms.append(ClusterAtom(
                    element="H", coords=mc, atom_name=f"HG2{idx}",
                    res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
                ))
        pka_thr = AMINO_ACID_PKA.get("THR", 13.5)
        if og1 and cb and ph < pka_thr and not any(k.startswith("HG1") for k in existing):
            h_crd = _place_hydroxyl_hydrogen(og1["coords"], cb["coords"])
            new_atoms.append(ClusterAtom(
                element="H", coords=h_crd, atom_name="HG1",
                res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
            ))

    elif res == "SER":
        og = target_atoms_dict.get("OG")
        if cb and ca and not any(k.startswith("HB") for k in existing):
            neighbors = [ca["coords"]]
            if og:
                neighbors.append(og["coords"])
            crd = _place_tetrahedral_hydrogen(cb["coords"], neighbors, target_bond_length=1.09)
            new_atoms.append(ClusterAtom(
                element="H", coords=crd, atom_name="HB1",
                res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
            ))
        pka_ser = AMINO_ACID_PKA.get("SER", 13.5)
        if og and cb and ph < pka_ser and not any(k.startswith("HG") for k in existing):
            h_crd = _place_hydroxyl_hydrogen(og["coords"], cb["coords"])
            new_atoms.append(ClusterAtom(
                element="H", coords=h_crd, atom_name="HG",
                res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
            ))

    elif res == "CYS":
        sg = target_atoms_dict.get("SG")
        if cb and ca and not any(k.startswith("HB") for k in existing):
            neighbors = [ca["coords"]]
            if sg:
                neighbors.append(sg["coords"])
            crd = _place_tetrahedral_hydrogen(cb["coords"], neighbors, target_bond_length=1.09)
            new_atoms.append(ClusterAtom(
                element="H", coords=crd, atom_name="HB1",
                res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
            ))
        pka_cys = AMINO_ACID_PKA.get("CYS", 8.3)
        if sg and cb:
            if ph < pka_cys and not any(k.startswith("HG") for k in existing):
                h_crd = _place_hydroxyl_hydrogen(sg["coords"], cb["coords"], target_bond_length=1.34)
                new_atoms.append(ClusterAtom(
                    element="H", coords=h_crd, atom_name="HG",
                    res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
                ))
            else:
                charge -= 1

    elif res == "LYS":
        pka_lys = AMINO_ACID_PKA.get("LYS", 10.5)
        if ph < pka_lys:
            charge += 1

    elif res in ("ASP", "GLU"):
        pka_acid = AMINO_ACID_PKA.get(res, 4.0)
        if ph >= pka_acid:
            charge -= 1

    elif res == "HIS":
        pka_his = AMINO_ACID_PKA.get("HIS", 6.5)
        if ph < pka_his:
            charge += 1

    # 3. Backbone amide hydrogen (H)
    if n and "H" not in existing and "HN" not in existing:
        ref_pts = [ca["coords"]] if ca else []
        crd = _place_tetrahedral_hydrogen(n["coords"], ref_pts, target_bond_length=1.01)
        new_atoms.append(ClusterAtom(
            element="H", coords=crd, atom_name="H",
            res_name=res_name, res_seq=res_num, chain_id=chain_id, is_cap=False
        ))

    return new_atoms, charge


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
    ph: float = 7.4,
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
    ph : float
        Solution pH for residue protonation and microstate assignment (default: 7.4).
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
    charge_delta = 0

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

        # Reconstruct missing sidechain and backbone hydrogens if not present in input
        reconstructed_h, charge_delta = _reconstruct_target_residue_hydrogens(
            target_atoms_dict=target_atoms_dict,
            res_name=target_resname,
            res_num=target_resnum,
            chain_id=target_chain or "A",
            ph=ph,
        )
        cluster_atoms.extend(reconstructed_h)

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
    net_charge = charge if charge is not None else (charge_delta if model_type.lower() == "minimal" else 0)

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

