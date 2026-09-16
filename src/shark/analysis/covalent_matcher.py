"""Near-Attack Conformation (NAC) and pocket nucleophile covalent matching engine.

Scans the receptor binding pocket (within a configurable cutoff, default 5.0 A)
to identify canonical nucleophilic residues (CYS, SER, THR, LYS, HIS, TYR).
Evaluates geometric proximity to ligand warhead electrophilic centers to detect
Near-Attack Conformations (d <= 3.5 A) and assess covalent reaction feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple, Union

# Canonical nucleophiles and their reactive atoms in covalent drug discovery
REACTIVE_NUCLEOPHILES = {
    "CYS": ("SG",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "LYS": ("NZ",),
    "HIS": ("ND1", "NE2"),
    "TYR": ("OH",),
}

# Standard Near-Attack Conformation (NAC) distance threshold in Angstroms
DEFAULT_NAC_CUTOFF = 3.5
DEFAULT_POCKET_CUTOFF = 5.0


@dataclass
class PocketNucleophile:
    """A catalytic or reactive nucleophilic residue detected in the receptor."""
    residue_name: str
    residue_number: int
    chain_id: str
    atom_name: str
    coordinates: Tuple[float, float, float]
    min_distance_to_ligand: float = 999.0

    @property
    def residue_label(self) -> str:
        """Formatted residue label, e.g., 'CYS145:A'."""
        return f"{self.residue_name}{self.residue_number}:{self.chain_id}"


@dataclass
class CovalentContact:
    """Pairwise geometric contact between a pocket nucleophile and a ligand atom."""
    nucleophile: PocketNucleophile
    ligand_atom_index: int
    ligand_atom_element: str
    distance_angstrom: float
    is_nac: bool
    feasibility_score: float
    warhead_rank: Optional[int] = None
    local_electrophilicity: Optional[float] = None
    burgi_dunitz_angle: Optional[float] = None
    catalytic_dyad_present: bool = False
    catalytic_dyad_residue: Optional[str] = None
    activation_factor: float = 1.0
    composite_feasibility: float = 0.0

    @property
    def summary(self) -> str:
        tag = "NAC" if self.is_nac else "PROXIMAL"
        rank_str = f" [Warhead Rank {self.warhead_rank}]" if self.warhead_rank else ""
        angle_str = f", Angle: {self.burgi_dunitz_angle:.1f}°" if self.burgi_dunitz_angle is not None else ""
        cfi_str = f", CFI: {self.composite_feasibility:.2f}" if self.composite_feasibility > 0 else ""
        dyad_str = f" [Dyad: {self.catalytic_dyad_residue}]" if self.catalytic_dyad_present else ""
        return (
            f"{self.nucleophile.residue_label} ({self.nucleophile.atom_name}) <--> "
            f"Ligand Atom #{self.ligand_atom_index} ({self.ligand_atom_element}){rank_str}: "
            f"{self.distance_angstrom:.2f} A ({tag}, Feasibility: {self.feasibility_score:.2f}{cfi_str}{angle_str}{dyad_str})"
        )


@dataclass
class CovalentMatchReport:
    """Summary of pocket nucleophiles and covalent Near-Attack Conformations."""
    receptor_name: str
    ligand_name: str
    pocket_nucleophiles: List[PocketNucleophile] = field(default_factory=list)
    contacts: List[CovalentContact] = field(default_factory=list)
    has_nac: bool = False
    nac_contacts: List[CovalentContact] = field(default_factory=list)
    best_match: Optional[CovalentContact] = None
    summary: str = ""
    ligand_atoms: List[Tuple[int, str, Tuple[float, float, float]]] = field(default_factory=list)


def _euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def _calculate_feasibility_score(distance: float, nac_cutoff: float = DEFAULT_NAC_CUTOFF) -> float:
    """Sigmoidal feasibility scoring based on Bruice Near-Attack Conformation distance."""
    if distance <= 0.0:
        return 0.0
    # Sigmoid centered around nac_cutoff with width parameter 0.4 A
    z = (distance - nac_cutoff) / 0.4
    score = 1.0 / (1.0 + math.exp(z))
    return round(score, 4)


def _find_adjacent_ligand_atom(
    target_idx: int,
    lig_atoms: List[Tuple[int, str, Tuple[float, float, float]]]
) -> Optional[Tuple[float, float, float]]:
    """Identifies a bonded neighbor atom in the ligand to determine the trajectory angle."""
    target_crd = next((crd for idx, _, crd in lig_atoms if idx == target_idx), None)
    if target_crd is None:
        return None
    candidates = []
    for idx, elem, crd in lig_atoms:
        if idx == target_idx:
            continue
        d = _euclidean_distance(target_crd, crd)
        if 1.05 <= d <= 1.65:
            # Prioritize O or N (e.g. carbonyl or furoxan system)
            priority = 0 if elem in ("O", "N") else 1
            candidates.append((priority, d, crd))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]
    return None


def _calculate_burgi_dunitz_angle(
    nucl_coord: Tuple[float, float, float],
    lig_coord: Tuple[float, float, float],
    adj_coord: Optional[Tuple[float, float, float]]
) -> Optional[float]:
    """Calculates the nucleophilic attack angle Nu ... C_target - C_adjacent."""
    if adj_coord is None:
        return None
    v1 = (nucl_coord[0] - lig_coord[0], nucl_coord[1] - lig_coord[1], nucl_coord[2] - lig_coord[2])
    v2 = (adj_coord[0] - lig_coord[0], adj_coord[1] - lig_coord[1], adj_coord[2] - lig_coord[2])
    n1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    n2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_theta = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (n1 * n2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return round(math.degrees(math.acos(cos_theta)), 2)


def _scan_catalytic_dyad(
    nucl_coord: Tuple[float, float, float],
    receptor_atoms: List[dict],
    nucl_res_seq: int,
    nucl_chain: str,
    max_distance: float = 3.8
) -> Tuple[bool, Optional[str]]:
    """Scans for nearby acidic/basic residues activating a nucleophile via proton abstraction."""
    CATALYTIC_BASES = {
        "HIS": ("ND1", "NE2"),
        "ASP": ("OD1", "OD2"),
        "GLU": ("OE1", "OE2"),
        "LYS": ("NZ",),
    }
    for a in receptor_atoms:
        if a["res_seq"] == nucl_res_seq and a["chain_id"] == nucl_chain:
            continue
        rname = a["res_name"].upper()
        if rname in CATALYTIC_BASES and a["atom_name"].upper() in CATALYTIC_BASES[rname]:
            d = _euclidean_distance(nucl_coord, a["coords"])
            if d <= max_distance:
                label = f"{rname}{a['res_seq']}:{a['chain_id']}"
                return True, label
    return False, None


def _compute_composite_feasibility(
    dist_feasibility: float,
    angle_deg: Optional[float],
    nucl_resname: str,
    dyad_present: bool,
    omega_k: Optional[float] = None
) -> Tuple[float, float, float]:
    """Computes angular factor, activation factor, and Composite Covalent Feasibility Index (CFI)."""
    # 1. Bürgi-Dunitz angular factor (centered at 107 deg, width 14 deg)
    if angle_deg is not None:
        f_ang = math.exp(-((angle_deg - 107.0) ** 2) / (2.0 * (14.0 ** 2)))
    else:
        f_ang = 0.85

    # 2. Catalytic activation / pKa factor
    res = nucl_resname.upper()
    if res == "CYS":
        f_act = 1.0  # Thiolate readily accessible
    elif res in ("SER", "THR"):
        f_act = 0.90 if dyad_present else 0.35  # Neutral alcohol without base is poorly reactive
    elif res in ("LYS", "TYR"):
        f_act = 0.75 if dyad_present else 0.40
    elif res == "HIS":
        f_act = 0.85
    else:
        f_act = 0.50

    # 3. Electrophilicity weighting
    if omega_k is not None and omega_k > 0:
        f_elec = min(1.0, max(0.5, omega_k / 1.5))
    else:
        f_elec = 1.0

    cfi = dist_feasibility * f_ang * f_act * f_elec
    return round(cfi, 4), round(f_ang, 3), round(f_act, 3)


def parse_pdb_atoms(pdb_path: str | Path) -> List[dict]:
    """Parses atom records from a PDB file."""
    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB file not found: {path}")

    atoms = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            record_type = line[0:6].strip()
            atom_name = line[12:16].strip()
            alt_loc = line[16:17].strip()
            if alt_loc and alt_loc not in ("A", "1", " "):
                continue  # Skip alternate conformers B, C...
            res_name = line[17:20].strip().upper()
            chain_id = line[21:22].strip() or "A"
            res_seq_str = line[22:26].strip()
            try:
                res_seq = int(res_seq_str)
            except ValueError:
                res_seq = 0
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                parts = line.split()
                dot_floats = []
                for p in parts:
                    if "." in p:
                        try:
                            dot_floats.append(float(p))
                        except ValueError:
                            pass
                if len(dot_floats) >= 3:
                    x, y, z = dot_floats[0], dot_floats[1], dot_floats[2]
                else:
                    continue

            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem:
                elem = re.sub(r"[\d\W]", "", atom_name)[:1].upper()

            atoms.append({
                "record_type": record_type,
                "atom_name": atom_name,
                "res_name": res_name,
                "chain_id": chain_id,
                "res_seq": res_seq,
                "coords": (x, y, z),
                "element": elem
            })
    return atoms


def parse_ligand_pose_coordinates(
    ligand_pose: Union[str, Path, Sequence[Tuple[float, float, float]]]
) -> List[Tuple[int, str, Tuple[float, float, float]]]:
    """Extracts heavy-atom coordinates and elements from a ligand pose file or list."""
    if isinstance(ligand_pose, (list, tuple)):
        # List of coordinates or tuples
        result = []
        for idx, item in enumerate(ligand_pose):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                result.append((idx, "C", (float(item[0]), float(item[1]), float(item[2]))))
            elif hasattr(item, "coordinates") and item.coordinates:
                elem = getattr(item, "element", "C")
                crd = item.coordinates
                result.append((idx, elem, (float(crd[0]), float(crd[1]), float(crd[2]))))
        return result

    p = Path(ligand_pose)
    if not p.is_file():
        raise FileNotFoundError(f"Ligand pose file not found: {p}")

    ext = p.suffix.lower()
    atoms = []
    if ext in (".pdb", ".pdbqt", ".ent"):
        pdb_atoms = parse_pdb_atoms(p)
        for idx, at in enumerate(pdb_atoms):
            if at["element"] != "H":
                atoms.append((idx, at["element"], at["coords"]))
    elif ext in (".sdf", ".mol"):
        # Try RDKit if available, else simple SDF coordinate reader
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(p), removeHs=False)
            if mol and mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                for i, at in enumerate(mol.GetAtoms()):
                    if at.GetSymbol().upper() != "H":
                        pos = conf.GetAtomPosition(i)
                        atoms.append((i, at.GetSymbol().upper(), (pos.x, pos.y, pos.z)))
                return atoms
        except ImportError:
            pass
        # Fallback SDF block parser
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        in_counts = False
        n_atoms = 0
        read_atoms = 0
        for line in lines:
            if "V2000" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        n_atoms = int(parts[0])
                        in_counts = True
                        continue
                    except ValueError:
                        pass
            if in_counts and read_atoms < n_atoms:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        elem = parts[3].upper()
                        if elem != "H":
                            atoms.append((read_atoms, elem, (x, y, z)))
                        read_atoms += 1
                    except ValueError:
                        pass
    return atoms


def extract_pocket_nucleophiles(
    receptor_pdb: str | Path,
    ligand_coords: Sequence[Tuple[float, float, float]],
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    target_residue: Optional[str] = None
) -> List[PocketNucleophile]:
    """Scans the receptor structure to locate nucleophiles within pocket_cutoff of ligand."""
    rec_atoms = parse_pdb_atoms(receptor_pdb)
    if not ligand_coords:
        return []

    target_res_filter = None
    target_num_filter = None
    if target_residue:
        # e.g., "CYS145", "CYS", "145"
        m = re.match(r"^([A-Za-z]+)?(\d+)?", target_residue.strip())
        if m:
            if m.group(1):
                target_res_filter = m.group(1).upper()
            if m.group(2):
                target_num_filter = int(m.group(2))

    pocket_nucleophiles = []
    for at in rec_atoms:
        r_name = at["res_name"]
        if r_name not in REACTIVE_NUCLEOPHILES:
            continue

        if target_res_filter and r_name != target_res_filter:
            continue
        if target_num_filter is not None and at["res_seq"] != target_num_filter:
            continue

        valid_atoms = REACTIVE_NUCLEOPHILES[r_name]
        if at["atom_name"] not in valid_atoms:
            continue

        # Compute minimum distance to ligand
        nucl_crd = at["coords"]
        min_d = min(_euclidean_distance(nucl_crd, lig_crd) for lig_crd in ligand_coords)

        if min_d <= pocket_cutoff:
            pocket_nucleophiles.append(PocketNucleophile(
                residue_name=r_name,
                residue_number=at["res_seq"],
                chain_id=at["chain_id"],
                atom_name=at["atom_name"],
                coordinates=nucl_crd,
                min_distance_to_ligand=round(min_d, 3)
            ))

    # Sort by minimum distance to ligand
    pocket_nucleophiles.sort(key=lambda n: n.min_distance_to_ligand)
    return pocket_nucleophiles


def match_covalent_pocket(
    receptor_pdb: str | Path,
    ligand_pose: Union[str, Path, Sequence[Tuple[float, float, float]]],
    warhead_candidates: Optional[Sequence[object]] = None,
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF,
    nac_cutoff: float = DEFAULT_NAC_CUTOFF,
    target_residue: Optional[str] = None,
    receptor_name: str = "",
    ligand_name: str = ""
) -> CovalentMatchReport:
    """Performs full Near-Attack Conformation (NAC) matching between pocket nucleophiles and warhead."""
    lig_atoms = parse_ligand_pose_coordinates(ligand_pose)
    if not lig_atoms:
        return CovalentMatchReport(
            receptor_name=receptor_name or Path(receptor_pdb).stem,
            ligand_name=ligand_name or "Unknown",
            summary="No heavy atoms found in ligand pose."
        )

    all_lig_coords = [crd for _, _, crd in lig_atoms]
    pocket_nucls = extract_pocket_nucleophiles(
        receptor_pdb=receptor_pdb,
        ligand_coords=all_lig_coords,
        pocket_cutoff=pocket_cutoff,
        target_residue=target_residue
    )

    # Filter target ligand atoms to warhead candidates if provided, else all heavy atoms
    warhead_map = {}
    if warhead_candidates:
        for cand in warhead_candidates:
            idx = getattr(cand, "atom_index", None)
            if idx is not None:
                warhead_map[idx] = cand

    eval_lig_atoms = []
    if warhead_map:
        for idx, elem, crd in lig_atoms:
            if idx in warhead_map:
                eval_lig_atoms.append((idx, elem, crd, warhead_map[idx]))
    else:
        for idx, elem, crd in lig_atoms:
            eval_lig_atoms.append((idx, elem, crd, None))

    rec_atoms = []
    try:
        rec_atoms = parse_pdb_atoms(receptor_pdb)
    except Exception:
        rec_atoms = []

    contacts: List[CovalentContact] = []
    for nucl in pocket_nucls:
        dyad_present, dyad_lbl = _scan_catalytic_dyad(
            nucl.coordinates, rec_atoms, nucl.residue_number, nucl.chain_id
        )
        for idx, elem, crd, cand_obj in eval_lig_atoms:
            dist = _euclidean_distance(nucl.coordinates, crd)
            if dist <= pocket_cutoff:
                is_nac = dist <= nac_cutoff
                feas = _calculate_feasibility_score(dist, nac_cutoff=nac_cutoff)
                rank = getattr(cand_obj, "rank", None) if cand_obj else None
                omega_k = getattr(cand_obj, "local_electrophilicity", None) if cand_obj else None

                adj_crd = _find_adjacent_ligand_atom(idx, lig_atoms)
                angle_bd = _calculate_burgi_dunitz_angle(nucl.coordinates, crd, adj_crd)
                cfi, f_ang, f_act = _compute_composite_feasibility(
                    dist_feasibility=feas,
                    angle_deg=angle_bd,
                    nucl_resname=nucl.residue_name,
                    dyad_present=dyad_present,
                    omega_k=omega_k
                )

                contacts.append(CovalentContact(
                    nucleophile=nucl,
                    ligand_atom_index=idx,
                    ligand_atom_element=elem,
                    distance_angstrom=round(dist, 3),
                    is_nac=is_nac,
                    feasibility_score=feas,
                    warhead_rank=rank,
                    local_electrophilicity=omega_k,
                    burgi_dunitz_angle=angle_bd,
                    catalytic_dyad_present=dyad_present,
                    catalytic_dyad_residue=dyad_lbl,
                    activation_factor=f_act,
                    composite_feasibility=cfi
                ))

    # Sort contacts by composite feasibility descending
    contacts.sort(key=lambda c: (c.is_nac, c.composite_feasibility, c.feasibility_score, -(c.distance_angstrom)), reverse=True)
    nac_list = [c for c in contacts if c.is_nac]
    has_nac = len(nac_list) > 0
    best_match = contacts[0] if contacts else None

    # Summary text
    rec_lbl = receptor_name or Path(receptor_pdb).stem
    lig_lbl = ligand_name or "Ligand"
    if has_nac:
        summary = (
            f"Positive Near-Attack Conformation detected for {lig_lbl} with {rec_lbl}. "
            f"{len(nac_list)} reactive contact(s) <= {nac_cutoff:.1f} A. "
            f"Best match: {best_match.summary}."
        )
    elif pocket_nucls:
        summary = (
            f"Pocket nucleophiles detected ({', '.join(n.residue_label for n in pocket_nucls)}), "
            f"but no Near-Attack Conformation met the <= {nac_cutoff:.1f} A threshold. "
            f"Closest approach: {best_match.distance_angstrom:.2f} A." if best_match else "None within range."
        )
    else:
        summary = f"No reactive pocket nucleophiles (Cys, Ser, Thr, Lys, His, Tyr) within {pocket_cutoff:.1f} A of {lig_lbl}."

    return CovalentMatchReport(
        receptor_name=rec_lbl,
        ligand_name=lig_lbl,
        pocket_nucleophiles=pocket_nucls,
        contacts=contacts,
        has_nac=has_nac,
        nac_contacts=nac_list,
        best_match=best_match,
        summary=summary,
        ligand_atoms=lig_atoms
    )

