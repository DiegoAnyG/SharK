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
class ReactiveGeometryResult:
    """Detailed evaluation of local reactive trajectory geometry."""
    distance_score: float
    angle_score: Optional[float]
    activation_score: float
    electrophilicity_score: Optional[float]
    rgi: Optional[float]
    rgi_partial: float
    is_complete: bool
    warnings: List[str] = field(default_factory=list)


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
    # Refactored Reactive Geometry Index (RGI) attributes
    rgi: Optional[float] = None
    rgi_partial: float = 0.0
    is_geometry_complete: bool = True
    candidate_catalytic_base_present: bool = False
    candidate_activation_partner: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        tag = "NAC" if self.is_nac else "PROXIMAL"
        rank_str = f" [Warhead Rank {self.warhead_rank}]" if self.warhead_rank else ""
        angle_str = f", Angle: {self.burgi_dunitz_angle:.1f}°" if self.burgi_dunitz_angle is not None else ""
        rgi_val = self.rgi if self.rgi is not None else self.composite_feasibility
        rgi_label = "RGI" if self.is_geometry_complete else "RGI (partial)"
        rgi_str = f", {rgi_label}: {rgi_val:.2f}" if rgi_val > 0 else ""
        base_lbl = self.candidate_activation_partner or self.catalytic_dyad_residue
        dyad_str = f" [Candidate Base: {base_lbl}]" if (self.candidate_catalytic_base_present or self.catalytic_dyad_present) else ""
        return (
            f"{self.nucleophile.residue_label} ({self.nucleophile.atom_name}) <--> "
            f"Ligand Atom #{self.ligand_atom_index} ({self.ligand_atom_element}){rank_str}: "
            f"{self.distance_angstrom:.2f} A ({tag}, Feasibility: {self.feasibility_score:.2f}{rgi_str}{angle_str}{dyad_str})"
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
    pocket_residue_atoms: List[dict] = field(default_factory=list)


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
    lig_atoms: List[Tuple[int, str, Tuple[float, float, float]]],
    rdkit_mol: Optional[object] = None
) -> Tuple[Optional[Tuple[float, float, float]], bool]:
    """Identifies a bonded neighbor atom in the ligand to determine the trajectory angle.

    Returns
    -------
    coords : tuple of float or None
    is_from_bond_graph : bool
        True if derived from real molecular bond connectivity; False if geometric fallback.
    """
    if rdkit_mol is not None:
        try:
            atom = rdkit_mol.GetAtomWithIdx(target_idx)
            neighbors = atom.GetNeighbors()
            if neighbors:
                # Prioritize heavy heteroatoms (O, N, S) over carbon
                sorted_neighs = sorted(neighbors, key=lambda n: (0 if n.GetSymbol() in ("O", "N", "S") else 1, n.GetIdx()))
                chosen_idx = sorted_neighs[0].GetIdx()
                conf = rdkit_mol.GetConformer()
                pos = conf.GetAtomPosition(chosen_idx)
                return (pos.x, pos.y, pos.z), True
        except Exception:
            pass

    target_crd = next((crd for idx, _, crd in lig_atoms if idx == target_idx), None)
    if target_crd is None:
        return None, False
    candidates = []
    for idx, elem, crd in lig_atoms:
        if idx == target_idx:
            continue
        d = _euclidean_distance(target_crd, crd)
        if 0.90 <= d <= 1.95:
            # Prioritize O or N (e.g. carbonyl or furoxan system)
            priority = 0 if elem in ("O", "N") else 1
            candidates.append((priority, d, crd))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2], False
    return None, False


def _calculate_burgi_dunitz_angle(
    nucl_coord: Tuple[float, float, float],
    lig_coord: Tuple[float, float, float],
    adj_coord: Optional[Tuple[float, float, float]]
) -> Optional[float]:
    """Calculates the nucleophilic attack angle Nu ... C_target - C_adjacent."""
    if adj_coord is None or not isinstance(adj_coord, (list, tuple)) or len(adj_coord) < 3:
        return None
    if adj_coord[0] is None or adj_coord[1] is None or adj_coord[2] is None:
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


def _scan_candidate_catalytic_base(
    nucl_coord: Tuple[float, float, float],
    receptor_atoms: List[dict],
    nucl_res_seq: int,
    nucl_chain: str,
    max_distance: float = 3.8
) -> Tuple[bool, Optional[str]]:
    """Scans for nearby acidic/basic residues that may serve as candidate activation partners."""
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


# Backward-compatible alias
_scan_catalytic_dyad = _scan_candidate_catalytic_base


def compute_reactive_geometry_index(
    dist_feasibility: float,
    angle_deg: Optional[float],
    nucl_resname: str,
    candidate_base_present: bool = False,
    omega_k: Optional[float] = None,
    theta0: float = 107.0,
    sigma_theta: float = 14.0,
) -> ReactiveGeometryResult:
    """Computes the Reactive Geometry Index (RGI = f_d * f_theta * f_act * f_elec).

    Strictly sets missing metrics to None without assuming favorable defaults.
    """
    warnings = []

    # 1. Bürgi-Dunitz angular factor
    if angle_deg is not None:
        f_ang = math.exp(-((angle_deg - theta0) ** 2) / (2.0 * (sigma_theta ** 2)))
        f_ang = round(f_ang, 4)
    else:
        f_ang = None
        warnings.append("Attack angle is missing or undefined; angle factor not evaluated.")

    # 2. Catalytic activation / microenvironment factor
    res = nucl_resname.upper()
    if res == "CYS":
        # Neutral thiol in unverified microenvironment is not 1.0
        f_act = 0.85 if candidate_base_present else 0.70
    elif res in ("SER", "THR"):
        f_act = 0.90 if candidate_base_present else 0.35
    elif res in ("LYS", "TYR"):
        f_act = 0.75 if candidate_base_present else 0.40
    elif res == "HIS":
        f_act = 0.85
    else:
        f_act = 0.50

    # 3. Electrophilicity weighting
    if omega_k is not None:
        if omega_k > 0:
            f_elec = round(min(1.0, max(0.5, omega_k / 1.5)), 4)
        else:
            f_elec = 0.50
    else:
        f_elec = None
        warnings.append("Electrophilicity index is missing; electrophilicity factor not evaluated.")

    is_complete = (f_ang is not None and f_elec is not None)
    if is_complete:
        rgi = round(dist_feasibility * f_ang * f_act * f_elec, 4)
    else:
        rgi = None

    # Partial heuristic index using available factors
    rgi_partial = round(
        dist_feasibility
        * (f_ang if f_ang is not None else 1.0)
        * f_act
        * (f_elec if f_elec is not None else 1.0),
        4
    )

    return ReactiveGeometryResult(
        distance_score=round(dist_feasibility, 4),
        angle_score=f_ang,
        activation_score=round(f_act, 4),
        electrophilicity_score=f_elec,
        rgi=rgi,
        rgi_partial=rgi_partial,
        is_complete=is_complete,
        warnings=warnings
    )


def _compute_composite_feasibility(
    dist_feasibility: float,
    angle_deg: Optional[float],
    nucl_resname: str,
    dyad_present: bool,
    omega_k: Optional[float] = None
) -> Tuple[float, Optional[float], float]:
    """Backward-compatible adapter for older callers."""
    res = compute_reactive_geometry_index(
        dist_feasibility=dist_feasibility,
        angle_deg=angle_deg,
        nucl_resname=nucl_resname,
        candidate_base_present=dyad_present,
        omega_k=omega_k
    )
    reported_cfi = res.rgi if res.rgi is not None else res.rgi_partial
    return reported_cfi, res.angle_score, res.activation_score


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

    # Normalize ligand coordinates if passed as (idx, elem, coords) tuples
    clean_coords = []
    for c in ligand_coords:
        if isinstance(c, (list, tuple)) and len(c) == 3 and isinstance(c[2], (list, tuple)):
            clean_coords.append(c[2])
        else:
            clean_coords.append(c)

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
        min_d = min(_euclidean_distance(nucl_crd, lig_crd) for lig_crd in clean_coords)

        is_target_match = (
            target_num_filter is not None
            and at["res_seq"] == target_num_filter
            and (target_res_filter is None or r_name == target_res_filter)
            and min_d <= 15.0
        )

        if min_d <= pocket_cutoff or is_target_match:
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
    ligand_name: str = "",
    rdkit_mol: Optional[object] = None
) -> CovalentMatchReport:
    """Performs full Near-Attack Conformation (NAC) matching between pocket nucleophiles and warhead."""
    lig_atoms = parse_ligand_pose_coordinates(ligand_pose)
    if not lig_atoms:
        return CovalentMatchReport(
            receptor_name=receptor_name or Path(receptor_pdb).stem,
            ligand_name=ligand_name or "Unknown",
            summary="No heavy atoms found in ligand pose."
        )

    # Attempt to load RDKit Mol for exact bond connectivity if not provided
    if rdkit_mol is None and isinstance(ligand_pose, (str, Path)):
        try:
            from rdkit import Chem
            p_str = str(ligand_pose)
            if p_str.endswith(".pdb"):
                rdkit_mol = Chem.MolFromPDBFile(p_str, removeHs=False)
            elif p_str.endswith(".mol2"):
                rdkit_mol = Chem.MolFromMol2File(p_str, removeHs=False)
            elif p_str.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(p_str, removeHs=False)
                rdkit_mol = next(suppl, None)
        except Exception:
            rdkit_mol = None

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
    target_res_filter = None
    target_num_filter = None
    if target_residue:
        m = re.match(r"^([A-Za-z]+)?(\d+)?", target_residue.strip())
        if m:
            if m.group(1):
                target_res_filter = m.group(1).upper()
            if m.group(2):
                target_num_filter = int(m.group(2))

    for nucl in pocket_nucls:
        dyad_present, dyad_lbl = _scan_candidate_catalytic_base(
            nucl.coordinates, rec_atoms, nucl.residue_number, nucl.chain_id
        )
        is_target_nucl = bool(
            target_num_filter is not None
            and nucl.residue_number == target_num_filter
            and (target_res_filter is None or nucl.residue_name == target_res_filter)
        )
        for idx, elem, crd, cand_obj in eval_lig_atoms:
            dist = _euclidean_distance(nucl.coordinates, crd)
            if dist <= pocket_cutoff or (is_target_nucl and dist <= 12.0):
                is_nac = dist <= nac_cutoff
                feas = _calculate_feasibility_score(dist, nac_cutoff=nac_cutoff)
                rank = getattr(cand_obj, "rank", None) if cand_obj else None
                omega_k = getattr(cand_obj, "local_electrophilicity", None) if cand_obj else None

                adj_crd, is_from_graph = _find_adjacent_ligand_atom(idx, lig_atoms, rdkit_mol=rdkit_mol)
                angle_bd = _calculate_burgi_dunitz_angle(nucl.coordinates, crd, adj_crd)
                rgi_res = compute_reactive_geometry_index(
                    dist_feasibility=feas,
                    angle_deg=angle_bd,
                    nucl_resname=nucl.residue_name,
                    candidate_base_present=dyad_present,
                    omega_k=omega_k
                )

                warnings_list = list(rgi_res.warnings)
                if not is_from_graph and adj_crd is not None:
                    warnings_list.append("Attack angle inferred from geometric connectivity; explicit bond graph unavailable.")

                reported_cfi = rgi_res.rgi if rgi_res.rgi is not None else rgi_res.rgi_partial

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
                    activation_factor=rgi_res.activation_score,
                    composite_feasibility=reported_cfi,
                    rgi=rgi_res.rgi,
                    rgi_partial=rgi_res.rgi_partial,
                    is_geometry_complete=rgi_res.is_complete,
                    candidate_catalytic_base_present=dyad_present,
                    candidate_activation_partner=dyad_lbl,
                    warnings=warnings_list
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

    pocket_res_atoms = []
    if rec_atoms and lig_atoms:
        for a in rec_atoms:
            acrd = a["coords"]
            for _, _, lcrd in lig_atoms:
                if (acrd[0]-lcrd[0])**2 + (acrd[1]-lcrd[1])**2 + (acrd[2]-lcrd[2])**2 <= 30.25:
                    pocket_res_atoms.append({
                        "residue": f"{a['res_name']}{a['res_seq']}:{a['chain_id']}",
                        "atom": a["atom_name"],
                        "element": a.get("element") or a["atom_name"][0],
                        "x": acrd[0],
                        "y": acrd[1],
                        "z": acrd[2]
                    })
                    break
            if len(pocket_res_atoms) >= 160:
                break

    return CovalentMatchReport(
        receptor_name=rec_lbl,
        ligand_name=lig_lbl,
        pocket_nucleophiles=pocket_nucls,
        contacts=contacts,
        has_nac=has_nac,
        nac_contacts=nac_list,
        best_match=best_match,
        summary=summary,
        ligand_atoms=lig_atoms,
        pocket_residue_atoms=pocket_res_atoms
    )


def compute_binding_score(
    dock_score: Optional[float],
    ref_score: float = -6.0,
    tau: float = 1.5
) -> Optional[float]:
    """Computes normalized reversible recognition score S_bind via sigmoidal transformation.

    Parameters
    ----------
    dock_score : float or None
        Non-covalent docking binding energy (e.g. -7.244 kcal/mol).
    ref_score : float
        Sigmoidal inflection reference energy (default -6.0 kcal/mol).
    tau : float
        Sigmoidal softness parameter in kcal/mol (default 1.5 kcal/mol).

    Returns
    -------
    float or None
        S_bind in (0.0, 1.0) or None if dock_score is missing/invalid.
    """
    if dock_score is None or not math.isfinite(dock_score):
        return None
    z = (dock_score - ref_score) / tau
    z = max(-50.0, min(50.0, z))
    return round(1.0 / (1.0 + math.exp(z)), 4)


@dataclass
class TotalCovalentFeasibility:
    """Unified covalent feasibility integrating Pillars 1 (Affinity), 2 (Dynamics), and 3 (Eyring TS)."""
    cfi_pre: Optional[float]
    cfi_final: Optional[float]
    status: str
    tier: str
    affinity_score: Optional[float]
    nac_score: Optional[float]
    ts_score: Optional[float]
    docking_score: Optional[float]
    p_nac: Optional[float]
    delta_g_ts: Optional[float]
    k_chem: Optional[float]
    weights: dict
    summary: str
    warnings: List[str] = field(default_factory=list)
    rgi_static: Optional[float] = None
    completeness: Dict[str, bool] = field(default_factory=dict)
    missing_components: List[str] = field(default_factory=list)
    s_bind: Optional[float] = None
    s_nac: Optional[float] = None
    s_chem: Optional[float] = None
    # Backward compatibility fields
    cfi_total: float = 0.0
    percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes feasibility data into schema v2 format."""
        return {
            "s_bind": self.s_bind if self.s_bind is not None else self.affinity_score,
            "p_nac": self.p_nac,
            "s_chem": self.s_chem if self.s_chem is not None else self.ts_score,
            "rgi_static": self.rgi_static,
            "cfi_pre": self.cfi_pre,
            "cfi_final": self.cfi_final,
            "status": self.status,
            "tier": self.tier,
            "completeness": dict(self.completeness),
            "missing_components": list(self.missing_components),
            "docking_score": self.docking_score,
            "delta_g_ts": self.delta_g_ts,
            "k_chem": self.k_chem,
            "weights": dict(self.weights),
            "summary": self.summary,
            "warnings": list(self.warnings),
            "cfi_total": self.cfi_total,
            "percentage": self.percentage,
        }


def compute_total_covalent_feasibility(
    docking_score: Optional[float] = None,
    p_nac: Optional[float] = None,
    delta_g_ts: Optional[float] = None,
    static_cfi: Optional[float] = None,
    k_chem: Optional[float] = None,
    w_bind: float = 0.20,
    w_nac: float = 0.40,
    w_chem: float = 0.40,
    ref_score: float = -6.0,
    tau: float = 1.5,
    ts_barrier_midpoint: float = 20.0,
    ts_barrier_width: float = 2.0,
    w_aff: Optional[float] = None,
    w_ts: Optional[float] = None,
) -> TotalCovalentFeasibility:
    """Computes the Unified Covalent Feasibility Indices (CFI_pre and CFI_final).

    Strict scientific invariants enforced:
    - INV-001: Static geometry must NEVER populate P_NAC or s_nac.
    - INV-002: CFI_pre requires dynamic P_NAC. Missing P_NAC means CFI_pre is None.
    - INV-003: CFI_final requires S_bind, P_NAC, and S_chem. Missing any means CFI_final is None.
    """
    if w_aff is not None:
        w_bind = w_aff
    if w_ts is not None:
        w_chem = w_ts

    warnings: List[str] = []
    EPS = 1e-12

    # 1. Pillar 1: Reversible binding recognition score S_bind
    s_bind = compute_binding_score(docking_score, ref_score=ref_score, tau=tau)
    if s_bind is None:
        warnings.append("Reversible binding score S_bind not evaluated (missing docking score).")

    # 2. Pillar 2: Near-Attack Conformation dynamics / preorganization
    # INV-001: Static geometry must NEVER populate P_NAC or s_nac.
    if p_nac is not None and math.isfinite(p_nac):
        s_nac = min(1.0, max(0.0, p_nac))
    else:
        s_nac = None
        warnings.append("Dynamic reactive preorganization P_NAC not evaluated.")

    # Explicit separate static reactive geometry index
    rgi_static = static_cfi if (static_cfi is not None and math.isfinite(static_cfi)) else None

    # 3. Pre-reactive feasibility CFI_pre (geometric mean of S_bind and P_NAC)
    # INV-002: Requires dynamic preorganization (P_NAC). If P_NAC is missing, CFI_pre is None.
    if s_bind is not None and s_nac is not None:
        tot_pre_w = w_bind + w_nac
        w1 = w_bind / tot_pre_w if tot_pre_w > 0 else 0.35
        w2 = w_nac / tot_pre_w if tot_pre_w > 0 else 0.65
        log_cfi_pre = w1 * math.log(max(s_bind, EPS)) + w2 * math.log(max(s_nac, EPS))
        cfi_pre = round(math.exp(log_cfi_pre), 4)
    else:
        cfi_pre = None

    # 4. Pillar 3: Chemical activation kinetics S_chem & CFI_final
    s_chem = None
    cfi_final = None
    weights = {"binding": w_bind, "nac": w_nac, "chem": w_chem}

    if delta_g_ts is not None and math.isfinite(delta_g_ts):
        z = (delta_g_ts - ts_barrier_midpoint) / ts_barrier_width
        z = max(-30.0, min(30.0, z))
        s_chem = round(1.0 / (1.0 + math.exp(z)), 4)

        if k_chem is None:
            # Eyring rate at 310.15 K, kappa = 1.0
            T_k = 310.15
            rt_kcal = 0.00198720425864083 * T_k
            prefac = 2.0836619e10 * T_k
            exp_arg = -delta_g_ts / rt_kcal
            k_chem = prefac * math.exp(exp_arg) if exp_arg > -700 else 0.0

        # INV-003: CFI_final requires all 3 pillars (S_bind, P_NAC, S_chem).
        if s_bind is not None and s_nac is not None and s_chem is not None:
            v_bind = max(s_bind, EPS)
            v_nac = max(s_nac, EPS)
            v_chem = max(s_chem, EPS)
            log_cfi = w_bind * math.log(v_bind) + w_nac * math.log(v_nac) + w_chem * math.log(v_chem)
            cfi_final = round(math.exp(log_cfi), 4)
            status = "Complete covalent evaluation"
            if cfi_final >= 0.70:
                tier = "High Covalent Feasibility"
            elif cfi_final >= 0.40:
                tier = "Moderate Covalent Feasibility"
            else:
                tier = "Low Covalent Feasibility"
        else:
            cfi_final = None
            tier = "Incomplete Data"
            missing_pillars = []
            if s_bind is None:
                missing_pillars.append("reversible binding score S_bind")
            if s_nac is None:
                missing_pillars.append("dynamic preorganization P_NAC")
            if s_chem is None:
                missing_pillars.append("chemical accessibility S_chem")
            status = f"Incomplete evaluation (missing {', '.join(missing_pillars)})"
            warnings.append(f"CFI_final cannot be computed: missing {', '.join(missing_pillars)}.")
    else:
        if cfi_pre is not None:
            status = "Pending transition-state calculation"
            warnings.append("Transition-state activation barrier DeltaG‡ not evaluated; chemical step pending.")
            if cfi_pre >= 0.70:
                tier = "Pre-reactive Favorable"
            elif cfi_pre >= 0.40:
                tier = "Pre-reactive Moderate"
            else:
                tier = "Pre-reactive Unfavorable"
        else:
            tier = "Incomplete Data"
            missing_pre = []
            if s_bind is None:
                missing_pre.append("reversible binding score")
            if s_nac is None:
                missing_pre.append("dynamic preorganization")
            status = f"Incomplete evaluation (missing {', '.join(missing_pre) if missing_pre else 'required pillars'})"
            warnings.append("Transition-state activation barrier DeltaG‡ not evaluated; chemical step pending.")

    completeness = {
        "binding": s_bind is not None,
        "docking": s_bind is not None,
        "p_nac": s_nac is not None,
        "dynamic_preorganization": s_nac is not None,
        "chemical_accessibility": s_chem is not None,
        "ts_barrier": s_chem is not None,
        "cfi_pre": cfi_pre is not None,
        "cfi_final": cfi_final is not None,
    }
    missing_components = []
    if s_bind is None:
        missing_components.append("docking")
    if s_nac is None:
        missing_components.append("p_nac")
    if s_chem is None:
        missing_components.append("ts_barrier")

    # Backward compatibility fields
    cfi_total = cfi_final if cfi_final is not None else (cfi_pre if cfi_pre is not None else 0.0)
    pct = round(cfi_total * 100.0, 1)

    parts = []
    if cfi_final is not None:
        parts.append(f"CFI_final: {cfi_final:.3f} ({tier}).")
    elif cfi_pre is not None:
        parts.append(f"CFI_pre: {cfi_pre:.3f} ({tier}, CFI_final: Pending transition-state calculation).")
    else:
        parts.append("CFI: Incomplete (CFI_pre and CFI_final not available).")

    if s_bind is not None:
        parts.append(f"Pillar 1 (S_bind): {s_bind:.3f} (docking = {docking_score} kcal/mol).")
    if s_nac is not None:
        parts.append(f"Pillar 2 (P_NAC): {s_nac:.3f}.")
    elif rgi_static is not None:
        parts.append(f"Static RGI: {rgi_static:.3f} (static pose only, not dynamic P_NAC).")
    if s_chem is not None:
        parts.append(f"Pillar 3 (S_chem): {s_chem:.3f} (ΔG‡ = {delta_g_ts:.2f} kcal/mol).")

    return TotalCovalentFeasibility(
        cfi_pre=cfi_pre,
        cfi_final=cfi_final,
        status=status,
        tier=tier,
        affinity_score=s_bind,
        nac_score=s_nac,
        ts_score=s_chem,
        docking_score=docking_score,
        p_nac=p_nac,
        delta_g_ts=delta_g_ts,
        k_chem=k_chem,
        weights=weights,
        summary=" ".join(parts),
        warnings=warnings,
        rgi_static=rgi_static,
        completeness=completeness,
        missing_components=missing_components,
        s_bind=s_bind,
        s_nac=s_nac,
        s_chem=s_chem,
        cfi_total=cfi_total,
        percentage=pct,
    )

