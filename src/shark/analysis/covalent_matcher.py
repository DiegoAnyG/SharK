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

    @property
    def summary(self) -> str:
        tag = "NAC" if self.is_nac else "PROXIMAL"
        rank_str = f" [Warhead Rank {self.warhead_rank}]" if self.warhead_rank else ""
        return (
            f"{self.nucleophile.residue_label} ({self.nucleophile.atom_name}) <--> "
            f"Ligand Atom #{self.ligand_atom_index} ({self.ligand_atom_element}){rank_str}: "
            f"{self.distance_angstrom:.2f} A ({tag}, Feasibility: {self.feasibility_score:.2f})"
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

    contacts: List[CovalentContact] = []
    for nucl in pocket_nucls:
        for idx, elem, crd, cand_obj in eval_lig_atoms:
            dist = _euclidean_distance(nucl.coordinates, crd)
            if dist <= pocket_cutoff:
                is_nac = dist <= nac_cutoff
                feas = _calculate_feasibility_score(dist, nac_cutoff=nac_cutoff)
                rank = getattr(cand_obj, "rank", None) if cand_obj else None
                omega_k = getattr(cand_obj, "local_electrophilicity", None) if cand_obj else None

                contacts.append(CovalentContact(
                    nucleophile=nucl,
                    ligand_atom_index=idx,
                    ligand_atom_element=elem,
                    distance_angstrom=round(dist, 3),
                    is_nac=is_nac,
                    feasibility_score=feas,
                    warhead_rank=rank,
                    local_electrophilicity=omega_k
                ))

    # Sort contacts by feasibility descending
    contacts.sort(key=lambda c: (c.is_nac, c.feasibility_score, -(c.distance_angstrom)), reverse=True)
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
        summary=summary
    )
