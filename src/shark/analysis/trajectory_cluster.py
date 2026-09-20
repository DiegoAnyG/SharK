"""RMSD trajectory clustering and representative solvated snapshot extractor.

Implements the Daura et al. (GROMOS) clustering algorithm on molecular dynamics
trajectories. Aligns frames on the protein/pocket backbone, computes the pairwise
RMSD matrix of ligand heavy atoms, identifies the dominant conformational cluster,
and extracts the true medoid (centroid) snapshot in explicit solvent for downstream
covalent and quantum reactivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Sequence
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is not None:
            return iterable
        class _DummyPbar:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def update(self, *a, **k): pass
            def close(self): pass
        return _DummyPbar()

STANDARD_RESIDUES_SET = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "CYX", "HID", "HIE", "HIP", "ASH", "GLH", "SOL", "WAT", "HOH", "TIP3",
    "NA", "CL", "K", "MG", "CA", "ZN", "GDP", "GTP", "ADP", "ATP"
}


@dataclass
class ReactiveFrameRecord:
    """Per-frame reactive geometry tracking record."""
    trajectory_frame: int
    time_ps: float
    distance_angstrom: float
    attack_angle_deg: Optional[float]
    distance_score: float
    angle_score: Optional[float]
    nac_score: Optional[float]
    conformational_cluster_id: Optional[int] = None
    reactive_cluster_id: Optional[int] = None


@dataclass
class TrajectoryClusterReport:
    """Statistical summary of trajectory clustering and representative snapshot extraction."""
    total_sampled_frames: int
    num_clusters: int
    top_cluster_size: int
    top_cluster_fraction: float

    # GROMOS center vs true mathematical medoid of dominant conformational cluster
    gromos_center_frame_index: int
    gromos_center_time_ps: float
    gromos_center_time_ns: float

    medoid_frame_index: int
    medoid_time_ps: float
    medoid_time_ns: float
    cutoff_angstrom: float
    sim_time_ns: float = 0.0
    total_sim_time_ns: float = 0.0

    # Dominant conformational representative snapshots (for ground-state pocket analysis)
    snapshot_complex_pdb: Optional[Path] = None
    snapshot_receptor_pdb: Optional[Path] = None
    snapshot_ligand_pdb: Optional[Path] = None
    cluster_sizes: List[int] = field(default_factory=list)

    # Continuous ensemble reactivity
    p_nac: Optional[float] = None
    distance_proximity_score: Optional[float] = None

    # Reactive subensemble metrics
    reactive_frame_count: int = 0
    reactive_frame_fraction: Optional[float] = None
    num_reactive_clusters: int = 0
    top_reactive_cluster_size: int = 0
    top_reactive_cluster_fraction: Optional[float] = None

    # Reactive medoid properties
    reactive_medoid_frame_index: Optional[int] = None
    reactive_medoid_time_ps: Optional[float] = None
    reactive_medoid_time_ns: Optional[float] = None
    reactive_medoid_nac_score: Optional[float] = None
    reactive_medoid_distance_angstrom: Optional[float] = None
    reactive_medoid_angle_deg: Optional[float] = None

    # Reactive representative snapshots (for Tier 4 QM/TS reaction initialization)
    reactive_snapshot_complex_pdb: Optional[Path] = None
    reactive_snapshot_receptor_pdb: Optional[Path] = None
    reactive_snapshot_ligand_pdb: Optional[Path] = None

    # Full frame history and diagnostics
    frame_records: List[ReactiveFrameRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def representative_snapshot_complex_pdb(self) -> Optional[Path]:
        return self.snapshot_complex_pdb

    @property
    def representative_snapshot_receptor_pdb(self) -> Optional[Path]:
        return self.snapshot_receptor_pdb

    @property
    def representative_snapshot_ligand_pdb(self) -> Optional[Path]:
        return self.snapshot_ligand_pdb


def compute_true_medoid(
    member_indices: Sequence[int],
    distance_matrix: np.ndarray,
) -> int:
    """Finds the true medoid of a cluster that minimizes the mean intra-cluster distance.

    Parameters
    ----------
    member_indices : Sequence[int]
        Indices of frames belonging to the cluster.
    distance_matrix : np.ndarray, shape (N, N)
        Pairwise distance/RMSD matrix.

    Returns
    -------
    medoid_idx : int
        The member index that minimizes the sum/mean of distances to other members.
    """
    if len(member_indices) <= 1:
        return member_indices[0]

    members = list(member_indices)
    best_idx = members[0]
    min_mean_dist = float("inf")

    for i in members:
        dists = [distance_matrix[i, j] for j in members if j != i]
        mean_d = float(np.mean(dists)) if dists else 0.0
        if mean_d < min_mean_dist:
            min_mean_dist = mean_d
            best_idx = i

    return best_idx


def select_reactive_frames(
    records: Sequence[ReactiveFrameRecord],
    nac_min: float = 0.10,
    distance_max: float = 3.8,
) -> List[int]:
    """Identifies indices of sampled frames that satisfy Near-Attack reactive criteria."""
    reactive_indices = []
    for idx, rec in enumerate(records):
        if rec.nac_score is not None:
            if rec.nac_score >= nac_min and rec.distance_angstrom <= distance_max:
                reactive_indices.append(idx)
        elif rec.distance_angstrom <= distance_max and rec.distance_score >= nac_min:
            reactive_indices.append(idx)
    return reactive_indices


def compute_frame_nac_score(
    nucl_coord: np.ndarray | Sequence[float],
    el_coord: np.ndarray | Sequence[float],
    adj_coord: Optional[np.ndarray | Sequence[float]] = None,
    d0: float = 3.5,
    sigma_d: float = 0.5,
    theta0: float = 107.0,
    sigma_theta: float = 14.0,
) -> Tuple[Optional[float], float, Optional[float], Optional[float]]:
    """Calculates Near-Attack Conformation (NAC) score for a single frame.

    Evaluates both contact distance d and approach angle theta:
        f_d(d) = 1 / (1 + exp((d - d0) / sigma_d))
        f_theta(theta) = exp(-(theta - theta0)^2 / (2 * sigma_theta^2))
        NAC_score = f_d(d) * f_theta(theta)

    Parameters
    ----------
    nucl_coord : array-like, shape (3,)
        Cartesian coordinates of the nucleophile reactive heavy atom.
    el_coord : array-like, shape (3,)
        Cartesian coordinates of the ligand electrophilic center.
    adj_coord : array-like, shape (3,), optional
        Cartesian coordinates of an adjacent bonded heavy atom in the ligand.
    d0 : float
        Midpoint distance parameter (default: 3.5 A).
    sigma_d : float
        Distance sigmoid width parameter (default: 0.5 A).
    theta0 : float
        Optimal approach angle in degrees (default: 107.0 deg, Bürgi-Dunitz).
    sigma_theta : float
        Angle Gaussian width in degrees (default: 14.0 deg).

    Returns
    -------
    nac_score : Optional[float]
        Combined distance-plus-angle score in [0, 1] if adj_coord is given, else None.
    f_d : float
        Sigmoidal distance score.
    f_theta : Optional[float]
        Gaussian angle score (None if adj_coord is missing).
    theta_deg : Optional[float]
        Approach angle in degrees (None if adj_coord is missing).
    """
    n_pos = np.asarray(nucl_coord, dtype=float)
    e_pos = np.asarray(el_coord, dtype=float)

    d = float(np.linalg.norm(n_pos - e_pos))
    arg_d = (d - d0) / sigma_d
    arg_d = max(-50.0, min(50.0, arg_d))
    f_d = float(1.0 / (1.0 + math.exp(arg_d)))

    if adj_coord is None:
        return None, f_d, None, None

    a_pos = np.asarray(adj_coord, dtype=float)
    u = n_pos - e_pos
    v = a_pos - e_pos
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u < 1e-6 or norm_v < 1e-6:
        return None, f_d, None, None

    cos_theta = float(np.dot(u, v) / (norm_u * norm_v))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_deg = float(np.arccos(cos_theta) * (180.0 / math.pi))

    arg_theta = -((theta_deg - theta0) ** 2) / (2.0 * (sigma_theta ** 2))
    arg_theta = max(-50.0, min(0.0, arg_theta))
    f_theta = float(math.exp(arg_theta))

    nac_score = f_d * f_theta
    return nac_score, f_d, f_theta, theta_deg


def rigid_body_superposition(
    mobile_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes optimal Kabsch rotation matrix superimposing mobile onto ref.

    Centers both coordinate sets at their centroids prior to SVD.

    Parameters
    ----------
    mobile_coords : np.ndarray, shape (N, 3)
        Coordinates of the mobile structure.
    ref_coords : np.ndarray, shape (N, 3)
        Coordinates of the reference structure.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Optimal rotation matrix.
    mobile_center : np.ndarray, shape (3,)
        Centroid of the mobile coordinates.
    ref_center : np.ndarray, shape (3,)
        Centroid of the reference coordinates.
    """
    mobile = np.asarray(mobile_coords, dtype=float)
    ref = np.asarray(ref_coords, dtype=float)

    if mobile.shape != ref.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Coordinate shape mismatch: mobile {mobile.shape}, ref {ref.shape}")

    mobile_center = np.mean(mobile, axis=0)
    ref_center = np.mean(ref, axis=0)

    p = mobile - mobile_center
    q = ref - ref_center

    # Covariance matrix H = P^T Q
    H = np.dot(p.T, q)
    U, S, Vt = np.linalg.svd(H)

    # Validate reflection
    d = np.linalg.det(np.dot(Vt.T, U.T))
    E = np.eye(3)
    if d < 0:
        E[2, 2] = -1.0

    R = np.dot(Vt.T, np.dot(E, U.T))
    return R, mobile_center, ref_center


def apply_rigid_body_transformation(
    coords: np.ndarray,
    mobile_center: np.ndarray,
    R: np.ndarray,
    ref_center: np.ndarray,
) -> np.ndarray:
    """Applies rigid-body transformation: (coords - mobile_center) @ R.T + ref_center."""
    pts = np.asarray(coords, dtype=float)
    return np.dot(pts - mobile_center, R.T) + ref_center


def _detect_ligand_resname(universe) -> str:
    """Autonomously detects the small-molecule ligand residue name from the topology."""
    candidates = []
    for res in universe.residues:
        name = res.resname.strip().upper()
        if name not in STANDARD_RESIDUES_SET and len(res.atoms) >= 2:
            if name not in candidates:
                candidates.append(name)
    if not candidates:
        # Fallback to UNL or LIG
        for default_name in ("UNL", "LIG", "DRG", "MOL"):
            if default_name in universe.residues.resnames:
                return default_name
        raise ValueError("Could not autonomously identify a small-molecule ligand in the topology. Supply --ligand-selection explicitly.")
    return candidates[0]


def _export_snapshot(
    universe,
    frame_index: int,
    out_dir: Path,
    prefix: str,
    protein_selection: str,
    lig_sel_str: str,
) -> Tuple[Path, Path, Path]:
    """Helper to export complex, receptor, and ligand PDB files for a trajectory frame."""
    universe.trajectory[frame_index]
    complex_pdb = out_dir / f"{prefix}_complex.pdb"
    receptor_pdb = out_dir / f"{prefix}_receptor.pdb"
    ligand_pdb = out_dir / f"{prefix}_ligand.pdb"

    universe.select_atoms(f"({protein_selection}) or resname GDP or resname GTP or {lig_sel_str}").write(str(complex_pdb))
    universe.select_atoms(f"({protein_selection}) or resname GDP or resname GTP").write(str(receptor_pdb))
    universe.select_atoms(lig_sel_str).write(str(ligand_pdb))
    return complex_pdb, receptor_pdb, ligand_pdb


def cluster_trajectory(
    topology: str | Path,
    trajectory: str | Path,
    ligand_selection: Optional[str] = None,
    protein_selection: str = "protein",
    cutoff_angstrom: float = 1.5,
    start_ns: float = 0.0,
    stop_ns: Optional[float] = None,
    stride: int = 5,
    output_dir: Optional[str | Path] = None,
    target_residue: Optional[str] = None,
    reactive_nac_min: float = 0.10,
    reactive_distance_max: float = 3.8,
    electrophile_atom: Optional[str] = None,
) -> TrajectoryClusterReport:
    """Performs GROMOS RMSD clustering on a trajectory and exports representative and reactive medoids.

    Parameters
    ----------
    topology : str | Path
        Path to GRO or supported topology file.
    trajectory : str | Path
        Path to XTC trajectory file.
    ligand_selection : str, optional
        Atom selection for ligand heavy atoms. If None, auto-detects non-standard residue.
    protein_selection : str
        Atom selection for protein alignment (default: "protein").
    cutoff_angstrom : float
        RMSD neighbor cutoff for Daura clustering (default: 1.5 A).
    start_ns : float
        Simulation time in ns to begin sampling (default: 0.0).
    stop_ns : float, optional
        Simulation time in ns to stop sampling.
    stride : int
        Frame step stride for sampling (default: 5).
    output_dir : str | Path, optional
        Directory where snapshot PDB files will be saved.
    target_residue : str, optional
        Target residue (e.g. 'THR309') to measure trajectory Near-Attack population (P_NAC).
    reactive_nac_min : float
        Minimum NAC score threshold for identifying reactive subensemble frames (default: 0.10).
    reactive_distance_max : float
        Maximum nucleophile-electrophile distance for identifying reactive subensemble frames (default: 3.8 A).
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align
        from MDAnalysis.coordinates.XTC import XTCReader
    except ImportError as exc:
        raise RuntimeError("Trajectory clustering requires MDAnalysis. Install with: pip install MDAnalysis") from exc

    top_path = Path(topology)
    traj_path = Path(trajectory)
    if not top_path.is_file():
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    class ReadOnlyXTCReader(XTCReader):
        def _load_offsets(self):
            self._read_offsets(store=False)
        def _read_offsets(self, store=False):
            return super()._read_offsets(store=False)

    universe = mda.Universe(str(top_path), str(traj_path), format=ReadOnlyXTCReader)

    # Determine ligand selection
    if ligand_selection is None:
        resname = _detect_ligand_resname(universe)
        lig_sel_str = f"resname {resname} and not name H*"
    else:
        lig_sel_str = ligand_selection

    ligand_atoms = universe.select_atoms(lig_sel_str)
    if len(ligand_atoms) == 0:
        raise ValueError(f"Ligand atom selection '{lig_sel_str}' matched 0 atoms.")

    protein_ca = universe.select_atoms(f"({protein_selection}) and name CA")
    if len(protein_ca) == 0:
        protein_ca = universe.select_atoms(protein_selection)
    if len(protein_ca) == 0:
        raise ValueError(f"Protein selection '{protein_selection}' matched 0 atoms.")

    # Target residue atoms for P_NAC assessment if requested
    target_atoms = None
    if target_residue:
        import re
        m = re.match(r"([A-Za-z]+)?(\d+)", target_residue)
        if m:
            rname, rnum = m.group(1), m.group(2)
            q = f"resid {rnum}"
            if rname:
                q += f" and resname {rname.upper()}"
            target_atoms = universe.select_atoms(q)
    else:
        # Automatically detect the nearest pocket nucleophile residue to the ligand
        try:
            from MDAnalysis.lib.distances import distance_array
            nucl_cand = universe.select_atoms(
                "protein and (resname CYS or resname SER or resname THR or resname LYS or resname HIS or resname TYR) and around 4.5 group lig",
                lig=ligand_atoms
            )
            if len(nucl_cand) > 0:
                d_cand = distance_array(nucl_cand.positions, ligand_atoms.positions)
                closest_idx = int(np.argmin(np.min(d_cand, axis=1)))
                best_res = nucl_cand[closest_idx].residue
                target_atoms = best_res.atoms
        except Exception:
            target_atoms = None

    warnings_list: List[str] = []

    # Identify fixed atom triplet for stable Near-Attack Conformation tracking across all frames
    nucl_atom_idx = None
    el_atom_idx = None
    adj_atom_idx = None

    if target_atoms is not None and len(target_atoms) > 0 and len(ligand_atoms) > 0:
        rname = target_atoms[0].residue.resname.upper()
        target_nucl_names = {
            "CYS": ["SG"],
            "SER": ["OG"],
            "THR": ["OG1"],
            "LYS": ["NZ"],
            "HIS": ["NE2", "ND1"],
            "TYR": ["OH"],
        }
        cand_names = target_nucl_names.get(rname, [])
        nucl_cand_atom = None
        for cn in cand_names:
            matches = target_atoms.select_atoms(f"name {cn}")
            if len(matches) > 0:
                nucl_cand_atom = matches[0]
                break
        if nucl_cand_atom is None:
            heavy_target = target_atoms.select_atoms("not name H*")
            if len(heavy_target) > 0:
                from MDAnalysis.lib.distances import distance_array
                d_init = distance_array(heavy_target.positions, ligand_atoms.positions)
                min_i = int(np.argmin(np.min(d_init, axis=1)))
                nucl_cand_atom = heavy_target[min_i]
            else:
                nucl_cand_atom = target_atoms[0]

        nucl_atom_idx = int(nucl_cand_atom.index)

        # Identify electrophile heavy atom in ligand
        # Priority 1: User-specified electrophile atom name (e.g. 'C7', 'C4')
        # Priority 2: Genuine electrophilic centers (C, S, P, B)
        # Priority 3: Fallback to non-oxygen heavy atoms (e.g. N)
        # Priority 4: Generic closest heavy atom fallback with warning
        from MDAnalysis.lib.distances import distance_array
        el_cand_atom = None
        if electrophile_atom:
            matches = ligand_atoms.select_atoms(f"name {electrophile_atom}")
            if len(matches) > 0:
                el_cand_atom = matches[0]
            else:
                warnings_list.append(f"Specified electrophile atom '{electrophile_atom}' not found in ligand; using chemically validated rule.")

        if el_cand_atom is None:
            cand_electrophiles = ligand_atoms.select_atoms(
                "(name C* or name S* or name P* or name B*) and not name H*"
            )
            if len(cand_electrophiles) > 0:
                d_cands = distance_array(nucl_cand_atom.position.reshape(1, 3), cand_electrophiles.positions)[0]
                el_cand_atom = cand_electrophiles[int(np.argmin(d_cands))]
            else:
                non_o_heavy = ligand_atoms.select_atoms("not name H* and not name O* and not name F* and not name Cl* and not name Br*")
                if len(non_o_heavy) > 0:
                    d_cands = distance_array(nucl_cand_atom.position.reshape(1, 3), non_o_heavy.positions)[0]
                    el_cand_atom = non_o_heavy[int(np.argmin(d_cands))]
                else:
                    heavy_all = ligand_atoms.select_atoms("not name H*")
                    d_cands = distance_array(nucl_cand_atom.position.reshape(1, 3), heavy_all.positions)[0]
                    el_cand_atom = heavy_all[int(np.argmin(d_cands))]
                    warnings_list.append("Electrophile selected from generic closest heavy atom fallback.")

        el_atom_idx = int(el_cand_atom.index)

        # Adjacent bonded heavy atom in ligand for approach angle
        adj_cand_atom = None
        if hasattr(el_cand_atom, "bonded_atoms") and len(el_cand_atom.bonded_atoms) > 0:
            bonded_heavy = [b for b in el_cand_atom.bonded_atoms if not b.name.startswith("H") and b.index != el_cand_atom.index]
            if bonded_heavy:
                adj_cand_atom = bonded_heavy[0]
        if adj_cand_atom is None:
            # Distance-based bonded neighbor fallback (covalent bond length between 0.9 and 1.9 A)
            lig_heavy = ligand_atoms.select_atoms("not name H*")
            other_lig = [a for a in lig_heavy if a.index != el_cand_atom.index]
            if other_lig:
                other_pos = np.array([a.position for a in other_lig])
                d_adj = distance_array(el_cand_atom.position.reshape(1, 3), other_pos)[0]
                cand_bonded = [(d_adj[k], other_lig[k]) for k in range(len(d_adj)) if 0.9 <= d_adj[k] <= 1.9]
                if cand_bonded:
                    cand_bonded.sort(key=lambda x: x[0])
                    adj_cand_atom = cand_bonded[0][1]

        if adj_cand_atom is not None:
            adj_atom_idx = int(adj_cand_atom.index)
        else:
            warnings_list.append("Approach angle reference atom could not be identified; dynamic P_NAC is None.")

    # Identify local reactive atom group (target residue side-chain + ligand) for local reactive RMSD clustering
    local_rxn_atoms = None
    if target_atoms is not None and len(target_atoms) > 0 and len(ligand_atoms) > 0:
        nucl_sidechain = target_atoms.select_atoms("not name H* and not (name N or name CA or name C or name O)")
        if len(nucl_sidechain) == 0:
            nucl_sidechain = target_atoms.select_atoms("not name H*")
        local_rxn_atoms = nucl_sidechain + ligand_atoms

    # Convert start/stop ns to ps
    start_ps = start_ns * 1000.0
    stop_ps = (stop_ns * 1000.0) if stop_ns is not None else float("inf")

    sampled_coords = []
    sampled_rxn_coords = []
    frame_indices = []
    frame_times = []
    frame_records: List[ReactiveFrameRecord] = []
    nac_frame_count = 0
    nac_scores: List[float] = []
    dist_scores: List[float] = []

    # Reference structure for alignment (first frame in interval)
    ref_ca_pos = None
    for ts in universe.trajectory:
        if ts.time < start_ps:
            continue
        if ts.time > stop_ps:
            break
        ref_ca_pos = protein_ca.positions.copy()
        break

    if ref_ca_pos is None:
        raise ValueError(f"No trajectory frames found in time interval [{start_ns}, {stop_ns}] ns.")

    sampled_slice = universe.trajectory[::stride]
    total_sampled = len(sampled_slice) if hasattr(sampled_slice, "__len__") else None

    with tqdm(total=total_sampled, desc="[CLUSTERING 1/3] Sampling & aligning frames", unit="frame") as pbar:
        for ts in sampled_slice:
            if ts.time < start_ps:
                pbar.update(1)
                continue
            if ts.time > stop_ps:
                break

            # Align protein CA to ref_ca_pos using rigid-body Kabsch transformation
            R, mobile_center, ref_center = rigid_body_superposition(protein_ca.positions, ref_ca_pos)
            aligned_lig = apply_rigid_body_transformation(ligand_atoms.positions, mobile_center, R, ref_center)

            sampled_coords.append(aligned_lig)
            if local_rxn_atoms is not None:
                aligned_rxn = apply_rigid_body_transformation(local_rxn_atoms.positions, mobile_center, R, ref_center)
                sampled_rxn_coords.append(aligned_rxn)
            else:
                sampled_rxn_coords.append(aligned_lig)

            frame_indices.append(ts.frame)
            frame_times.append(ts.time)

            # Evaluate Near-Attack Conformation on fixed atom identities
            if nucl_atom_idx is not None and el_atom_idx is not None:
                nucl_pos = universe.atoms[nucl_atom_idx].position
                el_pos = universe.atoms[el_atom_idx].position
                adj_pos = universe.atoms[adj_atom_idx].position if adj_atom_idx is not None else None

                frame_score, f_d, f_theta, theta_deg = compute_frame_nac_score(
                    nucl_coord=nucl_pos,
                    el_coord=el_pos,
                    adj_coord=adj_pos,
                    d0=3.5,
                    sigma_d=0.5,
                    theta0=107.0,
                    sigma_theta=14.0,
                )
                dist_scores.append(f_d)
                d_val = float(np.linalg.norm(nucl_pos - el_pos))
                if frame_score is not None:
                    nac_scores.append(frame_score)
                    if d_val <= 3.5 and theta_deg is not None and (90.0 <= theta_deg <= 135.0):
                        nac_frame_count += 1

                rec = ReactiveFrameRecord(
                    trajectory_frame=ts.frame,
                    time_ps=ts.time,
                    distance_angstrom=d_val,
                    attack_angle_deg=theta_deg,
                    distance_score=f_d,
                    angle_score=f_theta,
                    nac_score=frame_score,
                )
            else:
                rec = ReactiveFrameRecord(
                    trajectory_frame=ts.frame,
                    time_ps=ts.time,
                    distance_angstrom=float("inf"),
                    attack_angle_deg=None,
                    distance_score=0.0,
                    angle_score=None,
                    nac_score=None,
                )
            frame_records.append(rec)
            pbar.update(1)

    n_samples = len(sampled_coords)
    if n_samples == 0:
        raise ValueError("Zero frames sampled for clustering.")

    p_nac = None
    dist_proximity = None
    if dist_scores:
        dist_proximity = round(float(np.mean(dist_scores)), 4)
    if nac_scores:
        p_nac = round(float(np.mean(nac_scores)), 4)
    elif nucl_atom_idx is not None:
        p_nac = None
        warnings_list.append("Dynamic P_NAC unavailable: missing reference atom for attack angle.")

    # Compute pairwise RMSD matrix for sampled ligand configurations
    sampled_coords = np.array(sampled_coords)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples), desc="[CLUSTERING 2/3] Computing RMSD matrix", unit="frame", leave=False):
        for j in range(i + 1, n_samples):
            diff = sampled_coords[i] - sampled_coords[j]
            r = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            dist_matrix[i, j] = r
            dist_matrix[j, i] = r

    # Daura et al. (GROMOS) clustering algorithm for dominant conformational ensemble
    remaining = set(range(n_samples))
    clusters = []

    with tqdm(total=n_samples, desc="[CLUSTERING 3/3] GROMOS frame partitioning", unit="frame", leave=False) as pbar:
        while remaining:
            best_center = None
            best_neighbors = []

            for candidate in remaining:
                neigh = [other for other in remaining if dist_matrix[candidate, other] <= cutoff_angstrom]
                if len(neigh) > len(best_neighbors):
                    best_neighbors = neigh
                    best_center = candidate

            if not best_neighbors:
                break

            clusters.append({
                "center": best_center,
                "members": best_neighbors,
                "size": len(best_neighbors)
            })

            for member in best_neighbors:
                remaining.remove(member)
            pbar.update(len(best_neighbors))

    if not clusters:
        raise RuntimeError("Clustering algorithm failed to partition frames.")

    # Dominant conformational cluster: record GROMOS center and compute true mathematical medoid
    top_cluster = clusters[0]
    gromos_center_idx = top_cluster["center"]
    gromos_center_frame = frame_indices[gromos_center_idx]
    gromos_center_time_ps = frame_times[gromos_center_idx]
    gromos_center_time_ns = round(gromos_center_time_ps / 1000.0, 3)

    medoid_idx = compute_true_medoid(top_cluster["members"], dist_matrix)
    medoid_frame = frame_indices[medoid_idx]
    medoid_time_ps = frame_times[medoid_idx]
    medoid_time_ns = round(medoid_time_ps / 1000.0, 3)
    top_size = top_cluster["size"]
    top_frac = round(top_size / n_samples, 4)

    for c_id, c in enumerate(clusters, 1):
        for m in c["members"]:
            frame_records[m].conformational_cluster_id = c_id

    # Export dominant conformational representative snapshots
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        import tempfile
        scratch_base = Path(os.environ.get("SHARK_SCRATCH", tempfile.gettempdir()))
        out_dir = scratch_base / "shark_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    complex_pdb, receptor_pdb, ligand_pdb = _export_snapshot(
        universe=universe,
        frame_index=medoid_frame,
        out_dir=out_dir,
        prefix="representative_snapshot",
        protein_selection=protein_selection,
        lig_sel_str=lig_sel_str,
    )

    # Identify and cluster the reactive subensemble
    reactive_indices = select_reactive_frames(
        records=frame_records,
        nac_min=reactive_nac_min,
        distance_max=reactive_distance_max,
    )

    rxn_complex_pdb = None
    rxn_receptor_pdb = None
    rxn_ligand_pdb = None
    rxn_medoid_frame = None
    rxn_medoid_time_ps = None
    rxn_medoid_time_ns = None
    rxn_medoid_nac = None
    rxn_medoid_d = None
    rxn_medoid_theta = None
    rxn_num_clusters = 0
    top_rxn_size = 0
    top_rxn_frac = None
    rxn_frame_fraction = None

    if reactive_indices:
        reactive_frame_count = len(reactive_indices)
        rxn_frame_fraction = round(reactive_frame_count / n_samples, 4)
        m_rxn = len(reactive_indices)

        if m_rxn == 1:
            rxn_medoid_orig_idx = reactive_indices[0]
            frame_records[rxn_medoid_orig_idx].reactive_cluster_id = 1
            rxn_num_clusters = 1
            top_rxn_size = 1
            top_rxn_frac = round(1.0 / n_samples, 4)
        else:
            rxn_coords = [sampled_rxn_coords[i] for i in reactive_indices]
            rxn_dist_matrix = np.zeros((m_rxn, m_rxn))
            for i in range(m_rxn):
                for j in range(i + 1, m_rxn):
                    diff = rxn_coords[i] - rxn_coords[j]
                    r = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
                    rxn_dist_matrix[i, j] = r
                    rxn_dist_matrix[j, i] = r

            remaining_rxn = set(range(m_rxn))
            rxn_clusters = []
            while remaining_rxn:
                best_center = None
                best_neighbors = []
                for cand in remaining_rxn:
                    neigh = [o for o in remaining_rxn if rxn_dist_matrix[cand, o] <= cutoff_angstrom]
                    if len(neigh) > len(best_neighbors):
                        best_neighbors = neigh
                        best_center = cand
                if not best_neighbors:
                    break
                rxn_clusters.append({
                    "center": best_center,
                    "members": best_neighbors,
                    "size": len(best_neighbors)
                })
                for m in best_neighbors:
                    remaining_rxn.remove(m)

            if not rxn_clusters:
                rxn_clusters = [{"center": 0, "members": list(range(m_rxn)), "size": m_rxn}]

            # Rank reactive clusters: primary by population size, secondary by mean NAC score
            def _rxn_cluster_sort_key(c_item):
                members = c_item["members"]
                scores = [frame_records[reactive_indices[m]].nac_score or 0.0 for m in members]
                mean_score = float(np.mean(scores)) if scores else 0.0
                return (c_item["size"], mean_score)

            rxn_clusters.sort(key=_rxn_cluster_sort_key, reverse=True)

            for rc_id, rc in enumerate(rxn_clusters, 1):
                for local_m in rc["members"]:
                    orig_i = reactive_indices[local_m]
                    frame_records[orig_i].reactive_cluster_id = rc_id

            top_rxn_cluster = rxn_clusters[0]
            rxn_num_clusters = len(rxn_clusters)
            top_rxn_size = top_rxn_cluster["size"]
            top_rxn_frac = round(top_rxn_size / n_samples, 4)

            best_local_medoid = compute_true_medoid(top_rxn_cluster["members"], rxn_dist_matrix)
            rxn_medoid_orig_idx = reactive_indices[best_local_medoid]

        rxn_med_rec = frame_records[rxn_medoid_orig_idx]
        rxn_medoid_frame = rxn_med_rec.trajectory_frame
        rxn_medoid_time_ps = rxn_med_rec.time_ps
        rxn_medoid_time_ns = round(rxn_medoid_time_ps / 1000.0, 3)
        rxn_medoid_nac = rxn_med_rec.nac_score
        rxn_medoid_d = rxn_med_rec.distance_angstrom
        rxn_medoid_theta = rxn_med_rec.attack_angle_deg

        rxn_complex_pdb, rxn_receptor_pdb, rxn_ligand_pdb = _export_snapshot(
            universe=universe,
            frame_index=rxn_medoid_frame,
            out_dir=out_dir,
            prefix="reactive_snapshot",
            protein_selection=protein_selection,
            lig_sel_str=lig_sel_str,
        )
    else:
        reactive_frame_count = 0
        rxn_frame_fraction = 0.0
        warnings_list.append("No populated reactive subensemble was identified under the current reaction-geometry model.")

    if p_nac is not None:
        p_nac_info = f"Trajectory continuous P_NAC = {p_nac*100:.1f}%."
    elif dist_proximity is not None:
        p_nac_info = f"Trajectory proximity score = {dist_proximity:.3f} (distance only, angle missing)."
    else:
        p_nac_info = "Trajectory P_NAC = Not evaluated."

    rxn_info = ""
    if rxn_medoid_frame is not None:
        rxn_info = (
            f"Reactive representative medoid extracted at frame #{rxn_medoid_frame} ({rxn_medoid_time_ns:.2f} ns, "
            f"d={rxn_medoid_d:.2f} A, theta={rxn_medoid_theta:.1f} deg, NAC={rxn_medoid_nac:.2f}, "
            f"top reactive cluster size={top_rxn_size}/{n_samples} [{top_rxn_frac*100:.1f}%])."
        )
    else:
        rxn_info = "No reactive subensemble frames identified (reactive representative = None)."

    summary = (
        f"GROMOS clustering over {n_samples} frames ({frame_times[0]/1000.0:.1f}–{frame_times[-1]/1000.0:.1f} ns): "
        f"Top cluster encompasses {top_size}/{n_samples} frames ({top_frac*100:.1f}% population, cutoff {cutoff_angstrom} A). "
        f"Dominant conformational medoid at frame #{medoid_frame} ({medoid_time_ns:.2f} ns). "
        f"{p_nac_info} "
        f"{rxn_info}"
    )

    total_sim_time = round(frame_times[-1] / 1000.0, 2) if frame_times else 0.0

    return TrajectoryClusterReport(
        total_sampled_frames=n_samples,
        num_clusters=len(clusters),
        top_cluster_size=top_size,
        top_cluster_fraction=top_frac,
        gromos_center_frame_index=gromos_center_frame,
        gromos_center_time_ps=gromos_center_time_ps,
        gromos_center_time_ns=gromos_center_time_ns,
        medoid_frame_index=medoid_frame,
        medoid_time_ps=medoid_time_ps,
        medoid_time_ns=medoid_time_ns,
        cutoff_angstrom=cutoff_angstrom,
        sim_time_ns=total_sim_time,
        total_sim_time_ns=total_sim_time,
        snapshot_complex_pdb=complex_pdb,
        snapshot_receptor_pdb=receptor_pdb,
        snapshot_ligand_pdb=ligand_pdb,
        cluster_sizes=[c["size"] for c in clusters],
        p_nac=round(p_nac, 4) if p_nac is not None else None,
        distance_proximity_score=round(dist_proximity, 4) if dist_proximity is not None else None,
        reactive_frame_count=reactive_frame_count,
        reactive_frame_fraction=rxn_frame_fraction,
        num_reactive_clusters=rxn_num_clusters,
        top_reactive_cluster_size=top_rxn_size,
        top_reactive_cluster_fraction=top_rxn_frac,
        reactive_medoid_frame_index=rxn_medoid_frame,
        reactive_medoid_time_ps=rxn_medoid_time_ps,
        reactive_medoid_time_ns=rxn_medoid_time_ns,
        reactive_medoid_nac_score=round(rxn_medoid_nac, 4) if rxn_medoid_nac is not None else None,
        reactive_medoid_distance_angstrom=round(rxn_medoid_d, 3) if rxn_medoid_d is not None else None,
        reactive_medoid_angle_deg=round(rxn_medoid_theta, 2) if rxn_medoid_theta is not None else None,
        reactive_snapshot_complex_pdb=rxn_complex_pdb,
        reactive_snapshot_receptor_pdb=rxn_receptor_pdb,
        reactive_snapshot_ligand_pdb=rxn_ligand_pdb,
        frame_records=frame_records,
        warnings=warnings_list,
        summary=summary,
    )
