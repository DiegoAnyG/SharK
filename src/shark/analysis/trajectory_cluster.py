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

STANDARD_RESIDUES_SET = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "CYX", "HID", "HIE", "HIP", "ASH", "GLH", "SOL", "WAT", "HOH", "TIP3",
    "NA", "CL", "K", "MG", "CA", "ZN", "GDP", "GTP", "ADP", "ATP"
}


@dataclass
class TrajectoryClusterReport:
    """Statistical summary of trajectory clustering and representative snapshot extraction."""
    total_sampled_frames: int
    num_clusters: int
    top_cluster_size: int
    top_cluster_fraction: float
    medoid_frame_index: int
    medoid_time_ps: float
    medoid_time_ns: float
    cutoff_angstrom: float
    snapshot_complex_pdb: Optional[Path] = None
    snapshot_receptor_pdb: Optional[Path] = None
    snapshot_ligand_pdb: Optional[Path] = None
    cluster_sizes: List[int] = field(default_factory=list)
    p_nac: Optional[float] = None
    distance_proximity_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    summary: str = ""


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
    target_residue: Optional[str] = None
) -> TrajectoryClusterReport:
    """Performs GROMOS RMSD clustering on a trajectory and exports the medoid snapshot.

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

        # Electrophile heavy atom in ligand closest to nucleophile in initial frame
        lig_heavy = ligand_atoms.select_atoms("not name H*")
        if len(lig_heavy) == 0:
            lig_heavy = ligand_atoms
        from MDAnalysis.lib.distances import distance_array
        d_el = distance_array(nucl_cand_atom.position.reshape(1, 3), lig_heavy.positions)[0]
        el_cand_atom = lig_heavy[int(np.argmin(d_el))]
        el_atom_idx = int(el_cand_atom.index)

        # Adjacent bonded heavy atom in ligand for approach angle
        adj_cand_atom = None
        if hasattr(el_cand_atom, "bonded_atoms") and len(el_cand_atom.bonded_atoms) > 0:
            bonded_heavy = [b for b in el_cand_atom.bonded_atoms if not b.name.startswith("H") and b.index != el_cand_atom.index]
            if bonded_heavy:
                adj_cand_atom = bonded_heavy[0]
        if adj_cand_atom is None:
            # Distance-based bonded neighbor fallback (covalent bond length between 0.9 and 1.9 A)
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

    # Convert start/stop ns to ps
    start_ps = start_ns * 1000.0
    stop_ps = (stop_ns * 1000.0) if stop_ns is not None else float("inf")

    sampled_coords = []
    frame_indices = []
    frame_times = []
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

    for ts in universe.trajectory[::stride]:
        if ts.time < start_ps:
            continue
        if ts.time > stop_ps:
            break

        # Align protein CA to ref_ca_pos using rigid-body Kabsch transformation
        R, mobile_center, ref_center = rigid_body_superposition(protein_ca.positions, ref_ca_pos)
        aligned_lig = apply_rigid_body_transformation(ligand_atoms.positions, mobile_center, R, ref_center)

        sampled_coords.append(aligned_lig)
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
            if frame_score is not None:
                nac_scores.append(frame_score)
                d_val = float(np.linalg.norm(nucl_pos - el_pos))
                if d_val <= 3.5 and theta_deg is not None and (90.0 <= theta_deg <= 135.0):
                    nac_frame_count += 1

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
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = sampled_coords[i] - sampled_coords[j]
            r = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            dist_matrix[i, j] = r
            dist_matrix[j, i] = r

    # Daura et al. (GROMOS) clustering algorithm
    remaining = set(range(n_samples))
    clusters = []

    while remaining:
        # For each remaining frame, count neighbors within cutoff
        best_center = None
        best_neighbors = []

        for candidate in remaining:
            # Neighbors include itself and all remaining frames within cutoff
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

        # Remove clustered members from pool
        for member in best_neighbors:
            remaining.remove(member)

    if not clusters:
        raise RuntimeError("Clustering algorithm failed to partition frames.")

    # Top cluster represents the dominant conformational basin
    top_cluster = clusters[0]
    medoid_idx = top_cluster["center"]
    medoid_frame = frame_indices[medoid_idx]
    medoid_time = frame_times[medoid_idx]
    top_size = top_cluster["size"]
    top_frac = round(top_size / n_samples, 4)

    # Position trajectory at the medoid frame to export the snapshot
    universe.trajectory[medoid_frame]

    # Export representative snapshots
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        import tempfile
        scratch_base = Path(os.environ.get("SHARK_SCRATCH", tempfile.gettempdir()))
        out_dir = scratch_base / "shark_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    complex_pdb = out_dir / "representative_snapshot_complex.pdb"
    receptor_pdb = out_dir / "representative_snapshot_receptor.pdb"
    ligand_pdb = out_dir / "representative_snapshot_ligand.pdb"

    # Select components (excluding water/ions for the clean docking/covalent analysis)
    universe.select_atoms(f"({protein_selection}) or resname GDP or resname GTP or {lig_sel_str}").write(str(complex_pdb))
    universe.select_atoms(f"({protein_selection}) or resname GDP or resname GTP").write(str(receptor_pdb))
    universe.select_atoms(lig_sel_str).write(str(ligand_pdb))

    if p_nac is not None:
        p_nac_info = f"Trajectory P_NAC = {p_nac*100:.1f}% (distance-plus-angle)."
    elif dist_proximity is not None:
        p_nac_info = f"Trajectory P_NAC = Not available | Proximity score = {dist_proximity:.3f} (distance only, angle missing)."
    else:
        p_nac_info = "Trajectory P_NAC = Not evaluated."

    summary = (
        f"GROMOS clustering over {n_samples} frames ({frame_times[0]/1000.0:.1f}–{frame_times[-1]/1000.0:.1f} ns): "
        f"Top cluster encompasses {top_size}/{n_samples} frames ({top_frac*100:.1f}% population, cutoff {cutoff_angstrom} A). "
        f"Medoid representative snapshot extracted at frame #{medoid_frame} ({medoid_time/1000.0:.2f} ns). "
        f"{p_nac_info}"
    )

    return TrajectoryClusterReport(
        total_sampled_frames=n_samples,
        num_clusters=len(clusters),
        top_cluster_size=top_size,
        top_cluster_fraction=top_frac,
        medoid_frame_index=medoid_frame,
        medoid_time_ps=medoid_time,
        medoid_time_ns=round(medoid_time / 1000.0, 3),
        cutoff_angstrom=cutoff_angstrom,
        snapshot_complex_pdb=complex_pdb,
        snapshot_receptor_pdb=receptor_pdb,
        snapshot_ligand_pdb=ligand_pdb,
        cluster_sizes=[c["size"] for c in clusters],
        p_nac=round(p_nac, 4) if p_nac is not None else None,
        distance_proximity_score=round(dist_proximity, 4) if dist_proximity is not None else None,
        warnings=warnings_list,
        summary=summary
    )
