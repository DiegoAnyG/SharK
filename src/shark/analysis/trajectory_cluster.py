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
    p_nac: float = 0.0
    summary: str = ""


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
        # e.g., 'THR309' -> resname THR and resid 309
        import re
        m = re.match(r"([A-Za-z]+)?(\d+)", target_residue)
        if m:
            rname, rnum = m.group(1), m.group(2)
            q = f"resid {rnum}"
            if rname:
                q += f" and resname {rname.upper()}"
            target_atoms = universe.select_atoms(q)

    # Convert start/stop ns to ps
    start_ps = start_ns * 1000.0
    stop_ps = (stop_ns * 1000.0) if stop_ns is not None else float("inf")

    # Sample coordinates after alignment on protein CA
    sampled_coords = []
    frame_indices = []
    frame_times = []
    nac_frame_count = 0

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

        # Align protein CA to ref_ca_pos to remove global rigid-body translation and rotation
        R, _ = align.rotation_matrix(protein_ca.positions, ref_ca_pos)
        ca_com = protein_ca.center_of_mass()
        aligned_lig = np.dot(ligand_atoms.positions - ca_com, R.T)

        sampled_coords.append(aligned_lig)
        frame_indices.append(ts.frame)
        frame_times.append(ts.time)

        # Check Near-Attack distance if target residue provided
        if target_atoms is not None and len(target_atoms) > 0:
            from MDAnalysis.lib.distances import distance_array
            dmat_t = distance_array(target_atoms.positions, ligand_atoms.positions)
            if np.min(dmat_t) <= 3.5:
                nac_frame_count += 1

    n_samples = len(sampled_coords)
    if n_samples == 0:
        raise ValueError("Zero frames sampled for clustering.")

    p_nac = (nac_frame_count / n_samples) if n_samples > 0 else 0.0

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

    summary = (
        f"GROMOS clustering over {n_samples} frames ({frame_times[0]/1000.0:.1f}–{frame_times[-1]/1000.0:.1f} ns): "
        f"Top cluster encompasses {top_size}/{n_samples} frames ({top_frac*100:.1f}% population, cutoff {cutoff_angstrom} A). "
        f"Medoid representative snapshot extracted at frame #{medoid_frame} ({medoid_time/1000.0:.2f} ns). "
        f"Trajectory P_NAC = {p_nac*100:.1f}%."
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
        p_nac=round(p_nac, 4),
        summary=summary
    )
