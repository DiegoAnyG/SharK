"""Read-only periodic trajectory contacts; proximity is not chemical reactivity."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from ..core.frontier import file_hash


def analyze_contacts(topology, trajectory, *, ligand_selection, protein_selection='protein and not name H*',
                     cutoff_angstrom=4.0, stride=1, start_ns=0.0, stop_ns=None):
    """Analyze explicitly selected atoms in an XTC without writing trajectory caches.

    Selection expressions, sampled frames and input hashes are recorded. GRO does
    not preserve original PDB chains/insertion codes; missing identities stay null.
    Frame contact fractions are descriptive and are not independent-sample probabilities.
    """
    if not math.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0:
        raise ValueError('Contact cutoff must be finite and positive')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('Frame stride must be a positive integer')
    if not math.isfinite(start_ns) or start_ns < 0 or (stop_ns is not None and
            (not math.isfinite(stop_ns) or stop_ns < start_ns)):
        raise ValueError('Invalid analysis time interval')
    try:
        import MDAnalysis as mda
        from MDAnalysis.coordinates.XTC import XTCReader
        from MDAnalysis.lib.distances import distance_array
    except ImportError as exc:
        raise RuntimeError('Trajectory analysis requires the optional MDAnalysis dependency: install shark-qc[md]') from exc
    topology, trajectory = Path(topology), Path(trajectory)
    if trajectory.suffix.lower() != '.xtc':
        raise ValueError('This contact importer currently supports XTC trajectories')

    class ReadOnlyXTCReader(XTCReader):
        # XDR readers otherwise write offsets and lock files beside the source.
        def _load_offsets(self):
            self._read_offsets(store=False)

        def _read_offsets(self, store=False):
            return super()._read_offsets(store=False)

    try:
        universe = mda.Universe(str(topology), str(trajectory), format=ReadOnlyXTCReader)
    except (ValueError, OSError) as exc:
        raise ValueError('Cannot read topology/trajectory; check atom counts and format support. '
                         'For unsupported TPR versions, supply the matching GRO explicitly; '
                         'original chain identity will then be unavailable.') from exc
    try:
        try:
            ligand = universe.select_atoms(ligand_selection)
            protein = universe.select_atoms(protein_selection)
        except mda.exceptions.SelectionError as exc:
            raise ValueError('Invalid MDAnalysis atom selection') from exc
        if not len(ligand) or not len(protein):
            raise ValueError('Both atom selections must be nonempty')
        if len(ligand.residues) != 1:
            raise ValueError('Ligand selection must identify exactly one residue')
        if np.intersect1d(ligand.indices, protein.indices).size:
            raise ValueError('Ligand and partner selections must not overlap')
        residue_indices, inverse = np.unique(protein.resindices, return_inverse=True)
        counts = np.zeros(len(residue_indices), dtype=int)
        distance_sum = np.zeros(len(residue_indices))
        pair_min = np.full((len(protein), len(ligand)), np.inf)
        pair_counts = np.zeros(pair_min.shape, dtype=int)
        pair_frame = np.full(pair_min.shape, -1, dtype=int)
        pair_time = np.full(pair_min.shape, np.nan)
        times, frames = [], []
        for ts in universe.trajectory[::stride]:
            time_ns = float(ts.time) / 1000
            if time_ns < start_ns or (stop_ns is not None and time_ns > stop_ns):
                continue
            if ts.dimensions is None or not np.all(np.isfinite(ts.dimensions)) or np.any(ts.dimensions[:3] <= 0) or np.any(ts.dimensions[3:] <= 0) or np.any(ts.dimensions[3:] >= 180):
                raise ValueError('Periodic distances require a valid box in every sampled frame')
            dist = distance_array(protein.positions, ligand.positions, box=ts.dimensions)
            if not np.all(np.isfinite(dist)):
                raise ValueError('Nonfinite trajectory distances')
            per_residue = np.full(len(residue_indices), np.inf)
            np.minimum.at(per_residue, inverse, dist.min(axis=1))
            counts += per_residue <= cutoff_angstrom
            distance_sum += per_residue
            pair_counts += dist <= cutoff_angstrom
            lower = dist < pair_min
            pair_min[lower] = dist[lower]
            pair_frame[lower] = ts.frame
            pair_time[lower] = time_ns
            times.append(time_ns); frames.append(int(ts.frame))
        if not times:
            raise ValueError('No frames in the requested time interval')
        if any(b <= a for a,b in zip(times,times[1:])):
            raise ValueError('Sampled trajectory times must increase strictly')

        def identity(atom):
            return dict(index_zero_based=int(atom.index), atom_id=int(atom.id), name=str(atom.name),
                        resindex_zero_based=int(atom.resindex), resid=int(atom.resid), resname=str(atom.resname),
                        segid=str(atom.segid), chain_id=getattr(atom, 'chainID', None),
                        insertion_code=getattr(atom, 'icode', None))

        contacts = []
        for i, ri in enumerate(residue_indices):
            if not counts[i]:
                continue
            p_indices = np.flatnonzero(inverse == i)
            local_p, lig_idx = np.unravel_index(np.argmin(pair_min[p_indices]), (len(p_indices), len(ligand)))
            prot_idx = p_indices[local_p]
            pa, la = identity(protein[prot_idx]), identity(ligand[lig_idx])
            contacts.append(dict(residue=f"{pa['segid']}:{pa['resname']}{pa['resid']} [resindex {ri}]",
                occupancy_fraction=float(counts[i]/len(times)), contact_frames=int(counts[i]),
                minimum_distance_angstrom=float(pair_min[prot_idx,lig_idx]),
                mean_minimum_distance_angstrom=float(distance_sum[i]/len(times)),
                closest_protein_atom=pa['name'], closest_ligand_atom=la['name'],
                closest_pair_contact_fraction=float(pair_counts[prot_idx,lig_idx]/len(times)),
                protein_atom=pa, ligand_atom=la,
                minimum_frame=int(pair_frame[prot_idx,lig_idx]), minimum_time_ns=float(pair_time[prot_idx,lig_idx])))
        contacts.sort(key=lambda c:(-c['occupancy_fraction'], c['minimum_distance_angstrom'],c['residue']))
        pairs = [dict(protein_atom=identity(protein[i]), ligand_atom=identity(ligand[j]),
                      contact_fraction=float(pair_counts[i,j]/len(times)),
                      minimum_distance_angstrom=float(pair_min[i,j]),
                      minimum_frame=int(pair_frame[i,j]),minimum_time_ns=float(pair_time[i,j]))
                 for i,j in zip(*np.nonzero(pair_counts))]
        pairs.sort(key=lambda p:(-p['contact_fraction'],p['minimum_distance_angstrom']))
        return dict(schema_version=1, analysis='selected_atom_periodic_contacts',
            interpretation='Geometric proximity only; not nucleophilicity, a covalent bond, affinity or reaction probability.',
            topology_identity_note='Identities are those of the supplied topology. GRO lacks original PDB chains and insertion codes; segment labels are not chain mappings.',
            sampling_note='All selected frames contribute equally; temporal correlation is not removed. Residue contacts may involve different atom pairs in different frames.',
            chemical_identity_note='No MD-to-QM bond-order, protonation or atom mapping is inferred by this contact analysis.',
            inputs={'topology':{'file':topology.name,'sha256':file_hash(topology)},'trajectory':{'file':trajectory.name,'sha256':file_hash(trajectory)}},
            software={'MDAnalysis':mda.__version__,'numpy':np.__version__},
            parameters=dict(ligand_selection=ligand_selection,protein_selection=protein_selection,
                            cutoff_angstrom=cutoff_angstrom,stride=stride,start_ns=start_ns,stop_ns=stop_ns),
            atom_count=len(universe.atoms), ligand_atoms=[identity(atom) for atom in ligand],
            trajectory_frame_count=len(universe.trajectory), sampled_frame_count=len(times),
            sampled_frames=frames, sampled_times_ns=times, time_start_ns=times[0],time_end_ns=times[-1],
            sim_time_ns=times[-1]-times[0],contacts=contacts,atom_pair_contacts=pairs)
    finally:
        universe.trajectory.close()
