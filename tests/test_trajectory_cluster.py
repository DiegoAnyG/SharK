"""Unit tests for Daura trajectory clustering and Simple Gold Standard workflow."""

import json
from pathlib import Path
import zipfile
import pytest

from shark.analysis.trajectory_cluster import cluster_trajectory, TrajectoryClusterReport
from shark.cli import main as cli_main


def test_daura_clustering_synthetic_trajectory(tmp_path):
    mda = pytest.importorskip("MDAnalysis")

    # Create synthetic universe: 3 protein atoms (ALA) + 2 ligand atoms (UNL)
    u = mda.Universe.empty(5, n_residues=2, atom_resindex=[0, 0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["N", "CA", "C", "C1", "C2"])
    u.add_TopologyAttr("resnames", ["ALA", "UNL"])
    u.add_TopologyAttr("resids", [1, 99])
    u.add_TopologyAttr("ids", [1, 2, 3, 4, 5])
    u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]

    protein_coords = np_coords = [
        [10.0, 10.0, 10.0],
        [11.0, 10.0, 10.0],
        [12.0, 10.0, 10.0]
    ]
    lig_pos_a = [[15.0, 10.0, 10.0], [16.0, 10.0, 10.0]]
    lig_pos_b = [[22.0, 10.0, 10.0], [23.0, 10.0, 10.0]]

    topology_file = tmp_path / "system.gro"
    trajectory_file = tmp_path / "trajectory.xtc"

    u.atoms.positions = protein_coords + lig_pos_a
    u.atoms.write(str(topology_file))

    # Write 4 frames: 3 at pos_a (cluster 1, 75%), 1 at pos_b (cluster 2, 25%)
    with mda.Writer(str(trajectory_file), n_atoms=5) as writer:
        for t_ps, lig_pos in [(0.0, lig_pos_a), (10.0, lig_pos_a), (20.0, lig_pos_a), (30.0, lig_pos_b)]:
            u.trajectory.ts.time = t_ps
            u.atoms.positions = protein_coords + lig_pos
            writer.write(u)

    snapshot_dir = tmp_path / "snapshots"
    rep = cluster_trajectory(
        topology=topology_file,
        trajectory=trajectory_file,
        ligand_selection="resname UNL",
        protein_selection="protein",
        cutoff_angstrom=1.5,
        stride=1,
        output_dir=snapshot_dir
    )

    assert isinstance(rep, TrajectoryClusterReport)
    assert rep.total_sampled_frames == 4
    assert rep.num_clusters == 2
    assert rep.top_cluster_size == 3
    assert rep.top_cluster_fraction == 0.75
    assert rep.medoid_time_ps in (0.0, 10.0, 20.0)
    assert rep.snapshot_complex_pdb.is_file()
    assert rep.snapshot_receptor_pdb.is_file()
    assert rep.snapshot_ligand_pdb.is_file()

    # Verify PDB content
    rec_content = rep.snapshot_receptor_pdb.read_text(encoding="utf-8")
    assert "ALA" in rec_content
    assert "UNL" not in rec_content

    lig_content = rep.snapshot_ligand_pdb.read_text(encoding="utf-8")
    assert "UNL" in lig_content
    assert "ALA" not in lig_content


def test_cli_simple_gold_standard_and_fast_analysis(tmp_path):
    mda = pytest.importorskip("MDAnalysis")

    # 1. Prepare minimal session archive
    archive = tmp_path / "test_session.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True, "project": "GoldTest"}))
        z.writestr("receptors/8HTB_ready.pdb", (
            "ATOM      1  N   THR A 309      10.000  10.000  10.000  1.00 20.00           N\n"
            "ATOM      2  OG1 THR A 309      10.000  12.000  10.000  1.00 20.00           O\n"
            "END\n"
        ))
        z.writestr("poses/docking_8HTB_ready~Pk1_compounds_a_BENZ-model1.pdb", (
            "ATOM      1  C1  BENZ A   1      10.000  14.500  10.000  1.00 20.00           C\n"
            "ATOM      2  C2  BENZ A   1      11.000  14.500  10.000  1.00 20.00           C\n"
            "END\n"
        ))
        z.writestr("docking_results.csv", (
            "receptor,pose_name,compound_name,docking_score,engine\n"
            "8HTB_ready~Pk1,docking_8HTB_ready~Pk1_compounds_a_BENZ-model1,BENZ,-8.5,vina\n"
        ))

    # 2. Test Fast Analysis CLI flag
    fast_html = tmp_path / "fast_dossier.html"
    code_fast = cli_main([
        "--session", str(archive),
        "--fast-analysis",
        "--target-residue", "THR309",
        "--html", str(fast_html)
    ])
    assert code_fast == 0
    assert fast_html.is_file()
    fast_text = fast_html.read_text(encoding="utf-8")
    assert "Covalent Near-Attack Conformations (NAC)" in fast_text
    assert "THR309:A" in fast_text

    # 3. Prepare small MD trajectory matching the system
    u = mda.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["N", "OG1", "C1", "C2"])
    u.add_TopologyAttr("resnames", ["THR", "BENZ"])
    u.add_TopologyAttr("resids", [309, 1])
    u.add_TopologyAttr("ids", [1, 2, 3, 4])
    u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]

    coords = [
        [10.0, 10.0, 10.0],
        [10.0, 12.0, 10.0],
        [10.0, 14.8, 10.0],
        [11.0, 14.8, 10.0],
    ]
    topology_file = tmp_path / "md_prod.gro"
    trajectory_file = tmp_path / "md_noPBC.xtc"

    u.atoms.positions = coords
    u.atoms.write(str(topology_file))

    with mda.Writer(str(trajectory_file), n_atoms=4) as writer:
        for t in [0.0, 50.0, 100.0]:
            u.trajectory.ts.time = t
            writer.write(u)

    # 4. Test Simple Gold Standard CLI flag
    gold_html = tmp_path / "simple_gold_dossier.html"
    code_gold = cli_main([
        "--session", str(archive),
        "--simple-gold-standard",
        "--topology", str(topology_file),
        "--trajectory", str(trajectory_file),
        "--target-residue", "THR309",
        "--cluster-stride", "1",
        "--html", str(gold_html)
    ])
    assert code_gold == 0
    assert gold_html.is_file()
    gold_text = gold_html.read_text(encoding="utf-8")
    assert "MD Representative Snapshot (GROMOS Medoid)" in gold_text
    assert "Top cluster population" in gold_text
    assert "THR309" in gold_text


def test_nac_1_identical_distance_different_angle():
    """Test NAC-1: Two frames with identical distance but different attack angle must produce different scores."""
    from shark.analysis.trajectory_cluster import compute_frame_nac_score
    import numpy as np

    # Electrophile at origin, adjacent atom along +x
    el = np.array([0.0, 0.0, 0.0])
    adj = np.array([1.4, 0.0, 0.0])

    # Case A: nucleophile at 107 degrees (ideal Bürgi-Dunitz), distance 3.0 A
    theta_a = np.radians(107.0)
    nucl_a = np.array([3.0 * np.cos(theta_a), 3.0 * np.sin(theta_a), 0.0])

    # Case B: nucleophile at 160 degrees (poor angle), same distance 3.0 A
    theta_b = np.radians(160.0)
    nucl_b = np.array([3.0 * np.cos(theta_b), 3.0 * np.sin(theta_b), 0.0])

    score_a, fd_a, fth_a, ang_a = compute_frame_nac_score(nucl_a, el, adj)
    score_b, fd_b, fth_b, ang_b = compute_frame_nac_score(nucl_b, el, adj)

    assert pytest.approx(fd_a, rel=1e-3) == fd_b  # Distances are identical
    assert pytest.approx(ang_a, abs=0.5) == 107.0
    assert pytest.approx(ang_b, abs=0.5) == 160.0
    assert score_a > 0.5
    assert score_b < 0.1
    assert score_a > score_b * 5.0  # Favorable angle produces significantly higher score


def test_nac_2_favorable_angle_poor_distance():
    """Test NAC-2: Favorable angle with poor distance must remain low."""
    from shark.analysis.trajectory_cluster import compute_frame_nac_score
    import numpy as np

    el = np.array([0.0, 0.0, 0.0])
    adj = np.array([1.4, 0.0, 0.0])

    # Favorable angle (107 deg) but very far (6.0 A)
    theta = np.radians(107.0)
    nucl = np.array([6.0 * np.cos(theta), 6.0 * np.sin(theta), 0.0])

    score, fd, fth, ang = compute_frame_nac_score(nucl, el, adj)
    assert fth > 0.95  # Perfect angle
    assert fd < 0.01   # Crushed by distance
    assert score < 0.01


def test_nac_3_favorable_distance_poor_angle():
    """Test NAC-3: Favorable distance with poor angle must remain low."""
    from shark.analysis.trajectory_cluster import compute_frame_nac_score
    import numpy as np

    el = np.array([0.0, 0.0, 0.0])
    adj = np.array([1.4, 0.0, 0.0])

    # Close contact (2.5 A) but nearly collinear with back-side (30 deg)
    theta = np.radians(30.0)
    nucl = np.array([2.5 * np.cos(theta), 2.5 * np.sin(theta), 0.0])

    score, fd, fth, ang = compute_frame_nac_score(nucl, el, adj)
    assert fd > 0.85   # Good distance
    assert fth < 1e-4  # Crushed by angular penalty
    assert score < 1e-4


def test_nac_5_missing_angle_returns_none():
    """Test NAC-5: If angle data are unavailable, p_nac is None."""
    from shark.analysis.trajectory_cluster import compute_frame_nac_score
    import numpy as np

    nucl = np.array([0.0, 2.5, 0.0])
    el = np.array([0.0, 0.0, 0.0])

    score, fd, fth, ang = compute_frame_nac_score(nucl, el, adj_coord=None)
    assert score is None
    assert fth is None
    assert ang is None
    assert fd > 0.8


def test_rigid_body_superposition_synthetic():
    """Priority 13: Synthetic test for centered Kabsch alignment recovering known rotation/translation."""
    from shark.analysis.trajectory_cluster import rigid_body_superposition, apply_rigid_body_transformation
    import numpy as np

    # Reference coordinates: 4 non-coplanar points
    ref_coords = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 2.0],
        [2.0, 5.0, 1.0],
        [3.0, 2.0, 6.0],
    ])

    # Apply known 3D rotation (around z-axis by 45 deg) and arbitrary translation
    theta = np.radians(45.0)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    translation = np.array([12.5, -8.3, 44.1])

    # mobile = ref @ rot_z.T + translation
    mobile_coords = np.dot(ref_coords, rot_z.T) + translation

    # Compute optimal superposition of mobile onto ref
    R, mobile_center, ref_center = rigid_body_superposition(mobile_coords, ref_coords)
    aligned = apply_rigid_body_transformation(mobile_coords, mobile_center, R, ref_center)

    rmsd = np.sqrt(np.mean((aligned - ref_coords) ** 2))
    assert rmsd < 1e-6, f"Superposition failed to recover reference structure: RMSD = {rmsd}"


