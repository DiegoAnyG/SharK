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

