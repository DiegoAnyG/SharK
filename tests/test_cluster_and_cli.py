"""Unit tests for active-site cluster extraction, HTML dossier, and CLI."""

import json
from pathlib import Path
import tempfile
import zipfile
import pytest

from shark.core.cluster import extract_active_cluster, QuantumCluster
from shark.reports.dossier import generate_html_dossier
from shark.cli import main as cli_main


@pytest.fixture
def sample_structures(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    rec_pdb.write_text("""ATOM      1  N   ALA A   1      10.000  10.000  10.000
ATOM      2  CA  ALA A   1      11.000  10.000  10.000
HETATM   10  FE  HEM A 500      15.000  10.000  10.000
END
""")
    lig_pdb = tmp_path / "lig_pose.pdb"
    lig_pdb.write_text("""HETATM    1  N1  VOR A   1      14.000  10.000  10.000
HETATM    2  C2  VOR A   1      13.500  10.500  10.000
END
""")
    return rec_pdb, lig_pdb


def test_cluster_extraction(sample_structures, tmp_path):
    rec_pdb, lig_pdb = sample_structures
    cluster = extract_active_cluster(rec_pdb, lig_pdb, cutoff_angstrom=3.5, include_heme=True)
    
    assert isinstance(cluster, QuantumCluster)
    assert cluster.num_atoms >= 2
    # Fe in HEM should set multiplicity to 6
    assert cluster.multiplicity == 6
    
    xyz_path = tmp_path / "cluster.xyz"
    cluster.write_xyz(xyz_path)
    assert xyz_path.is_file()
    
    inp_path = tmp_path / "cluster.inp"
    cluster.generate_orca_input(inp_path, method="r2SCAN-3c")
    assert inp_path.is_file()
    content = inp_path.read_text()
    assert "r2SCAN-3c" in content
    assert "* xyz" in content


def test_html_dossier_generation(tmp_path):
    out_html = tmp_path / "test_dossier.html"
    poses = [
        {"ligand_id": "VOR", "pose_idx": 1, "score": -8.4, "delta_e_bind_kcal": -39.14, "homo_ev": -6.5, "lumo_ev": -2.1, "gap_ev": 4.4}
    ]
    md_summary = {
        "sim_time_ns": 10.0,
        "backbone_rmsd_final": 2.14,
        "mean_rmsf": 0.90,
        "mean_coord_dist": "2.94 +- 0.33"
    }
    p = generate_html_dossier(
        project_name="Unit_Test",
        poses_data=poses,
        out_html=out_html,
        md_summary=md_summary
    )
    assert p.is_file()
    text = p.read_text()
    assert "SharK Quantum & MD Dossier" in text
    assert "VOR" in text
    assert "-39.14 kcal/mol" in text
