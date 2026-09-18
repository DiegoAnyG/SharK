"""Unit tests for Tier 4 3Dmol.js adduct visualizer."""

from pathlib import Path
import pytest

from shark.reports.adduct_viewer import (
    _get_3dmol_js,
    build_adduct_pdb,
    generate_adduct_viewer_html,
)


def test_get_3dmol_js():
    js_content = _get_3dmol_js()
    assert isinstance(js_content, str)
    assert len(js_content) > 1000
    assert "$3Dmol" in js_content


def test_build_adduct_pdb(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    rec_pdb.write_text(
        "ATOM      1  OG1 THR A 309      10.000  10.000  10.000  1.00 20.00           O\n"
        "ATOM      2  OD1 ASP A 199      12.000  10.000  10.000  1.00 20.00           O\n"
        "ATOM      3  CA  GLY A 100      30.000  30.000  30.000  1.00 20.00           C\n"
        "END\n"
    )

    lig_pdb = tmp_path / "lig.pdb"
    lig_pdb.write_text(
        "HETATM    1  C1  UNL A   1      10.000  13.300  10.000  1.00 20.00           C\n"
        "END\n"
    )

    adduct_pdb = build_adduct_pdb(
        ligand_pdb_or_xyz=lig_pdb,
        receptor_pdb=rec_pdb,
        target_residue="THR309",
        dyad_residue="ASP199"
    )

    assert "THR A 309" in adduct_pdb
    assert "ASP A 199" in adduct_pdb
    assert "GLY A 100" not in adduct_pdb
    assert "LIG" in adduct_pdb
    assert "END" in adduct_pdb


def test_generate_adduct_viewer_html():
    pdb_snippet = (
        "ATOM      1  OG1 THR A 309      10.000  10.000  10.000  1.00 20.00           O\n"
        "ATOM      2  OD1 ASP A 199      12.000  10.000  10.000  1.00 20.00           O\n"
        "HETATM    3  C1  LIG A   1      10.000  13.330  10.000  1.00 20.00           C\n"
        "END\n"
    )

    html_snippet = generate_adduct_viewer_html(
        pdb_data=pdb_snippet,
        target_residue="THR309",
        nucl_atom_coords=(10.0, 10.0, 10.0),
        el_atom_coords=(10.0, 13.33, 10.0),
        attack_distance=3.33,
        burgi_dunitz_angle=107.2,
        dyad_residue="ASP199",
        standalone=False
    )

    assert "adduct-viewer-container" in html_snippet
    assert "adduct_viewer_3dmol" in html_snippet
    assert "3.33" in html_snippet
    assert "107.2" in html_snippet
    assert "Thr309" in html_snippet
    assert "$3Dmol" in html_snippet

    # Test standalone mode
    html_standalone = generate_adduct_viewer_html(
        pdb_data=pdb_snippet,
        target_residue="THR309",
        standalone=True
    )
    assert "<!doctype html>" in html_standalone
    assert "</html>" in html_standalone


def test_generate_adduct_viewer_with_orca_cluster_cubes():
    pdb_snippet = "ATOM      1  OG1 THR A 309      10.000  10.000  10.000  1.00 20.00           O\nEND\n"
    cluster_qm = {
        "cluster_name": "QM_Cluster_THR309_minimal",
        "n_atoms": 43,
        "charge": 0,
        "multiplicity": 1,
        "method": "r2SCAN-3c",
        "solvent": "Water",
        "homo_idx": 78,
        "lumo_idx": 79,
        "homo_energy_ev": -3.0757,
        "lumo_energy_ev": -2.3381,
        "gap_ev": 0.7376,
        "success": True,
    }

    mock_lumo_cube = "Mock LUMO CUBE header\n  1  0.0 0.0 0.0\n"
    mock_homo_cube = "Mock HOMO CUBE header\n  1  0.0 0.0 0.0\n"

    html = generate_adduct_viewer_html(
        pdb_data=pdb_snippet,
        target_residue="THR309",
        homo_cube_data=mock_homo_cube,
        lumo_cube_data=mock_lumo_cube,
        cluster_qm_data=cluster_qm,
        standalone=False
    )

    assert "ORCA ab initio (r2SCAN-3c)" in html
    assert "LUMO (MO 79)" in html
    assert "HOMO (MO 78)" in html
    assert "-2.34 eV" in html or "-2.338" in html
    assert "Toggle Labels" in html
    assert "clearLabels()" in html
    assert "createLabels()" in html


def test_generate_adduct_viewer_with_none_values():
    pdb_snippet = "ATOM      1  OG1 THR A 309      10.000  10.000  10.000  1.00 20.00           O\nEND\n"
    cluster_qm_none = {
        "cluster_name": "QM_Cluster_None",
        "homo_idx": None,
        "lumo_idx": None,
        "homo_energy_ev": None,
        "lumo_energy_ev": None,
        "gap_ev": None,
        "success": False,
    }
    adduct_qm_none = {
        "fmo_symmetry": {
            "symmetry_type": "sigma-type",
            "overlap_integral_estimate": None,
            "fmo_energy_gap_ev": None,
        },
        "polarization": {
            "delta_lumo_ev": None,
            "delta_electrophilicity_ev": None,
            "stabilization_kcal_mol": None,
        },
        "regiospecificity": {
            "target_atom_index": 0,
            "target_atom_symbol": "C",
            "target_rank": 1,
            "sites": [{"fukui_electrophilic": None}],
        },
        "bond_nature": {
            "wiberg_bond_order": None,
            "charge_transfer_e": None,
            "bond_type": "Pre-reactive Contact (Unreacted)",
        },
    }

    html = generate_adduct_viewer_html(
        pdb_data=pdb_snippet,
        target_residue="THR309",
        attack_distance=None,
        burgi_dunitz_angle=None,
        dyad_residue=None,
        cluster_qm_data=cluster_qm_none,
        adduct_qm_data=adduct_qm_none,
        standalone=False
    )
    assert "adduct-viewer-container" in html
    assert "Pre-reactive Contact (Unreacted)" in html
    assert "Wiberg BO = N/A" in html


