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

