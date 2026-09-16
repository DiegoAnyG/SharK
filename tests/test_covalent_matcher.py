"""Unit tests for Near-Attack Conformation (NAC) covalent pocket matcher."""

from pathlib import Path
import pytest

from shark.analysis.covalent_matcher import (
    parse_pdb_atoms,
    parse_ligand_pose_coordinates,
    extract_pocket_nucleophiles,
    match_covalent_pocket,
    CovalentMatchReport,
    PocketNucleophile,
    DEFAULT_NAC_CUTOFF
)
from shark.analysis.reactivity import LocalReactivityAtom


@pytest.fixture
def synthetic_complex(tmp_path):
    """Creates a mock receptor PDB and ligand pose with a CYS residue at 3.1 A (NAC)."""
    rec_pdb = tmp_path / "receptor.pdb"
    # CYS145: SG at (10.0, 10.0, 10.0)
    # SER200: OG at (18.0, 10.0, 10.0) - far away
    rec_content = (
        "ATOM      1  N   CYS A 145      10.000   8.500  10.000  1.00 20.00           N\n"
        "ATOM      2  CA  CYS A 145      10.000   9.000  10.000  1.00 20.00           C\n"
        "ATOM      3  CB  CYS A 145      10.000   9.500  10.000  1.00 20.00           C\n"
        "ATOM      4  SG  CYS A 145      10.000  10.000  10.000  1.00 20.00           S\n"
        "ATOM      5  N   SER A 200      18.000   8.500  10.000  1.00 20.00           N\n"
        "ATOM      6  OG  SER A 200      18.000  10.000  10.000  1.00 20.00           O\n"
        "END\n"
    )
    rec_pdb.write_text(rec_content, encoding="utf-8")

    # Ligand pose: atom 0 (C) at (10.0, 13.1, 10.0) -> distance to SG is 3.1 A (NAC!)
    # Ligand atom 1 (C) at (10.0, 15.0, 10.0) -> distance to SG is 5.0 A
    lig_pdb = tmp_path / "ligand_pose.pdb"
    lig_content = (
        "HETATM    1  C1  LIG A   1      10.000  13.100  10.000  1.00 20.00           C\n"
        "HETATM    2  C2  LIG A   1      10.000  15.000  10.000  1.00 20.00           C\n"
        "HETATM    3  H1  LIG A   1      10.000  13.100  11.000  1.00 20.00           H\n"
        "END\n"
    )
    lig_pdb.write_text(lig_content, encoding="utf-8")

    return rec_pdb, lig_pdb


def test_parse_pdb_atoms(synthetic_complex):
    rec_pdb, _ = synthetic_complex
    atoms = parse_pdb_atoms(rec_pdb)
    assert len(atoms) == 6
    assert atoms[3]["atom_name"] == "SG"
    assert atoms[3]["res_name"] == "CYS"
    assert atoms[3]["res_seq"] == 145
    assert atoms[3]["coords"] == (10.0, 10.0, 10.0)


def test_parse_ligand_pose_coordinates(synthetic_complex):
    _, lig_pdb = synthetic_complex
    lig_atoms = parse_ligand_pose_coordinates(lig_pdb)
    # H atom should be excluded
    assert len(lig_atoms) == 2
    assert lig_atoms[0][0] == 0
    assert lig_atoms[0][1] == "C"
    assert lig_atoms[0][2] == (10.0, 13.1, 10.0)


def test_extract_pocket_nucleophiles(synthetic_complex):
    rec_pdb, lig_pdb = synthetic_complex
    lig_coords = [(10.0, 13.1, 10.0), (10.0, 15.0, 10.0)]

    nucls = extract_pocket_nucleophiles(rec_pdb, lig_coords, pocket_cutoff=5.0)
    # CYS145 SG is 3.1 A away -> in pocket
    # SER200 OG is ~8.5 A away -> NOT in pocket
    assert len(nucls) == 1
    assert nucls[0].residue_name == "CYS"
    assert nucls[0].residue_number == 145
    assert pytest.approx(nucls[0].min_distance_to_ligand, 0.01) == 3.1


def test_match_covalent_pocket_nac_detected(synthetic_complex):
    rec_pdb, lig_pdb = synthetic_complex
    cand = LocalReactivityAtom(
        atom_index=0,
        element="C",
        charge=0.25,
        local_electrophilicity=1.2,
        is_warhead_candidate=True,
        rank=1
    )

    report = match_covalent_pocket(
        receptor_pdb=rec_pdb,
        ligand_pose=lig_pdb,
        warhead_candidates=[cand],
        pocket_cutoff=5.0,
        nac_cutoff=3.5,
        receptor_name="MockEnzyme",
        ligand_name="Inhibitor_1"
    )

    assert isinstance(report, CovalentMatchReport)
    assert report.has_nac is True
    assert len(report.nac_contacts) == 1
    assert report.best_match is not None
    assert pytest.approx(report.best_match.distance_angstrom, 0.01) == 3.1
    assert report.best_match.is_nac is True
    assert report.best_match.feasibility_score > 0.7
    assert report.best_match.warhead_rank == 1
    assert "Positive Near-Attack Conformation detected" in report.summary


def test_target_residue_filter(synthetic_complex):
    rec_pdb, lig_pdb = synthetic_complex
    
    # Filter for CYS145 -> matches
    rep1 = match_covalent_pocket(rec_pdb, lig_pdb, target_residue="CYS145")
    assert len(rep1.pocket_nucleophiles) == 1

    # Filter for SER -> no SER within pocket
    rep2 = match_covalent_pocket(rec_pdb, lig_pdb, target_residue="SER")
    assert len(rep2.pocket_nucleophiles) == 0
    assert rep2.has_nac is False


def test_compute_total_covalent_feasibility():
    from shark.analysis.covalent_matcher import compute_total_covalent_feasibility, TotalCovalentFeasibility

    # Case 1: High feasibility (strong docking, high P_NAC, low activation barrier)
    res_high = compute_total_covalent_feasibility(
        docking_score=-7.8,
        p_nac=0.76,
        delta_g_ts=17.5,
    )
    assert isinstance(res_high, TotalCovalentFeasibility)
    assert res_high.cfi_total >= 0.70
    assert res_high.tier in ("High Covalent Feasibility", "Moderate Covalent Feasibility")
    assert "Pillar 1" in res_high.summary
    assert "Pillar 2" in res_high.summary
    assert "Pillar 3" in res_high.summary

    # Case 2: Infeasible (very high TS barrier)
    res_low = compute_total_covalent_feasibility(
        docking_score=-4.5,
        p_nac=0.10,
        delta_g_ts=32.0,
    )
    assert res_low.cfi_total < 0.50
    assert res_low.tier == "Low Covalent Feasibility"

    # Case 3: No TS modeled yet (falls back to affinity + trajectory P_NAC)
    res_no_ts = compute_total_covalent_feasibility(
        docking_score=-8.0,
        p_nac=0.80,
        delta_g_ts=None,
    )
    assert res_no_ts.ts_score is None
    assert res_high.cfi_total > 0

