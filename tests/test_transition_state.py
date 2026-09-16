"""Unit tests for Tier 4: Transition State modeling and activation energy engine."""

from pathlib import Path
import pytest

from shark.analysis.qm_cluster import (
    ClusterAtom,
    QMCluster,
    extract_qm_cluster,
    _create_capping_hydrogen,
)
from shark.analysis.transition_state import (
    parse_orca_scan_output,
    parse_orca_ts_output,
    compute_reaction_profile,
    prepare_ts_workflow_directory,
    ScanResult,
    TSVerificationResult,
    ReactionEnergyProfile,
)


@pytest.fixture
def mock_active_site(tmp_path):
    """Creates a mock receptor PDB with THR309 and adjacent residues, plus a ligand pose."""
    rec_pdb = tmp_path / "mock_receptor.pdb"
    # Residue 308 (LEU), Residue 309 (THR), Residue 310 (VAL)
    rec_lines = [
        "ATOM      1  N   LEU A 308     -16.000 -16.000  10.000  1.00 20.00           N",
        "ATOM      2  CA  LEU A 308     -16.200 -16.500  11.000  1.00 20.00           C",
        "ATOM      3  C   LEU A 308     -16.373 -16.798  11.977  1.00 20.00           C",
        "ATOM      4  O   LEU A 308     -16.500 -17.000  12.500  1.00 20.00           O",
        "ATOM      5  N   THR A 309     -16.787 -16.815  13.241  1.00 20.00           N",
        "ATOM      6  H   THR A 309     -16.991 -15.750  13.719  1.00 20.00           H",
        "ATOM      7  CA  THR A 309     -16.144 -17.659  14.240  1.00 20.00           C",
        "ATOM      8  HA  THR A 309     -15.578 -18.560  13.718  1.00 20.00           H",
        "ATOM      9  C   THR A 309     -17.190 -18.491  14.957  1.00 20.00           C",
        "ATOM     10  O   THR A 309     -18.320 -18.023  15.157  1.00 20.00           O",
        "ATOM     11  CB  THR A 309     -15.346 -16.832  15.258  1.00 20.00           C",
        "ATOM     12  HB  THR A 309     -14.898 -17.446  16.179  1.00 20.00           H",
        "ATOM     13  OG1 THR A 309     -16.200 -15.872  15.881  1.00 20.00           O",
        "ATOM     14  HG1 THR A 309     -15.653 -15.524  16.881  1.00 20.00           H",
        "ATOM     15  CG2 THR A 309     -14.184 -16.118  14.570  1.00 20.00           C",
        "ATOM     16  N   VAL A 310     -16.814 -19.716  15.320  1.00 20.00           N",
        "ATOM     17  CA  VAL A 310     -17.000 -20.500  16.000  1.00 20.00           C",
        "END",
    ]
    rec_pdb.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    lig_pdb = tmp_path / "mock_ligand.pdb"
    # Ligand with reactive electrophilic center at 3.2 A from THR309:OG1 (-16.200, -15.872, 15.881)
    lig_lines = [
        "HETATM    1  C1  LIG A   1     -16.200 -15.872  19.081  1.00 20.00           C",
        "HETATM    2  N1  LIG A   1     -15.200 -15.872  19.500  1.00 20.00           N",
        "HETATM    3  O1  LIG A   1     -14.200 -15.872  19.800  1.00 20.00           O",
        "HETATM    4  H1  LIG A   1     -17.000 -15.872  19.500  1.00 20.00           H",
        "END",
    ]
    lig_pdb.write_text("\n".join(lig_lines) + "\n", encoding="utf-8")

    return rec_pdb, lig_pdb


def test_create_capping_hydrogen():
    # Kept N at (0, 0, 0), deleted C at (0, 0, -1.33)
    # Capping H should be at (0, 0, -1.01)
    kept = (0.0, 0.0, 0.0)
    deleted = (0.0, 0.0, -1.33)
    cap = _create_capping_hydrogen(kept, deleted, target_bond_length=1.01)
    assert cap[0] == 0.0
    assert cap[1] == 0.0
    assert pytest.approx(cap[2], 0.01) == -1.01


def test_extract_qm_cluster_minimal(mock_active_site):
    rec_pdb, lig_pdb = mock_active_site
    cluster = extract_qm_cluster(
        receptor_pdb=rec_pdb,
        ligand_pose=lig_pdb,
        model_type="minimal",
        target_residue="THR309",
    )
    assert cluster.n_atoms > 0
    # Ligand (4 atoms) + THR309 (11 atoms) + 2 capping H = 17 atoms
    assert cluster.n_atoms == 17
    assert cluster.model_type == "minimal"
    assert cluster.charge == 0
    assert cluster.multiplicity == 1

    # Nucleophile should be THR309 OG1
    assert cluster.nucleophile_idx is not None
    nucl_atom = cluster.atoms[cluster.nucleophile_idx]
    assert nucl_atom.atom_name == "OG1"
    assert nucl_atom.element == "O"

    # Electrophile should be LIG atom closest to OG1 (C1)
    assert cluster.electrophile_idx is not None
    el_atom = cluster.atoms[cluster.electrophile_idx]
    assert el_atom.res_name == "LIG"

    # Capping atoms check
    caps = [a for a in cluster.atoms if a.is_cap]
    assert len(caps) == 2
    assert any(a.atom_name == "HN_CAP" for a in caps)
    assert any(a.atom_name == "HC_CAP" for a in caps)


def test_extract_qm_cluster_extended(mock_active_site):
    rec_pdb, lig_pdb = mock_active_site
    cluster = extract_qm_cluster(
        receptor_pdb=rec_pdb,
        ligand_pose=lig_pdb,
        model_type="extended",
        cutoff_radius=5.0,
        target_residue="THR309",
        freeze_backbone=True,
    )
    assert cluster.n_atoms > 15
    assert cluster.model_type == "extended"
    # Backbone atoms of non-target residues should be frozen
    assert len(cluster.frozen_indices) > 0


def test_qm_cluster_orca_input(mock_active_site):
    rec_pdb, lig_pdb = mock_active_site
    cluster = extract_qm_cluster(
        receptor_pdb=rec_pdb,
        ligand_pose=lig_pdb,
        model_type="minimal",
        target_residue="THR309",
    )
    # Test scan input
    scan_inp = cluster.to_orca_input(job_type="scan", method="r2SCAN-3c", scan_start=3.2, scan_end=1.45, scan_steps=10)
    assert "! Opt r2SCAN-3c CPCM(Water) TightSCF" in scan_inp
    assert "Scan" in scan_inp
    assert "B " in scan_inp
    assert "* xyz 0 1" in scan_inp

    # Test OptTS input
    optts_inp = cluster.to_orca_input(job_type="optts", method="r2SCAN-3c")
    assert "! OptTS Freq r2SCAN-3c CPCM(Water) TightSCF" in optts_inp
    assert "Calc_Hess true" in optts_inp


def test_parse_orca_scan_output():
    sample_scan_text = """
RELAXED SURFACE SCAN RESULTS
----------------------------

Column   1: NONAME

The Calculated Surface using the 'Actual Energy'
   3.30000000 -678.54000000
   2.80000000 -678.53000000
   2.30000000 -678.51000000
   1.95000000 -678.50000000
   1.65000000 -678.52000000
   1.45000000 -678.56000000
"""
    res = parse_orca_scan_output(sample_scan_text)
    assert res.converged is True
    assert len(res.points) == 6
    assert res.max_energy_step == 4
    assert res.ts_guess_coord_value == 1.95
    # Barrier = (-678.500 - -678.540) * 627.509 = 0.040 * 627.509 = 25.10 kcal/mol
    assert pytest.approx(res.barrier_estimate_kcal, 0.2) == 25.10


def test_parse_orca_ts_output_valid_ts():
    mock_ts_out = """
VIBRATIONAL FREQUENCIES
-----------------------
Scaling factor for frequencies =  1.000000000  (already applied!)
     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:       0.00 cm**-1
     6:    -350.25 cm**-1
     7:      55.10 cm**-1
     8:     120.40 cm**-1

IR SPECTRUM
-----------
Mode   freq       eps      Int      T**2         TX        TY        TZ
   6:   -350.25   0.001000   15.00  0.000010   0.001   0.002   0.003
   7:     55.10   0.000500    2.00  0.000005   0.001   0.001   0.001

Final Gibbs free energy         ...   -678.50500000 Eh
                             ****ORCA TERMINATED NORMALLY****
"""
    res = parse_orca_ts_output(mock_ts_out, name="Test_TS")
    assert res.converged is True
    assert res.n_imaginary_frequencies == 1
    assert res.imaginary_modes == [-350.25]
    assert res.is_valid_first_order_saddle_point is True
    assert pytest.approx(res.gibbs_free_energy_hartree, 1e-5) == -678.50500000


def test_parse_orca_ts_output_invalid_ts():
    # Stationary minimum (0 imaginary modes)
    mock_min_out = """
VIBRATIONAL FREQUENCIES
-----------------------
     0:       0.00 cm**-1
     6:      50.20 cm**-1
     7:     120.40 cm**-1

Final Gibbs free energy         ...   -678.54000000 Eh
                             ****ORCA TERMINATED NORMALLY****
"""
    res = parse_orca_ts_output(mock_min_out, name="Test_Min")
    assert res.converged is True
    assert res.n_imaginary_frequencies == 0
    assert res.is_valid_first_order_saddle_point is False


def test_compute_reaction_profile():
    # Ground state: -678.5400 Eh
    # TS:           -678.5050 Eh -> delta G‡ = +0.0350 Eh = 21.96 kcal/mol
    # Product:      -678.5600 Eh -> delta G_rxn = -0.0200 Eh = -12.55 kcal/mol
    profile = compute_reaction_profile(
        reactants_gibbs=-678.5400,
        ts_gibbs=-678.5050,
        product_gibbs=-678.5600,
        temperature_k=298.15,
        is_first_order_ts=True,
    )
    assert pytest.approx(profile.delta_g_activation_kcal, 0.1) == 21.96
    assert pytest.approx(profile.delta_g_reaction_kcal, 0.1) == -12.55
    assert profile.kinetic_feasibility == "High Covalent Feasibility"
    assert "minutes" in profile.estimated_half_life_str or "seconds" in profile.estimated_half_life_str
    assert profile.is_first_order_ts is True


def test_prepare_ts_workflow_directory(mock_active_site, tmp_path):
    rec_pdb, lig_pdb = mock_active_site
    cluster = extract_qm_cluster(
        receptor_pdb=rec_pdb,
        ligand_pose=lig_pdb,
        model_type="minimal",
        target_residue="THR309",
    )
    wf_dir = tmp_path / "ts_wf"
    wf_dict = prepare_ts_workflow_directory(cluster, wf_dir)
    assert wf_dict["scan_inp"].is_file()
    assert wf_dict["initial_xyz"].is_file()
    assert wf_dict["optts_template"].is_file()
    assert wf_dict["run_script"].is_file()
    assert "run_tier4_ts.sh" in str(wf_dict["run_script"])

