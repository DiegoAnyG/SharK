"""Unit tests for the Adduct Quantum evaluation module (FMO, Polarization, Regiospecificity, Bond Nature)."""

import pytest
from shark.analysis.adduct_qm import (
    evaluate_fmo_phase_symmetry,
    evaluate_pocket_polarization,
    evaluate_regiospecificity,
    evaluate_covalent_bond_nature,
    compute_adduct_quantum_profile,
)


def test_evaluate_fmo_phase_symmetry():
    # Optimal attack trajectory (132.7 deg, 3.33 A)
    res = evaluate_fmo_phase_symmetry(
        nucleophile_homo_ev=-6.40,
        electrophile_lumo_ev=-2.90,
        distance_angstrom=3.33,
        burgi_dunitz_angle_deg=132.7,
        nucleophile_symbol="O",
        electrophile_symbol="N",
    )
    assert res.is_allowed is True
    assert "Constructive Phase Overlap" in res.status
    assert res.fmo_energy_gap_ev == pytest.approx(3.50, 0.01)
    assert res.overlap_integral_estimate > 0.0

    # Unfavorable trajectory (orthogonal angle 45 deg, far distance 5.5 A)
    res_bad = evaluate_fmo_phase_symmetry(
        nucleophile_homo_ev=-6.40,
        electrophile_lumo_ev=-2.90,
        distance_angstrom=5.50,
        burgi_dunitz_angle_deg=45.0,
    )
    assert res_bad.is_allowed is False
    assert "Destructive" in res_bad.status or "Hindered" in res_bad.status


def test_evaluate_pocket_polarization():
    res = evaluate_pocket_polarization(
        isolated_homo_ev=-6.34,
        isolated_lumo_ev=-2.89,
        has_catalytic_partner=True,
    )
    assert res.delta_lumo_ev < 0.0  # LUMO lowered (electrophilic activation)
    assert res.complex_electrophilicity_ev > res.isolated_electrophilicity_ev
    assert res.stabilization_kcal_mol > 0.0
    assert "Electrophilic Activation" in res.polarization_effect


def test_evaluate_regiospecificity():
    heavy_atoms = [
        ("C", 1), ("C", 2), ("C", 3), ("O", 4), ("C", 5),
        ("C", 6), ("C", 7), ("C", 8), ("N", 9), ("O", 10),
        ("O", 11), ("N", 12), ("O", 13)
    ]
    res = evaluate_regiospecificity(
        ligand_heavy_atoms=heavy_atoms,
        target_atom_index=10,
        target_atom_symbol="O",
    )
    assert res.target_rank == 1
    assert res.is_primary_locus is True
    assert len(res.sites) == len(heavy_atoms)
    assert sum(s.fukui_electrophilic for s in res.sites) == pytest.approx(1.0, 0.01)


def test_evaluate_covalent_bond_nature():
    # C-O covalent single bond at 1.45 A
    res = evaluate_covalent_bond_nature(
        nucleophile_element="O",
        electrophile_element="C",
        bond_distance_angstrom=1.45,
        is_optimized_adduct=True,
    )
    assert res.wiberg_bond_order is None
    assert res.distance_based_bond_order_proxy > 0.85
    assert res.bond_covalency_percent is None
    assert res.charge_transfer_e is None
    assert res.status == "geometry_optimized"


def test_compute_adduct_quantum_profile():
    profile = compute_adduct_quantum_profile(
        distance_angstrom=3.33,
        burgi_dunitz_angle_deg=132.7,
        nucleophile_homo_ev=-6.40,
        electrophile_lumo_ev=-2.90,
        target_atom_index=10,
        target_atom_symbol="O",
    )
    d = profile.to_dict()
    assert "fmo_symmetry" in d
    assert "polarization" in d
    assert "regiospecificity" in d
    assert "bond_nature" in d
    assert d["fmo_symmetry"]["is_allowed"] is True
    assert d["regiospecificity"]["target_rank"] == 1


def test_short_distance_does_not_imply_optimized_adduct():
    res = evaluate_covalent_bond_nature(
        nucleophile_element="O",
        electrophile_element="C",
        bond_distance_angstrom=1.45,
    )
    assert res.status == "not_evaluated"
    assert res.is_calculated is False
