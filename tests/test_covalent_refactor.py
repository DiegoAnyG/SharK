"""Validation tests A through H for the covalent feasibility model refactor.

Ensures adherence to COVALENT_MODEL_REFACTOR_SPEC.md:
- Test A: Sigmoidal binding normalization (S_bind ~= 0.696 for -7.244 kcal/mol)
- Test B: Missing attack angle returns None without optimistic default
- Test C: Missing electrophilicity returns None without optimistic default
- Test D: Missing TS barrier leaves CFI_final as None and flags pending calculation
- Test E: Chemistry term dominates impossible chemical reaction (CFI_final < 0.10)
- Test F: Eyring rate constant verification at 310.15 K
- Test G: Separation of kinetic accessibility from thermodynamic spontaneity
- Test H: Wiberg bond index wording avoids 'percent covalent character'
"""

import math
import pytest

from shark.analysis.covalent_matcher import (
    compute_binding_score,
    compute_reactive_geometry_index,
    compute_total_covalent_feasibility,
    TotalCovalentFeasibility,
    ReactiveGeometryResult,
)
from shark.analysis.transition_state import (
    compute_eyring_rate_constant,
    compute_reaction_profile,
    KB_OVER_H,
    R_GAS_KCAL,
)
from shark.analysis.adduct_qm import (
    evaluate_covalent_bond_nature,
    CovalentBondNatureResult,
)


def test_a_binding_normalization():
    """Test A: Verify S_bind sigmoidal normalization against standard reference values."""
    dock_score = -7.244
    ref_score = -6.0
    tau = 1.5

    s_bind = compute_binding_score(dock_score=dock_score, ref_score=ref_score, tau=tau)
    assert s_bind is not None

    # Analytical calculation:
    # z = (-7.244 - (-6.0)) / 1.5 = -1.244 / 1.5 = -0.829333...
    # S_bind = 1 / (1 + exp(-0.829333)) ~= 0.6962
    expected_s_bind = 1.0 / (1.0 + math.exp((dock_score - ref_score) / tau))
    assert pytest.approx(s_bind, abs=0.005) == 0.696
    assert pytest.approx(s_bind, rel=1e-3) == expected_s_bind

    # Missing dock score must return None
    assert compute_binding_score(None) is None


def test_b_missing_angle_no_optimistic_default():
    """Test B: Missing attack angle must yield angle_score = None and incomplete status."""
    res = compute_reactive_geometry_index(
        dist_feasibility=0.85,
        angle_deg=None,
        nucl_resname="CYS",
        candidate_base_present=False,
        omega_k=1.2,
    )
    assert isinstance(res, ReactiveGeometryResult)
    assert res.angle_score is None
    assert res.is_complete is False
    assert res.rgi is None
    assert res.rgi_partial is not None
    assert any("angle" in w.lower() for w in res.warnings)


def test_c_missing_electrophilicity_no_optimistic_default():
    """Test C: Missing electrophilicity must yield electrophilicity_score = None without assuming 1.0."""
    res = compute_reactive_geometry_index(
        dist_feasibility=0.85,
        angle_deg=107.0,
        nucl_resname="THR",
        candidate_base_present=True,
        omega_k=None,
    )
    assert isinstance(res, ReactiveGeometryResult)
    assert res.electrophilicity_score is None
    assert res.is_complete is False
    assert res.rgi is None
    assert res.rgi_partial is not None
    assert any("electrophilicity" in w.lower() for w in res.warnings)


def test_d_missing_ts_leaves_cfi_final_pending():
    """Test D: Missing TS barrier must NOT declare High Covalent Feasibility or redistribute TS weight."""
    res = compute_total_covalent_feasibility(
        docking_score=-7.244,
        p_nac=0.76,
        delta_g_ts=None,
    )
    assert isinstance(res, TotalCovalentFeasibility)
    assert res.cfi_pre is not None
    assert res.cfi_pre > 0.60
    assert res.cfi_final is None
    assert res.ts_score is None
    assert res.status == "Pending transition-state calculation"
    assert res.tier != "High Covalent Feasibility"
    assert "Pre-reactive" in res.tier


def test_e_chemistry_dominates_impossible_reaction():
    """Test E: Extremely high chemical barrier must crush CFI_final regardless of binding affinity."""
    # Super-high affinity (-12 kcal/mol) and perfect NAC persistence (0.98), but impossible TS (35 kcal/mol)
    res = compute_total_covalent_feasibility(
        docking_score=-12.0,
        p_nac=0.98,
        delta_g_ts=35.0,
    )
    assert res.cfi_final is not None
    assert res.cfi_final < 0.10
    assert res.tier == "Low Covalent Feasibility"


def test_f_eyring_rate_constant():
    """Test F: Verify Eyring chemical rate constant and half-life at physiological temperature."""
    delta_g_dagger = 18.0  # kcal/mol
    t_k = 310.15           # 37 C physiological
    kappa = 1.0

    k_chem, half_life_str = compute_eyring_rate_constant(
        delta_g_dagger_kcal=delta_g_dagger,
        temperature_k=t_k,
        kappa=kappa
    )

    rt = R_GAS_KCAL * t_k
    k_expected = kappa * (KB_OVER_H * t_k) * math.exp(-delta_g_dagger / rt)

    assert pytest.approx(k_chem, rel=1e-4) == k_expected
    assert k_chem > 0.5  # At 18 kcal/mol and 310.15 K, k_chem ~ 1.3 s^-1
    assert "second" in half_life_str or "millisecond" in half_life_str


def test_g_kinetic_vs_thermodynamic_labels():
    """Test G: Low activation barrier with endergonic reaction must NOT be labeled spontaneous."""
    # Reactants: -100.0 Eh, TS: -99.976 Eh (barrier ~ 15.06 kcal/mol), Product: -99.990 Eh (DeltaG_rxn = +6.28 kcal/mol)
    profile = compute_reaction_profile(
        reactants_gibbs=-100.0000,
        ts_gibbs=-99.9760,
        product_gibbs=-99.9900,
        temperature_k=310.15,
        is_first_order_ts=True,
    )
    assert profile.delta_g_activation_kcal < 18.0
    assert profile.delta_g_reaction_kcal is not None
    assert profile.delta_g_reaction_kcal > 0.0  # Endergonic!
    assert "spontaneous" not in profile.kinetic_feasibility.lower()
    assert profile.kinetic_feasibility == "Very Rapid Predicted Chemical Step"
    assert "Endergonic" in profile.notes


def test_h_wiberg_wording_no_percent_covalency():
    """Test H: Adduct explanation must describe bond order without misleading 'percent covalent character'."""
    res = evaluate_covalent_bond_nature(
        nucleophile_element="O",
        electrophile_element="C",
        bond_distance_angstrom=1.45,
        is_optimized_adduct=True,
    )
    assert isinstance(res, CovalentBondNatureResult)
    assert "97% covalent character" not in res.explanation
    assert "covalent character" not in res.explanation.lower()
    assert "no Wiberg bond order" in res.explanation
    assert res.wiberg_bond_order is None
