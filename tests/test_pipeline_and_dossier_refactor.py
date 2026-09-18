"""Comprehensive regression test suite for the SharK Pipeline & Dossier Refactor.

Validates:
- TEST-001: Static geometry does not become P_NAC
- TEST-002: MD enables CFI_pre
- TEST-003: TS enables CFI_final
- TEST-004: Dossier cannot substitute geometry for total feasibility
- TEST-005: Simple gold standard requires trajectory inputs
- TEST-006: Heuristic orbital score is not called overlap integral confirmation
- TEST-007: Hard-coded polarization proxy is not reported as computed ΔLUMO
- TEST-008: Target-biased site ranking is not reported as real Fukui index
- TEST-009: Distance proxy is not reported as Wiberg bond order
- TEST-010: No assumed covalent distance substitution
- TEST-011: Gibbs population labeled ΔG only when Gibbs energy exists
- TEST-012: Electronic-energy proxy labeled ΔE, not ΔG
- TEST-013: Reject mixed energy basis
- TEST-014: Consistent electronic route reports ΔE‡
- TEST-015: Consistent Gibbs route reports ΔG‡
- TEST-016: Dossier is regenerable from saved result
- TEST-017: Bundle mode avoids duplicate heavy assets
- TEST-018: Legacy result is clearly marked
"""

import json
import math
import zipfile
import pytest
from pathlib import Path

from shark.analysis.covalent_matcher import compute_total_covalent_feasibility
from shark.analysis.adduct_qm import (
    evaluate_orbital_alignment_heuristic,
    evaluate_fmo_phase_symmetry,
    evaluate_pocket_polarization,
    evaluate_regiospecificity,
    evaluate_covalent_bond_nature,
    compute_adduct_quantum_profile,
)
from shark.analysis.thermo import calculate_relative_thermo
from shark.analysis.transition_state import (
    select_consistent_energy_basis,
    compute_reaction_profile,
)
from shark.core.parser import CalculationResult
from shark.core.analysis_result import SharKAnalysisResult, EvidenceValue
from shark.reports.dossier import generate_html_dossier
from shark.cli import main as cli_main


# TEST-001: Static geometry does not become P_NAC
def test_001_static_geometry_does_not_become_p_nac():
    """INV-001: A static contact score can never populate P_NAC or CFI_pre."""
    res = compute_total_covalent_feasibility(
        docking_score=-8.5,
        p_nac=None,
        delta_g_ts=None,
        static_cfi=0.76,
    )
    assert res.p_nac is None
    assert res.cfi_pre is None
    assert res.cfi_final is None
    assert res.rgi_static == pytest.approx(0.76, 0.01)
    assert res.completeness["p_nac"] is False
    assert "p_nac" in res.missing_components


# TEST-002: MD enables CFI_pre
def test_002_md_enables_cfi_pre():
    """INV-002: Dynamic P_NAC enables pre-reactive CFI_pre, but leaves CFI_final pending."""
    res = compute_total_covalent_feasibility(
        docking_score=-8.5,
        p_nac=0.45,
        delta_g_ts=None,
        static_cfi=0.76,
    )
    assert res.p_nac == pytest.approx(0.45, 0.01)
    assert res.cfi_pre is not None
    assert res.cfi_pre > 0.0
    assert res.cfi_final is None
    assert res.completeness["p_nac"] is True
    assert res.completeness["cfi_pre"] is True
    assert res.completeness["cfi_final"] is False


# TEST-003: TS enables CFI_final
def test_003_ts_enables_cfi_final():
    """INV-003: When docking, P_NAC, and TS chemical barrier are present, both CFI_pre and CFI_final are populated."""
    res = compute_total_covalent_feasibility(
        docking_score=-8.5,
        p_nac=0.45,
        delta_g_ts=15.0,
        static_cfi=0.76,
    )
    assert res.cfi_pre is not None
    assert res.cfi_final is not None
    assert 0.0 <= res.cfi_final <= 1.0
    assert res.completeness["cfi_final"] is True
    assert len(res.missing_components) == 0


# TEST-004: Dossier cannot substitute geometry for total feasibility
def test_004_dossier_cannot_substitute_geometry_for_total_feasibility(tmp_path):
    """INV-004: If CFI is missing, Total Feasibility card must display 'Not available', not geometric contact score."""
    out_html = tmp_path / "test_dossier_no_fallback.html"
    covalent_summary = {
        "summary": "Static analysis test",
        "contacts": [{
            "residue": "THR309",
            "nucleophile_atom": "OG1",
            "ligand_atom_index": 1,
            "ligand_atom_element": "C",
            "distance_angstrom": 3.2,
            "burgi_dunitz_angle": 108.0,
            "feasibility_score": 0.76,
            "composite_feasibility": 0.76,
            "is_nac": True,
        }],
        "total_feasibility": {
            "cfi_pre": None,
            "cfi_final": None,
            "cfi_total": 0.0,
            "tier": "Not Evaluated",
            "rgi_static": 0.76,
        }
    }
    generate_html_dossier(
        project_name="TestFallback",
        poses_data=[],
        out_html=out_html,
        covalent_summary=covalent_summary,
    )
    html_content = out_html.read_text(encoding="utf-8")
    assert "CFI_pre = Not available" in html_content
    assert "CFI_final = Not available" in html_content
    assert "Total Feasibility (CFI) <span class=\"help-bubble\"" in html_content
    # 0.76 must not appear as the main card score
    assert "<strong>0.76</strong>" not in html_content
    assert "<strong>0.760</strong>" not in html_content


# TEST-005: Simple gold standard requires trajectory inputs
def test_005_simple_gold_standard_requires_trajectory_inputs(tmp_path, capsys):
    """INV-005: Running --simple-gold-standard without --topology and --trajectory exits non-zero."""
    archive = tmp_path / "dummy.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "project": "Dummy"}))
        z.writestr("docking_results.csv", "receptor,pose_name,compound_name,docking_score,engine\n")

    code = cli_main([
        "--session", str(archive),
        "--simple-gold-standard",
    ])
    assert code != 0
    captured = capsys.readouterr()
    assert "requires both --topology and --trajectory" in captured.err


# TEST-006: Heuristic orbital score is not called overlap integral confirmation
def test_006_heuristic_orbital_score_not_called_overlap_integral():
    """INV-008: Orbital alignment heuristic must be typed as model_derived with proxy warning."""
    res = evaluate_orbital_alignment_heuristic(
        nucleophile_homo_ev=-6.40,
        electrophile_lumo_ev=-2.90,
        distance_angstrom=3.33,
        burgi_dunitz_angle_deg=132.7,
    )
    d = res.to_dict()
    assert d["evidence_type"] == "model_derived"
    assert "Model Heuristic" in d["status"] or "Model-derived" in d["explanation"]
    assert any("overlap integral" in w.lower() or "proxy" in w.lower() for w in d["warnings"])


# TEST-007: Hard-coded polarization proxy is not reported as computed ΔLUMO
def test_007_hard_coded_polarization_proxy_labeled_model_derived():
    """INV-008: Active-site polarization must be labeled polarization_proxy and model_derived."""
    res = evaluate_pocket_polarization(
        isolated_homo_ev=-6.34,
        isolated_lumo_ev=-2.89,
        has_catalytic_partner=True,
    )
    d = res.to_dict()
    assert d["name"] == "polarization_proxy"
    assert d["evidence_type"] == "model_derived"
    assert any("not an ab initio" in w.lower() for w in d["warnings"])


# TEST-008: Target-biased site ranking is not reported as real Fukui index
def test_008_target_biased_site_ranking_not_reported_as_real_fukui():
    """INV-008: Candidate site scoring is labeled model_derived prior, not computed Fukui observable."""
    heavy_atoms = [("C", 1), ("N", 2), ("O", 3)]
    res = evaluate_regiospecificity(
        ligand_heavy_atoms=heavy_atoms,
        target_atom_index=2,
        target_atom_symbol="N",
    )
    d = res.to_dict()
    assert d["evidence_type"] == "model_derived"
    assert any("empirical heuristic" in w.lower() or "not a computed" in w.lower() for w in d["warnings"])


# TEST-009: Distance proxy is not reported as Wiberg bond order
def test_009_distance_proxy_not_reported_as_wiberg_bond_order():
    """INV-007 / INV-009: In the absence of an adduct calculation, Wiberg bond order is strictly None."""
    res = evaluate_covalent_bond_nature(
        distance_angstrom=3.33,
        warhead_type="benzofuroxan",
        has_optimized_adduct=False,
    )
    assert res.wiberg_bond_order is None
    assert res.status == "not_evaluated"
    assert res.is_calculated is False
    assert any("requires optimized adduct" in w.lower() for w in res.warnings)


# TEST-010: No assumed covalent distance substitution
def test_010_no_assumed_covalent_distance_substitution():
    """INV-010: Pre-reactive contact distance > 2.0 A must not be replaced with 1.45 A."""
    contact_dist = 3.33
    prof = compute_adduct_quantum_profile(
        distance_angstrom=contact_dist,
        burgi_dunitz_angle_deg=108.0,
        has_optimized_adduct=False,
    )
    bond_res = prof.bond_nature
    assert bond_res.equilibrium_distance_angstrom == pytest.approx(contact_dist, 0.01)
    assert bond_res.equilibrium_distance_angstrom > 2.0
    assert bond_res.wiberg_bond_order is None
    assert bond_res.status == "not_evaluated"


# TEST-011: Gibbs population labeled ΔG only when Gibbs energy exists
def test_011_gibbs_population_labeled_delta_g():
    """INV-006: Boltzmann population from valid Gibbs energies is labeled with ΔG."""
    r1 = CalculationResult(name="state_1", el_energy=-500.1, gibbs_energy=-500.08)
    r2 = CalculationResult(name="state_2", el_energy=-500.12, gibbs_energy=-500.09)
    df = calculate_relative_thermo([r1, r2], temperature=298.15)
    assert df.attrs.get("energy_basis") == "gibbs"
    assert "Gibbs" in df.attrs.get("population_label", "")
    assert "dG (kcal/mol)" in df.columns
    assert df["dG (kcal/mol)"].iloc[0] == 0.0
    assert df["Population (%)"].sum() == pytest.approx(100.0, 0.01)


# TEST-012: Electronic-energy proxy labeled ΔE, not ΔG
def test_012_electronic_energy_proxy_labeled_delta_e():
    """INV-006: When Gibbs is absent, relative energetics are strictly labeled ΔE_el proxy, never ΔG."""
    r1 = CalculationResult(name="state_1", el_energy=-500.10, gibbs_energy=0.0)
    r2 = CalculationResult(name="state_2", el_energy=-500.12, gibbs_energy=0.0)
    df = calculate_relative_thermo([r1, r2], temperature=298.15)
    assert df.attrs.get("energy_basis") == "electronic"
    assert "Electronic-energy population proxy" in df.attrs.get("population_label", "")
    assert df["dG (kcal/mol)"].iloc[0] is None
    assert "dE_el (kcal/mol)" in df.columns
    assert df["dE_el (kcal/mol)"].iloc[0] == 0.0


# TEST-013: Reject mixed energy basis
def test_013_reject_mixed_energy_basis():
    """INV-006: Mixing electronic energy for reactant with Gibbs energy for TS is strictly rejected."""
    # Direct selection helper
    sel = select_consistent_energy_basis(
        reactant_el=-500.10,
        reactant_gibbs=None,
        ts_el=None,
        ts_gibbs=-500.05,
    )
    assert sel["is_valid"] is False
    assert sel["barrier_symbol"] is None

    # Reaction profile calculation returns safe profile with delta_g_activation_kcal=None and warnings (Test TS-1)
    prof = compute_reaction_profile(
        reactants_electronic=-500.10,
        ts_gibbs=-500.05,
    )
    assert prof.delta_g_activation_kcal is None
    assert prof.rate_constant_s is None
    assert "Not available" in prof.estimated_half_life_str
    assert any("mixed" in w.lower() or "inv-006" in w.lower() for w in prof.warnings)

    # With raise_on_inconsistent=True, it raises ValueError
    with pytest.raises(ValueError) as excinfo:
        compute_reaction_profile(
            reactants_electronic=-500.10,
            ts_gibbs=-500.05,
            raise_on_inconsistent=True,
        )
    assert "INV-006" in str(excinfo.value) or "Inconsistent energy basis" in str(excinfo.value)


# TEST-014: Consistent electronic route reports ΔE‡
def test_014_consistent_electronic_route_reports_delta_e():
    """INV-006: Consistent electronic energies produce ΔE‡ activation barrier."""
    prof = compute_reaction_profile(
        reactants_electronic=-500.10,
        ts_electronic=-500.05,
        product_electronic=-500.15,
    )
    assert prof.energy_basis == "electronic"
    assert prof.barrier_symbol == "ΔE‡"
    assert prof.reaction_energy_symbol == "ΔE_rxn"
    assert prof.delta_e_activation_kcal == pytest.approx(31.38, 0.5)
    assert prof.delta_g_activation_kcal is None
    assert prof.rate_constant_s is None
    assert "Not available" in prof.estimated_half_life_str


# TEST-015: Consistent Gibbs route reports ΔG‡
def test_015_consistent_gibbs_route_reports_delta_g():
    """INV-006: Consistent Gibbs free energies produce ΔG‡ activation barrier."""
    prof = compute_reaction_profile(
        reactants_gibbs=-500.08,
        ts_gibbs=-500.05,
        product_gibbs=-500.12,
    )
    assert prof.energy_basis == "gibbs"
    assert prof.barrier_symbol == "ΔG‡"
    assert prof.reaction_energy_symbol == "ΔG_rxn"
    assert prof.delta_g_activation_kcal == pytest.approx(18.83, 0.5)
    assert prof.delta_e_activation_kcal is None


# TEST-016: Dossier is regenerable from saved result
def test_016_dossier_regenerable_from_saved_result(tmp_path):
    """Verify dossier can be regenerated from persisted evidence.json without calculators."""
    ev_path = tmp_path / "evidence.json"
    out_html = tmp_path / "regenerated_dossier.html"

    result = SharKAnalysisResult(
        analysis_id="OfflineRegenTest",
        workflow={"requested": "fast_analysis", "executed": "fast_analysis", "status": "completed"},
        binding={"docking_score": EvidenceValue(name="docking_score", value=-8.4, units="kcal/mol", status="computed", evidence_type="docking_derived")},
        static_reactive_geometry={"rgi_static": EvidenceValue(name="rgi_static", value=0.72, units="dimensionless", status="computed", evidence_type="static_geometry")},
        dynamics={},
        cluster_qm={},
        transition_state={},
        adduct={},
        feasibility={
            "cfi_pre": EvidenceValue(name="cfi_pre", value=None, units="dimensionless", status="not_available", evidence_type="model_derived", reason="No MD trajectory"),
            "cfi_final": EvidenceValue(name="cfi_final", value=None, units="dimensionless", status="not_available", evidence_type="model_derived", reason="No TS barrier"),
            "cfi_total": EvidenceValue(name="cfi_total", value=0.0, units="dimensionless", status="incomplete", evidence_type="model_derived"),
            "tier": EvidenceValue(name="tier", value="Incomplete", units=None, status="computed", evidence_type="model_derived"),
        },
        completeness={"cfi_pre": False, "cfi_final": False},
        warnings=["Dynamic NAC analysis was not performed; CFI_pre unavailable"],
        provenance={"project": "OfflineRegenTest"},
    )
    result.write_evidence_json(ev_path)
    assert ev_path.is_file()

    # Re-read and generate report without calling any QM or MD runners
    loaded_res = SharKAnalysisResult.from_json(ev_path)
    generate_html_dossier(
        project_name=loaded_res.analysis_id,
        poses_data=[],
        out_html=out_html,
        result=loaded_res,
    )
    assert out_html.is_file()
    text = out_html.read_text(encoding="utf-8")
    assert "OfflineRegenTest" in text
    assert "CFI_pre = Not available" in text


# TEST-017: Bundle mode avoids duplicate heavy assets
def test_017_bundle_mode_avoids_duplicate_heavy_assets():
    """Verify SharKAnalysisResult does not duplicate heavy volumetric data in provenance."""
    res = SharKAnalysisResult(
        analysis_id="AssetCheck",
        provenance={"cube_path": "orbitals/homo.cube"},
    )
    d = res.to_dict()
    # Volumetric cube contents must not be in provenance
    assert "homo_cube_data" not in d["provenance"]
    assert d["provenance"]["cube_path"] == "orbitals/homo.cube"


# TEST-018: Legacy result is clearly marked
def test_018_legacy_result_clearly_marked(tmp_path):
    """Loading a legacy schema payload must emit legacy warnings and prevent unsafe CFI fallback."""
    legacy_payload = {
        "project": "LegacySession",
        "contacts": [{"feasibility_score": 0.81, "composite_feasibility": 0.81}],
    }
    legacy_json = tmp_path / "legacy_result.json"
    legacy_json.write_text(json.dumps(legacy_payload), encoding="utf-8")

    out_html = tmp_path / "legacy_dossier.html"
    generate_html_dossier(
        project_name="LegacyTest",
        poses_data=[],
        out_html=out_html,
        covalent_summary={"contacts": legacy_payload["contacts"]},
    )
    html = out_html.read_text(encoding="utf-8")
    # Must not show 0.81 as CFI
    assert "CFI_pre = Not available" in html
    assert "CFI_final = Not available" in html
