"""Verification suite for Phase 8: Dossier HTML synchronization and JOSS benchmark harness."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from shark.core.analysis_result import SharKAnalysisResult
from shark.reports.dossier import generate_html_dossier
from shark.workflows.benchmark_suite import (
    evaluate_session_benchmark,
    render_benchmark_markdown,
    run_acceptance_verification_suite,
)


def _create_phase8_session(tmp_path: Path) -> Path:
    archive = tmp_path / "phase8_study.poliscreen"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": 1, "full": True, "project": "Phase8Benchmark"}),
        )
        zf.writestr(
            "ligands_meta.csv",
            "name,smiles,role\nBENZ_1OX,C=O,candidate\nBENZ_3OX,CC=O,candidate\n",
        )
        rec_pdb = (
            "ATOM      1  N   THR A 309      10.000  10.000  10.000  1.00 20.00           N\n"
            "ATOM      2  CA  THR A 309      10.500  11.000  10.000  1.00 20.00           C\n"
            "ATOM      3  CB  THR A 309      10.200  11.500  10.000  1.00 20.00           C\n"
            "ATOM      4  OG1 THR A 309      10.000  12.000  10.000  1.00 20.00           O\n"
            "END\n"
        )
        for rec_name in ("6Q9N_ready.pdb", "6Q9N.pdb", "8HTB_ready.pdb", "8HTB.pdb"):
            zf.writestr(f"receptors/{rec_name}", rec_pdb)
        pose_pdb = (
            "ATOM      1  C1  BEN A   1      10.000  12.800  10.000  1.00 20.00           C\n"
            "ATOM      2  O1  BEN A   1      11.200  12.800  10.000  1.00 20.00           O\n"
            "END\n"
        )
        zf.writestr("poses/docking_6Q9N_ready_compounds_a_BENZ_1OX-model1.pdb", pose_pdb)
        zf.writestr("poses/docking_6Q9N_ready~Pk1_compounds_a_BENZ_1OX-model1.pdb", pose_pdb)
        zf.writestr("poses/docking_8HTB_ready~Pk1_compounds_a_BENZ_3OX-model1.pdb", pose_pdb)
        zf.writestr(
            "docking_results.csv",
            "receptor,pose_name,compound_name,docking_score,effectiveness,engine\n"
            "6Q9N_ready,docking_6Q9N_ready_compounds_a_BENZ_1OX-model1,BENZ_1OX,-7.20,74.5,vina\n"
            "6Q9N_ready~Pk1,docking_6Q9N_ready~Pk1_compounds_a_BENZ_1OX-model1,BENZ_1OX,-6.30,65.0,vina\n"
            "8HTB_ready~Pk1,docking_8HTB_ready~Pk1_compounds_a_BENZ_3OX-model1,BENZ_3OX,-6.90,71.2,vina\n",
        )
    return archive


def test_phase8_dossier_renders_all_phase5_6_7_sections(tmp_path):
    out_html = tmp_path / "dossier_phase8.html"
    poses_data = [{"ligand_id": "BENZ_1OX", "pose_idx": 1, "score": -7.20, "homo_ev": -6.80, "lumo_ev": -2.10, "gap_ev": 4.70}]
    pocket_breakdown = [
        {
            "target_id": "6Q9N",
            "pocket_id": "principal",
            "ligand_id": "BENZ_1OX",
            "pose_count": 2,
            "best_score": -7.20,
            "mean_score": -7.05,
            "std_score": 0.15,
            "iqr_score": 0.15,
            "min_score": -7.20,
            "max_score": -6.90,
            "mean_effectiveness": 74.5,
        },
        {
            "target_id": "6Q9N",
            "pocket_id": "Pk1",
            "ligand_id": "BENZ_1OX",
            "pose_count": 1,
            "best_score": -6.30,
            "mean_score": -6.30,
            "std_score": 0.00,
            "iqr_score": 0.00,
            "min_score": -6.30,
            "max_score": -6.30,
            "mean_effectiveness": 65.0,
        },
    ]
    qm_summary = {
        "jobs": [
            {
                "status": "completed",
                "selection": {"ligand_id": "BENZ_1OX", "pocket_breakdown": pocket_breakdown},
                "parameters": {"method": "r2SCAN-3c", "solvent": "Water", "charge": 0, "multiplicity": 1, "fukui": True, "redox": True},
                "results": {
                    "electronic_energy_hartree": -512.345678,
                    "optimization_converged": True,
                    "stationary_minimum_verified": True,
                    "frequencies_cm1": [120.0, 450.0, 1650.0],
                    "imaginary_frequencies_cm1": [],
                    "orbitals": {"0": {"homo": {"energy_eV": -6.80}, "lumo": {"energy_eV": -2.10}, "gap_ev": 4.70}},
                },
                "fukui": {
                    "status": "completed",
                    "validated": True,
                    "charge_scheme": "hirshfeld",
                    "setup": {
                        "charge_scheme": "hirshfeld",
                        "geometry_source": "calculation.xyz",
                        "states": {
                            "n": {"charge": 0, "multiplicity": 1},
                            "n_plus_1": {"charge": -1, "multiplicity": 2},
                            "n_minus_1": {"charge": 1, "multiplicity": 2},
                        },
                    },
                    "states": {
                        "n_plus_1": {"spin_contamination_s2": 0.753, "ideal_s2": 0.750},
                        "n_minus_1": {"spin_contamination_s2": 0.756, "ideal_s2": 0.750},
                    },
                    "validation_metrics": {
                        "valid": True,
                        "sum_f_plus": 1.0000,
                        "sum_f_minus": 1.0000,
                        "sum_f_zero": 1.0000,
                    },
                    "profile": {
                        "fukui_method": "finite_difference",
                        "atoms": [
                            {"atom_index": 0, "element": "C", "fukui_plus": 0.62, "fukui_minus": 0.30, "fukui_zero": 0.46, "local_electrophilicity": 1.31},
                            {"atom_index": 1, "element": "O", "fukui_plus": 0.38, "fukui_minus": 0.70, "fukui_zero": 0.54, "local_electrophilicity": 0.80},
                        ],
                    },
                },
                "redox": {
                    "status": "completed",
                    "thermodynamics": {
                        "vertical_electron_affinity_ev": 2.312,
                        "vertical_ionization_potential_ev": 7.619,
                        "adiabatic_electron_affinity_ev": 2.585,
                        "reorganization_energy_ev": 0.273,
                        "delta_g_solv_red_kcal_mol": -59.61,
                        "e_red_vs_she_v": -1.695,
                        "e_red_vs_sce_v": -1.936,
                        "e_red_vs_ag_agcl_v": -1.892,
                    },
                },
            }
        ]
    }
    md_summary = {
        "run_id": "md_run_1",
        "sim_time_ns": 20.0,
        "status": "completed",
        "md_preset": "amber14sb_tip3p",
        "force_fields": "AMBER14SB (protein) + GAFF2/BCC (ligand) + TIP3P (0.15 M NaCl)",
    }
    covalent_summary = {
        "has_nac": True,
        "clustering": {
            "total_sampled_frames": 200,
            "top_cluster_size": 150,
            "top_cluster_fraction": 0.75,
            "medoid_frame_index": 42,
            "medoid_time_ns": 4.2,
            "p_nac": 0.35,
            "reactive_frame_count": 70,
            "reactive_frame_fraction": 0.35,
            "reactive_medoid_frame_index": 55,
            "reactive_medoid_time_ns": 5.5,
        },
        "transition_state": {
            "status": "completed",
            "is_first_order_ts": True,
            "energy_basis": "gibbs",
            "barrier_symbol": "ΔG‡",
            "reaction_energy_symbol": "ΔG_rxn",
            "delta_g_activation_kcal": 16.40,
            "delta_g_reaction_kcal": -8.20,
            "activation_barrier_kcal": 16.40,
            "reaction_energy_kcal": -8.20,
            "kinetic_feasibility": "Feasible (fast covalent reaction)",
            "half_life": "12.4 s",
            "model_type": "minimal",
            "structure_provenance": {"structure_source": "MD reactive medoid (5.50 ns)"},
            "irc": {
                "connects_reactant_and_product": True,
                "forward_completed": True,
                "backward_completed": True,
                "delta_e_forward_kcal": -14.80,
                "delta_e_backward_kcal": -6.90,
            },
            "homogeneity": {"homogeneous": True, "mismatches": []},
        },
        "total_feasibility": {
            "cfi_final": 0.685,
            "cfi_pre": 0.512,
            "tier": "High Feasibility",
            "affinity_score": 0.700,
            "nac_score": 0.350,
            "ts_score": 0.820,
            "docking_score": -7.20,
            "delta_g_ts": 16.40,
        },
    }
    result = SharKAnalysisResult(
        analysis_id="Phase8Study",
        workflow={
            "requested": "full_gold_standard",
            "executed": "full_gold_standard",
            "status": "completed",
            "degraded": False,
            "recovery": {"reusable": ["preflight", "molecular_dynamics", "tier_4", "ligand_dft"], "review_required": [], "repeat_required": [], "pending": []},
            "stage_timings": {
                "preflight": {"status": "completed", "validation": "validated", "elapsed_seconds": 0.45},
                "ligand_dft": {"status": "completed", "validation": "validated", "elapsed_seconds": 18.20},
            },
        },
        binding={"docking_score": -7.20, "affinity_score": 0.700, "pocket_breakdown": pocket_breakdown},
        dynamics=covalent_summary["clustering"],
        transition_state=covalent_summary["transition_state"],
        feasibility=covalent_summary["total_feasibility"],
    )

    generate_html_dossier(
        "Phase8Study",
        poses_data,
        out_html,
        qm_summary=qm_summary,
        md_summary=md_summary,
        covalent_summary=covalent_summary,
        result=result,
    )
    html_text = out_html.read_text(encoding="utf-8")

    # Verify Phase 5 elements in HTML
    assert "Target &amp; Binding Pocket Disaggregation" in html_text
    assert "Pk1" in html_text
    assert "Formal Outcome: COMPLETED_VALIDATED" in html_text
    assert "Six-Layer Epistemological Evidence Separation" in html_text

    # Verify Phase 6 elements in HTML
    assert "amber14sb_tip3p" in html_text
    assert "Stage Execution &amp; Timing Telemetry" in html_text
    assert "18.20 s" in html_text

    # Verify Phase 7 elements in HTML
    assert "Hirshfeld Population Scheme" in html_text
    assert "Doublet spin check:" in html_text
    assert "Delta-SCF One-Electron Redox Thermodynamics" in html_text
    assert "-1.695 V" in html_text
    assert "-1.936 V" in html_text
    assert "IRC pathway connectivity" in html_text
    assert "Verified (forward ΔE = -14.80 kcal/mol, backward ΔE = -6.90 kcal/mol)" in html_text
    assert "Verified (homogeneous theory/solvent/charge/multiplicity)" in html_text


def test_phase8_benchmark_suite_and_acceptance_invariants(tmp_path):
    acceptance = run_acceptance_verification_suite()
    assert acceptance["passed"] is True
    assert acceptance["formal_outcome_state"] == "NON_REACTIVE_ENSEMBLE"
    assert acceptance["epistemic_layer_count"] == 6

    archive = _create_phase8_session(tmp_path)
    out_dir = tmp_path / "bench_out"
    summary = evaluate_session_benchmark(archive, output_dir=out_dir)
    assert summary["target_count"] == 2
    assert summary["target_pocket_pair_count"] == 3
    assert summary["compound_count"] == 2
    assert (out_dir / "benchmark_report.json").is_file()
    assert (out_dir / "benchmark_summary.md").is_file()
    md_text = render_benchmark_markdown(summary)
    assert "| 6Q9N | principal | BENZ_1OX |" in md_text
    assert "| 6Q9N | Pk1 | BENZ_1OX |" in md_text
    assert "| 8HTB | Pk1 | BENZ_3OX |" in md_text
