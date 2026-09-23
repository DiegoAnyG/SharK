"""Finite-difference Fukui workflow and dossier semantics."""

import json
from pathlib import Path

import pytest

from shark.core.parser import CalculationResult
from shark.core.runner import ProcessResult, run_fukui_analysis
from shark.core.workflow_state import inspect_workflow, workflow_recovery_plan
from shark.reports.dossier import generate_html_dossier


def _completed_reference_job(root: Path) -> Path:
    root.mkdir()
    (root / "geometry.xyz").write_text(
        "5\nmethane\nC 0 0 0\nH 1 0 0\nH -1 0 0\nH 0 1 0\nH 0 -1 0\n",
        encoding="utf-8",
    )
    record = {
        "schema_version": 1,
        "status": "completed",
        "selection": {"ligand_id": "methane"},
        "parameters": {
            "method": "r2SCAN-3c",
            "solvent": "Water",
            "nprocs": 1,
            "maxcore_mb": 256,
            "charge": 0,
            "multiplicity": 1,
            "optimize": False,
            "fukui": True,
        },
        "inputs": {},
        "results": {},
    }
    (root / "job.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def test_three_state_fukui_is_validated_and_resumable(tmp_path, monkeypatch):
    job = _completed_reference_job(tmp_path / "job")
    executable = tmp_path / "orca"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("shark.core.runner.find_orca", lambda: executable)

    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["input_file"])
        Path(kwargs["output_file"]).write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
        return ProcessResult(returncode=0, terminated_normally=True)

    monkeypatch.setattr("shark.core.runner.run_orca_process", fake_run)
    populations = {
        "n": [0.20, -0.05, -0.05, -0.05, -0.05],
        "n_plus_1": [-0.20, -0.20, -0.20, -0.20, -0.20],
        "n_minus_1": [0.60, 0.10, 0.10, 0.10, 0.10],
    }

    def fake_parse(path):
        name = Path(path).name
        return CalculationResult(
            name=name,
            converged=True,
            homo_energy=-8.0,
            lumo_energy=1.0,
            loewdin_charges=populations[name],
            atomic_symbols=["C", "H", "H", "H", "H"],
        )

    monkeypatch.setattr("shark.core.parser.parse_orca_results", fake_parse)
    result = run_fukui_analysis(job, timeout=10)

    assert calls == ["n.inp", "n_plus_1.inp", "n_minus_1.inp"]
    assert result["status"] == "completed" and result["validated"] is True
    assert result["profile"]["fukui_method"] == "finite_difference"
    assert sum(atom["fukui_plus"] for atom in result["profile"]["atoms"]) == pytest.approx(1.0)
    assert "* xyz -1 2" in (job / "fukui" / "n_plus_1.inp").read_text(encoding="utf-8")
    assert "* xyz 1 2" in (job / "fukui" / "n_minus_1.inp").read_text(encoding="utf-8")

    assert run_fukui_analysis(job, timeout=10, resume=True)["status"] == "completed"
    assert calls == ["n.inp", "n_plus_1.inp", "n_minus_1.inp"]
    _, jobs = inspect_workflow(tmp_path)
    assert jobs[0]["fukui_status"] == "completed"
    assert "ligand_fukui:methane" in workflow_recovery_plan(None, jobs)["reusable"]


def test_incomplete_charge_state_is_not_called_finite_difference():
    from shark.analysis.reactivity import build_reactivity_profile

    profile = build_reactivity_profile(
        {
            "homo_ev": -8.0,
            "lumo_ev": 1.0,
            "loewdin_charges": [0.2, -0.2],
            "atomic_symbols": ["C", "O"],
        },
        anion_charges=[-0.3, -0.7],
    )
    assert profile.fukui_method == "mixed_charge_difference_and_proxy"
    assert "not a complete finite-difference" in profile.warnings[0]


def test_dossier_separates_true_fukui_from_candidate_prior(tmp_path):
    fukui = {
        "status": "completed",
        "validated": True,
        "profile": {
            "fukui_method": "finite_difference",
            "atoms": [{
                "atom_index": 0,
                "element": "C",
                "fukui_plus": 0.6,
                "fukui_minus": 0.3,
                "fukui_zero": 0.45,
                "local_electrophilicity": 1.2,
            }],
        },
        "setup": {
            "charge_scheme": "loewdin",
            "geometry_source": "calculation.xyz",
            "states": {
                "n": {"charge": 0, "multiplicity": 1},
                "n_plus_1": {"charge": -1, "multiplicity": 2},
                "n_minus_1": {"charge": 1, "multiplicity": 2},
            },
        },
    }
    job = {
        "status": "completed",
        "selection": {"ligand_id": "fixture"},
        "parameters": {"fukui": True},
        "fukui": fukui,
    }
    adduct = {
        "regiospecificity": {
            "sites": [{
                "atom_index": 0,
                "atom_symbol": "C",
                "candidate_site_score": 0.9,
                "is_target_site": True,
            }]
        }
    }
    report = generate_html_dossier(
        "fixture", [], tmp_path / "dossier.html",
        qm_summary={"jobs": [job]},
        covalent_summary={"adduct_qm": adduct},
    ).read_text(encoding="utf-8")

    assert "Finite-Difference Condensed Fukui Functions" in report
    assert "Target-Biased Candidate Site Prior (Heuristic)" in report
    assert "Local Fukui Electrophilic Function" not in report
