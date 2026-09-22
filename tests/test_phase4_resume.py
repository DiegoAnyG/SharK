"""Phase 4 checkpoint and dependency-continuation boundaries."""

import json
from pathlib import Path

from shark.cli import main
from shark.core.runner import ProcessResult
from shark.core.workflow_state import initialize_workflow_manifest, update_workflow_stage
from shark.workflows.tier4_resume import (
    resume_tier4_workflow,
    tier4_stage_reusable,
    write_tier4_stage_record,
)


SCAN_OUTPUT = """
RELAXED SURFACE SCAN RESULTS
----------------------------
The Calculated Surface using the 'Actual Energy'
  3.00000000 -100.00000000
  2.00000000  -99.90000000

ORCA TERMINATED NORMALLY
"""

VALID_TS_OUTPUT = """
THE OPTIMIZATION HAS CONVERGED
VIBRATIONAL FREQUENCIES
-----------------------
  0: -500.00 cm**-1
  1:   30.00 cm**-1

ORCA TERMINATED NORMALLY
"""

INVALID_TS_OUTPUT = """
THE OPTIMIZATION HAS CONVERGED
VIBRATIONAL FREQUENCIES
-----------------------
  0:   20.00 cm**-1
  1:   30.00 cm**-1

ORCA TERMINATED NORMALLY
"""


def _prepared_tier4(tmp_path: Path) -> Path:
    root = tmp_path / "transition_state"
    root.mkdir()
    (root / "reaction_definition.json").write_text(json.dumps({
        "fingerprint": "reaction-fingerprint",
        "validation_status": "INFO",
    }), encoding="utf-8")
    (root / "01_scan.inp").write_text(
        "%geom\n Scan\n  B 1 2 = 3.00, 2.00, 2\n end\nend\n* xyz 0 1\nH 0 0 0\nH 0 0 3\n*\n",
        encoding="utf-8",
    )
    (root / "02_optts_template.inp").write_text(
        "! OptTS Freq\n* xyz 0 1\nH 0 0 0\nH 0 0 2\n*\n",
        encoding="utf-8",
    )
    (root / "03_adduct_opt_template.inp").write_text(
        "! Opt\n* xyz 0 1\nH 0 0 0\nH 0 0 2\n*\n",
        encoding="utf-8",
    )
    (root / "01_scan.out").write_text(SCAN_OUTPUT, encoding="utf-8")
    for step, distance in ((1, "3.0"), (2, "2.0")):
        (root / f"01_scan.{step:03d}.xyz").write_text(
            f"2\nstep {step}\nH 0 0 0\nH 0 0 {distance}\n", encoding="utf-8"
        )
    write_tier4_stage_record(
        root, "scan", "reaction-fingerprint", valid=True,
        validation="complete coordinate scan",
    )
    return root


def test_manifest_persists_stage_parameters_for_typed_resume(tmp_path):
    path = initialize_workflow_manifest(
        tmp_path, project="fixture", session_sha256="session",
        stages={"tier_4": {"full_thermo": True, "scan_steps": 18}},
    )
    stage = json.loads(path.read_text(encoding="utf-8"))["stages"]["tier_4"]
    assert stage["parameters"] == {"full_thermo": True, "scan_steps": 18}


def test_tier4_resume_reuses_valid_scan_and_regenerates_only_ts(tmp_path, monkeypatch):
    root = _prepared_tier4(tmp_path)
    executable = tmp_path / "orca"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["input_file"])
        assert kwargs["input_file"] == "02_optts.inp"
        (root / kwargs["output_file"]).write_text(VALID_TS_OUTPUT, encoding="utf-8")
        return ProcessResult(returncode=0, terminated_normally=True)

    monkeypatch.setattr("shark.workflows.tier4_resume.find_orca", lambda: executable)
    monkeypatch.setattr("shark.workflows.tier4_resume.run_orca_process", fake_run)

    states = resume_tier4_workflow(root, timeout=10)

    assert states == {"scan": "reused", "transition_state": "completed"}
    assert calls == ["02_optts.inp"]
    assert "H 0 0 2.0" in (root / "02_optts.inp").read_text(encoding="utf-8")
    reusable, _ = tier4_stage_reusable(
        root, "transition_state", "reaction-fingerprint",
        dependencies={"scan_output_sha256": json.loads(
            (root / "02_optts.meta.json").read_text(encoding="utf-8")
        )["dependencies"]["scan_output_sha256"]},
    )
    assert reusable is True

    (root / "01_scan.002.xyz").write_text("2\nchanged\nH 0 0 0\nH 0 0 1.9\n", encoding="utf-8")
    assert tier4_stage_reusable(root, "scan", "reaction-fingerprint")[0] is False
    (root / "01_scan.002.xyz").write_text(
        "2\nstep 2\nH 0 0 0\nH 0 0 2.0\n", encoding="utf-8"
    )

    calls.clear()
    assert resume_tier4_workflow(root, timeout=10) == {
        "scan": "reused", "transition_state": "reused"
    }
    assert calls == []


def test_tier4_resume_does_not_rerun_completed_invalid_ts(tmp_path, monkeypatch):
    root = _prepared_tier4(tmp_path)
    (root / "02_optts.inp").write_text(
        (root / "02_optts_template.inp").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "02_optts.out").write_text(INVALID_TS_OUTPUT, encoding="utf-8")
    scan_hash = json.loads((root / "01_scan.meta.json").read_text(encoding="utf-8"))["output_sha256"]
    write_tier4_stage_record(
        root, "transition_state", "reaction-fingerprint", valid=False,
        validation="Invalid transition_state result",
        dependencies={"scan_output_sha256": scan_hash},
    )
    executable = tmp_path / "orca"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("shark.workflows.tier4_resume.find_orca", lambda: executable)
    monkeypatch.setattr(
        "shark.workflows.tier4_resume.run_orca_process",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("invalid TS must not be rerun")),
    )

    try:
        resume_tier4_workflow(root, timeout=10)
    except ValueError as exc:
        assert "not a first-order saddle point" in str(exc)
    else:
        raise AssertionError("Invalid completed TS should require a changed setup")


def test_cli_resume_continues_recorded_md_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "workflow"
    run_dir = root / "md" / "run_ligand_10ns"
    prep_dir = run_dir / "00_prep"
    prep_dir.mkdir(parents=True)
    (run_dir / "config.env").write_text("SIM_TIME_NS=10\n", encoding="utf-8")
    (prep_dir / "md_prod.tpr").write_bytes(b"tpr")
    (prep_dir / "md_prod.cpt").write_bytes(b"checkpoint")
    manifest = initialize_workflow_manifest(
        root, project="fixture", session_sha256="session",
        stages={"molecular_dynamics": {"time_ns": 10}},
    )
    update_workflow_stage(
        root, "molecular_dynamics", status="failed", validation="failed",
        details={"run_dir": "md/run_ligand_10ns", "sim_time_ns": 10},
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gmx = bin_dir / "gmx"
    gmx.write_text(
        "#!/bin/sh\n"
        "echo 'Finished mdrun on rank 0' > md_prod.log\n"
        "echo structure > md_prod.gro\n"
        "echo trajectory > md_prod.xtc\n",
        encoding="utf-8",
    )
    gmx.chmod(0o755)
    monkeypatch.setenv("SHARK_GROMACS_BIN_DIRS", str(bin_dir))

    assert main(["resume", "--work-dir", str(root), "--timeout", "10"]) == 0
    stage = json.loads(manifest.read_text(encoding="utf-8"))["stages"]["molecular_dynamics"]
    assert stage["status"] == "completed"
    assert stage["validation"] == "validated"


def test_cli_resume_passes_recorded_tier4_options(tmp_path, monkeypatch):
    root = tmp_path / "workflow"
    initialize_workflow_manifest(
        root, project="fixture", session_sha256="session",
        stages={"tier_4": {"full_thermo": True, "optimize_adduct": True}},
    )
    captured = {}

    def fake_resume(directory, **kwargs):
        captured.update(directory=directory, **kwargs)
        return {"scan": "reused", "transition_state": "completed", "product": "completed"}

    monkeypatch.setattr("shark.workflows.tier4_resume.resume_tier4_workflow", fake_resume)

    assert main(["resume", "--work-dir", str(root), "--timeout", "25"]) == 0
    assert captured == {
        "directory": root / "transition_state",
        "timeout": 25.0,
        "full_thermo": True,
        "optimize_adduct": True,
    }
    manifest = json.loads((root / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["tier_4"]["validation"] == "validated"
