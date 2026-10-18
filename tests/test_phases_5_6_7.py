"""Comprehensive verification suite for Phases 5, 6, and 7 of the SharK roadmap.

Covers:
- Phase 5: Repository hygiene, non-blocking stage continuation (--keep-going and
  non-reactive MD P_NAC = 0%), target/pocket-aware PoliScreen disaggregation, and
  6-layer formal epistemic separation with formal outcome states.
- Phase 6: Validated MD presets and preflight capabilities, pre-production MD
  resumption, stage timing history and calibrated duration estimates, and
  Windows-compatible process liveness and memory detection.
- Phase 7: Hirshfeld and Loewdin condensed Fukui functions with spin-contamination
  and ionic-state support, Delta-SCF redox thermodynamics, and Tier 4 IRC
  connectivity plus inter-stage homogeneity validation.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shark.analysis.reactivity import (
    build_reactivity_profile,
    validate_fukui_profile,
)
from shark.analysis.transition_state import (
    parse_orca_irc_output,
    validate_tier4_homogeneity,
)
from shark.cli import main as cli_main
from shark.core.analysis_result import (
    FORMAL_OUTCOME_STATES,
    SharKAnalysisResult,
)
from shark.core.parser import parse_orca_results
from shark.core.preflight import run_preflight
from shark.core.runner import ProcessResult, run_fukui_analysis
from shark.core.session import (
    read_poliscreen_session,
    split_receptor_pocket,
)
from shark.core.workflow_state import (
    _process_is_alive,
    estimate_stage_duration,
    initialize_workflow_manifest,
    update_workflow_stage,
)
from shark.workflows.ligand_qm import (
    _resources,
    prepare_ligand_jobs,
    select_ligands,
)
from shark.workflows.md_bridge import (
    MD_PRESETS,
    resolve_md_preset,
    resume_md_from_checkpoint,
)
from shark.workflows.redox_qm import (
    calculate_redox_thermodynamics,
    prepare_and_run_redox_workflow,
)


def _create_multi_pocket_session(
    tmp_path: Path,
    distant_ligand: bool = False,
    anionic: bool = False,
) -> Path:
    archive = tmp_path / ("multi_pocket_anionic.poliscreen" if anionic else "multi_pocket.poliscreen")
    lig_y = 28.000 if distant_ligand else 12.800
    smiles = "C(=O)[O-]" if anionic else "C=O"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": 1, "full": True, "project": "MultiPocketStudy"}),
        )
        zf.writestr(
            "ligands_meta.csv",
            f"name,smiles,role\nBENZ_1OX,{smiles},candidate\n",
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
            f"ATOM      1  C1  BEN A   1      10.000  {lig_y:6.3f}  10.000  1.00 20.00           C\n"
            f"ATOM      2  O1  BEN A   1      11.200  {lig_y:6.3f}  10.000  1.00 20.00           O\n"
            "END\n"
        )
        zf.writestr("poses/docking_6Q9N_ready_compounds_a_BENZ_1OX-model1.pdb", pose_pdb)
        zf.writestr("poses/docking_6Q9N_ready~Pk1_compounds_a_BENZ_1OX-model1.pdb", pose_pdb)
        zf.writestr("poses/docking_8HTB_ready~Pk1_compounds_a_BENZ_1OX-model1.pdb", pose_pdb)
        zf.writestr(
            "docking_results.csv",
            "receptor,pose_name,compound_name,docking_score,effectiveness,engine\n"
            "6Q9N_ready,docking_6Q9N_ready_compounds_a_BENZ_1OX-model1,BENZ_1OX,-7.10,72.0,vina\n"
            "6Q9N_ready~Pk1,docking_6Q9N_ready~Pk1_compounds_a_BENZ_1OX-model1,BENZ_1OX,-6.20,64.0,vina\n"
            "8HTB_ready~Pk1,docking_8HTB_ready~Pk1_compounds_a_BENZ_1OX-model1,BENZ_1OX,-6.80,70.0,vina\n",
        )
    return archive


# ---------------------------------------------------------------------------
# Phase 5 Tests
# ---------------------------------------------------------------------------


def test_phase5_repo_hygiene_internal_docs_untracked():
    repo_root = Path(__file__).resolve().parent.parent
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("CLAUDE.md", "COVALENT_MODEL_REFACTOR_SPEC.md", "next.md"):
        assert pattern in gitignore
    tracked = subprocess.run(
        ["git", "ls-files", "CLAUDE.md", "COVALENT_MODEL_REFACTOR_SPEC.md", "next.md"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert tracked == ""


def test_phase5_target_and_pocket_disaggregation(tmp_path):
    assert split_receptor_pocket("6Q9N_ready~Pk1") == ("6Q9N", "Pk1")
    assert split_receptor_pocket("6Q9N_ready") == ("6Q9N", "principal")
    assert split_receptor_pocket("8HTB~allosteric") == ("8HTB", "allosteric")

    archive = _create_multi_pocket_session(tmp_path)
    session = read_poliscreen_session(archive)
    try:
        rows = session.summarize_by_target_and_pocket()
        assert len(rows) == 3
        pockets = {(r["target_id"], r["pocket_id"]) for r in rows}
        assert pockets == {("6Q9N", "principal"), ("6Q9N", "Pk1"), ("8HTB", "Pk1")}
        scores = {f"{r['target_id']}~{r['pocket_id']}": r["best_score"] for r in rows}
        assert scores["6Q9N~principal"] == pytest.approx(-7.10)
        assert scores["6Q9N~Pk1"] == pytest.approx(-6.20)
        assert scores["8HTB~Pk1"] == pytest.approx(-6.80)

        selections = select_ligands(session, top=1)
        assert len(selections) == 1
        assert "pocket_breakdown" in selections[0]
        assert len(selections[0]["pocket_breakdown"]) == 3
    finally:
        import shutil
        shutil.rmtree(session.extract_dir, ignore_errors=True)


def test_phase5_keep_going_skips_unresolvable_tier4_and_completes_dft(tmp_path):
    archive = _create_multi_pocket_session(tmp_path, distant_ligand=True)
    work_dir = tmp_path / "job_keep_going"
    rc = cli_main(
        [
            "--session",
            str(archive),
            "--tier-4-ts",
            "--dft",
            "--geometry",
            "smiles",
            "--keep-going",
            "--work-dir",
            str(work_dir),
        ]
    )
    assert rc == 0
    manifest = json.loads((work_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["tier_4"]["status"] == "skipped"
    assert manifest["stages"]["tier_4"]["validation"] == "not_applicable"
    assert manifest["stages"]["ligand_dft"]["status"] == "prepared"
    assert (work_dir / "dossier.html").is_file()
    evidence = json.loads((work_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["workflow"]["degraded"] is False
    assert "tier_4" in evidence["workflow"]["recovery"]["reusable"]


def test_phase5_non_reactive_md_marks_tier4_not_applicable(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    archive = _create_multi_pocket_session(tmp_path, distant_ligand=False)

    u = mda.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("names", ["N", "OG1", "C1", "O1"])
    u.add_TopologyAttr("resnames", ["THR", "BEN"])
    u.add_TopologyAttr("resids", [309, 1])
    u.add_TopologyAttr("ids", [1, 2, 3, 4])
    u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]

    prot_coords = [[10.0, 10.0, 10.0], [10.0, 12.0, 10.0]]
    lig_far = [[10.0, 22.0, 10.0], [11.0, 22.0, 10.0]]

    top_file = tmp_path / "nonreactive.gro"
    xtc_file = tmp_path / "nonreactive.xtc"
    u.atoms.positions = prot_coords + lig_far
    u.atoms.write(str(top_file))
    with mda.Writer(str(xtc_file), n_atoms=4) as writer:
        for t in (0.0, 10.0, 20.0):
            u.trajectory.ts.time = t
            u.atoms.positions = prot_coords + lig_far
            writer.write(u)

    work_dir = tmp_path / "job_nonreactive_md"
    rc = cli_main(
        [
            "--session",
            str(archive),
            "--simple-gold-standard",
            "--topology",
            str(top_file),
            "--trajectory",
            str(xtc_file),
            "--target-residue",
            "THR309",
            "--cluster-stride",
            "1",
            "--tier-4-ts",
            "--work-dir",
            str(work_dir),
        ]
    )
    assert rc == 0
    manifest = json.loads((work_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["tier_4"]["status"] == "skipped"
    assert manifest["stages"]["tier_4"]["validation"] == "not_applicable"
    evidence = json.loads((work_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["evidence_layers"]["validated_outcome"]["formal_state"] == "NON_REACTIVE_ENSEMBLE"


def test_phase5_epistemic_layers_and_formal_outcome_states():
    res = SharKAnalysisResult(
        analysis_id="EpistemicCheck",
        workflow={"status": "completed", "degraded": False, "failures": []},
        binding={"docking_score": -6.8, "affinity_score": 0.63},
        dynamics={"p_nac": 0.0, "reactive_frame_count": 0},
        transition_state={"status": "skipped", "validation": "not_applicable"},
        feasibility={"cfi_pre": 0.0, "cfi_final": None, "status": "Non-reactive ensemble"},
    )
    layers = res.build_evidence_layers()
    assert set(layers.keys()) == {
        "experimental_measurement",
        "docking_score",
        "dft_descriptor",
        "md_score",
        "reaction_evidence",
        "validated_outcome",
    }
    state = layers["validated_outcome"]["formal_state"]
    assert state in FORMAL_OUTCOME_STATES
    assert state == "NON_REACTIVE_ENSEMBLE"

    res_valid = SharKAnalysisResult(
        analysis_id="ValidCheck",
        workflow={"status": "completed", "degraded": False},
        dynamics={"p_nac": 0.45, "reactive_frame_count": 12},
        transition_state={"status": "completed", "is_first_order_ts": True, "activation_barrier_kcal": 17.5},
        feasibility={"cfi_final": 0.71},
    )
    assert res_valid.compute_formal_outcome_state() == "COMPLETED_VALIDATED"


# ---------------------------------------------------------------------------
# Phase 6 Tests
# ---------------------------------------------------------------------------


def test_phase6_md_presets_and_preflight_validation(tmp_path):
    assert "amber99sb-ildn_spce" in MD_PRESETS
    preset = resolve_md_preset("amber99sb-ildn_spce")
    assert preset["protein_ff"] == "amber99sb-ildn"
    assert preset["water_model"] == "spce"
    with pytest.raises(ValueError, match="Unknown MD preset"):
        resolve_md_preset("nonexistent_forcefield")

    archive = _create_multi_pocket_session(tmp_path)
    session = read_poliscreen_session(archive)
    try:
        report_ok = run_preflight(
            session=session,
            execute=False,
            run_md=False,
            pipeline_dir=None,
            topology=None,
            trajectory=None,
            nprocs=None,
            maxcore_mb=None,
            require_orca=False,
            md_preset="amber14sb_tip3p",
        )
        assert not report_ok.blocking
        assert report_ok.md_capabilities["preset"] == "amber14sb_tip3p"
        assert "HEM" in report_ok.md_capabilities["supported_cofactors"]

        report_bad = run_preflight(
            session=session,
            execute=False,
            run_md=False,
            pipeline_dir=None,
            topology=None,
            trajectory=None,
            nprocs=None,
            maxcore_mb=None,
            require_orca=False,
            md_preset="invalid_preset_name",
        )
        assert any(c.code == "MD_PRESET_INVALID" for c in report_bad.blocking)
    finally:
        import shutil
        shutil.rmtree(session.extract_dir, ignore_errors=True)


def test_phase6_pre_production_md_checkpoint_resumption(tmp_path):
    run_dir = tmp_path / "run_preprod_10ns"
    prep_dir = run_dir / "00_prep"
    prep_dir.mkdir(parents=True)
    (run_dir / "config.env").write_text("SIM_TIME_NS=10\n", encoding="utf-8")
    runner = run_dir / "run_pipeline.sh"
    runner.write_text(
        "#!/bin/sh\n"
        "mkdir -p 00_prep\n"
        "echo 'Finished mdrun on rank 0' > 00_prep/md_prod.log\n"
        "echo 'gro' > 00_prep/md_prod.gro\n"
        "echo 'xtc' > 00_prep/md_prod.xtc\n"
        "echo 'cpt' > 00_prep/md_prod.cpt\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    res = resume_md_from_checkpoint(run_dir, timeout=10.0)
    assert res.status == "completed"
    assert res.artifact_hashes and "md_prod.xtc" in res.artifact_hashes


def test_phase6_stage_timing_history_and_calibrated_estimates(tmp_path):
    manifest_path = initialize_workflow_manifest(
        tmp_path,
        project="TimingProject",
        session_sha256="hash",
        stages={"ligand_dft": {"top": 1}},
    )
    manifest_init = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = estimate_stage_duration("ligand_dft", manifest=manifest_init)
    assert prior["source"] == "static_default"
    assert prior["sample_count"] == 0

    update_workflow_stage(tmp_path, "ligand_dft", status="running", validation="pending")
    update_workflow_stage(
        tmp_path,
        "ligand_dft",
        status="completed",
        validation="validated",
        details={"execution_time_s": 42.5},
    )
    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_rec = manifest_after["stages"]["ligand_dft"]
    assert stage_rec["started_utc"] is not None
    assert stage_rec["completed_utc"] is not None
    assert stage_rec["elapsed_seconds"] >= 0.0

    calibrated = estimate_stage_duration("ligand_dft", manifest=manifest_after)
    assert calibrated["source"] == "calibrated_history"
    assert calibrated["sample_count"] == 1


def test_phase6_windows_process_liveness_and_memory_detection(monkeypatch):
    import os as real_os

    class FakeKernel32:
        def OpenProcess(self, access, inherit, pid):
            return 0 if pid == 99999 else 1234

        def GetExitCodeProcess(self, handle, exit_code_ptr):
            exit_code_ptr._obj.value = 259  # STILL_ACTIVE
            return 1

        def CloseHandle(self, handle):
            return 1

        def GlobalMemoryStatusEx(self, mem_struct_ptr):
            mem_struct_ptr._obj.ullAvailPhys = 8 * 1024 * 1024 * 1024  # 8 GiB
            return 1

    monkeypatch.setattr(
        "shark.core.workflow_state.os",
        SimpleNamespace(name="nt", kill=real_os.kill),
    )
    monkeypatch.setattr(
        "ctypes.windll",
        SimpleNamespace(kernel32=FakeKernel32()),
        raising=False,
    )
    assert _process_is_alive(12345) is True
    assert _process_is_alive(99999) is False

    monkeypatch.setattr(
        "shark.workflows.ligand_qm.os",
        SimpleNamespace(name="nt", environ={}, cpu_count=lambda: 8),
    )
    import builtins
    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if str(file) == "/proc/meminfo":
            raise FileNotFoundError("/proc/meminfo")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert _resources(2, 2000) == (2, 2000)
    with pytest.raises(ValueError, match="exceeds 75% of available host memory"):
        _resources(2, 4000)


# ---------------------------------------------------------------------------
# Phase 7 Tests
# ---------------------------------------------------------------------------


ORCA_HIRS_LOEW_TEMPLATE = """
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  C      0.000000    0.000000    0.000000
  O      0.000000    0.000000    1.220000
  H      0.000000    0.940000   -0.540000

FINAL SINGLE POINT ENERGY      {energy:.8f}

----------------
ORBITAL ENERGIES
----------------

  NO   OCC          E(Eh)            E(eV)
   0   2.0000      -0.250000        -6.8028
   1   0.0000      -0.050000        -1.3606

LOEWDIN ATOMIC CHARGES
----------------------
   0 C :   {l_c:.6f}
   1 O :   {l_o:.6f}
   2 H :   {l_h:.6f}

HIRSHFELD ANALYSIS
------------------
  ATOM     CHARGE      SPIN
    0 C   {h_c:.6f}    0.000000
    1 O   {h_o:.6f}    0.000000
    2 H   {h_h:.6f}    0.000000

UHF SPIN CONTAMINATION
----------------------
Expectation value of <S**2>     :   {s2:.6f}
Ideal value S*(S+1)             :   {ideal_s2:.6f}

****ORCA TERMINATED NORMALLY****
"""


def test_phase7_hirshfeld_fukui_spin_contamination_and_ionic_ligand(tmp_path, monkeypatch):
    out_file = tmp_path / "sp.out"
    out_file.write_text(
        ORCA_HIRS_LOEW_TEMPLATE.format(
            energy=-113.80,
            l_c=0.10,
            l_o=-0.20,
            l_h=0.10,
            h_c=0.15,
            h_o=-0.25,
            h_h=0.10,
            s2=0.754,
            ideal_s2=0.750,
        ),
        encoding="utf-8",
    )
    parsed = parse_orca_results(out_file)
    assert len(parsed.hirshfeld_charges) == 3
    assert parsed.hirshfeld_charges[0] == pytest.approx(0.15)
    assert parsed.spin_contamination_s2 == pytest.approx(0.754)
    assert parsed.ideal_s2 == pytest.approx(0.750)

    neutral = parse_orca_results(out_file, name="neutral")
    anionic_file = tmp_path / "anionic.out"
    anionic_file.write_text(
        ORCA_HIRS_LOEW_TEMPLATE.format(
            energy=-113.85,
            l_c=-0.30,
            l_o=-0.60,
            l_h=-0.10,
            h_c=-0.25,
            h_o=-0.65,
            h_h=-0.10,
            s2=0.755,
            ideal_s2=0.750,
        ),
        encoding="utf-8",
    )
    cationic_file = tmp_path / "cationic.out"
    cationic_file.write_text(
        ORCA_HIRS_LOEW_TEMPLATE.format(
            energy=-113.45,
            l_c=0.50,
            l_o=0.20,
            l_h=0.30,
            h_c=0.55,
            h_o=0.15,
            h_h=0.30,
            s2=0.752,
            ideal_s2=0.750,
        ),
        encoding="utf-8",
    )
    anionic = parse_orca_results(anionic_file, name="anionic")
    cationic = parse_orca_results(cationic_file, name="cationic")

    profile_h = build_reactivity_profile(
        neutral,
        name="hirshfeld_test",
        charge_type="hirshfeld",
        anion_charges=anionic.hirshfeld_charges,
        cation_charges=cationic.hirshfeld_charges,
    )
    val_metrics = validate_fukui_profile(profile_h)
    assert val_metrics["sum_f_plus_valid"] is True
    assert val_metrics["sum_f_minus_valid"] is True
    assert val_metrics["sum_f_plus"] == pytest.approx(1.0, abs=1e-4)
    assert val_metrics["sum_f_minus"] == pytest.approx(1.0, abs=1e-4)

    # Test run_fukui_analysis on an anionic closed-shell ligand (formate C(=O)[O-], charge = -1, mult = 1)
    archive = _create_multi_pocket_session(tmp_path, anionic=True)
    session = read_poliscreen_session(archive)
    try:
        jobs = prepare_ligand_jobs(
            session,
            tmp_path / "qm_anionic",
            top=1,
            charge=-1,
            multiplicity=1,
            geometry="smiles",
            optimize=False,
            fukui=True,
        )
        job_dir = jobs[0]
        job_rec = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        job_rec["status"] = "completed"
        (job_dir / "job.json").write_text(json.dumps(job_rec, indent=2), encoding="utf-8")
        # Write completed reference output with charge = -1 (4 atoms: C, O, O, H)
        xyz_lines = (job_dir / "geometry.xyz").read_text(encoding="utf-8").splitlines()[2:]
        atoms_list = [ln.split()[0] for ln in xyz_lines if ln.strip()]
        assert len(atoms_list) == 4  # HCOO-

        def _make_4atom_out(energy: float, net_q: float, s2: float, ideal_s2: float) -> str:
            q_each = net_q / 4.0
            coords = "\n".join(f"  {ln.strip()}" for ln in xyz_lines if ln.strip())
            loew = "\n".join(f"   {i} {el} :   {q_each:.6f}" for i, el in enumerate(atoms_list))
            hirs = "\n".join(f"    {i} {el}   {q_each:.6f}    0.000000" for i, el in enumerate(atoms_list))
            return (
                f"CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n{coords}\n\n"
                f"FINAL SINGLE POINT ENERGY      {energy:.8f}\n\n"
                f"----------------\nORBITAL ENERGIES\n----------------\n\n"
                f"  NO   OCC          E(Eh)            E(eV)\n"
                f"   0   2.0000      -0.250000        -6.8028\n"
                f"   1   0.0000      -0.050000        -1.3606\n\n"
                f"LOEWDIN ATOMIC CHARGES\n----------------------\n{loew}\n\n"
                f"HIRSHFELD ANALYSIS\n------------------\n  ATOM     CHARGE      SPIN\n{hirs}\n\n"
                f"UHF SPIN CONTAMINATION\n----------------------\n"
                f"Expectation value of <S**2>     :   {s2:.6f}\n"
                f"Ideal value S*(S+1)             :   {ideal_s2:.6f}\n\n"
                f"****ORCA TERMINATED NORMALLY****\n"
            )

        (job_dir / "orca.out").write_text(_make_4atom_out(-189.10, -1.0, 0.0, 0.0), encoding="utf-8")
        orca_bin = tmp_path / "fake_orca"
        orca_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        orca_bin.chmod(0o755)
        monkeypatch.setattr("shark.core.runner.find_orca", lambda allow_none=False: orca_bin)

        def fake_orca_proc(**kwargs):
            inp_name = Path(kwargs["input_file"]).name
            out_path = Path(kwargs["output_file"])
            if not out_path.is_absolute():
                out_path = Path(kwargs["cwd"]) / out_path
            if inp_name == "n.inp":
                out_path.write_text(_make_4atom_out(-189.10, -1.0, 0.0, 0.0), encoding="utf-8")
            elif inp_name == "n_plus_1.inp":
                # N+1 for Q=-1 has net charge -2
                out_path.write_text(_make_4atom_out(-188.95, -2.0, 0.76, 0.75), encoding="utf-8")
            else:
                # N-1 for Q=-1 has net charge 0
                out_path.write_text(_make_4atom_out(-189.02, 0.0, 0.76, 0.75), encoding="utf-8")
            return ProcessResult(returncode=0, terminated_normally=True, execution_time_s=1.2)

        monkeypatch.setattr("shark.core.runner.run_orca_process", fake_orca_proc)
        fukui_res = run_fukui_analysis(job_dir, timeout=10.0, charge_scheme="hirshfeld")
        assert fukui_res["status"] == "completed"
        assert fukui_res["charge_scheme"] == "hirshfeld"
        assert fukui_res["validation_metrics"]["sum_f_plus_valid"] is True
    finally:
        import shutil
        shutil.rmtree(session.extract_dir, ignore_errors=True)


def test_phase7_delta_scf_redox_workflow(tmp_path):
    thermo = calculate_redox_thermodynamics(
        e_ox_hartree=-500.0000,
        e_red_hartree=-500.0950,
        g_ox_hartree=-499.9200,
        g_red_hartree=-500.0120,
        e_red_vertical_hartree=-500.0850,
        e_ox_vertical_hartree=-499.7200,
    )
    assert thermo["vertical_electron_affinity_ev"] == pytest.approx(0.0850 * 27.211386245988, rel=1e-4)
    assert thermo["vertical_ionization_potential_ev"] == pytest.approx(0.2800 * 27.211386245988, rel=1e-4)
    assert thermo["adiabatic_electron_affinity_ev"] == pytest.approx(0.0950 * 27.211386245988, rel=1e-4)
    assert thermo["reorganization_energy_ev"] == pytest.approx(0.0100 * 27.211386245988, rel=1e-4)
    assert thermo["e_red_vs_she_v"] is not None
    assert thermo["e_red_vs_sce_v"] == pytest.approx(
        thermo["e_red_vs_she_v"] - 0.241, abs=1e-3
    )

    archive = _create_multi_pocket_session(tmp_path)
    session = read_poliscreen_session(archive)
    try:
        jobs = prepare_ligand_jobs(session, tmp_path / "qm_redox", top=1, geometry="smiles")
        job_dir = jobs[0]
        prep_only = prepare_and_run_redox_workflow(job_dir, adiabatic=True, execute=False)
        assert prep_only["status"] == "prepared"
        assert (job_dir / "redox" / "ox_sp.inp").is_file()
        assert (job_dir / "redox" / "red_vert.inp").is_file()
        assert (job_dir / "redox" / "ox_cat_vert.inp").is_file()
        assert (job_dir / "redox" / "red_opt.inp").is_file()
    finally:
        import shutil
        shutil.rmtree(session.extract_dir, ignore_errors=True)


def test_phase7_tier4_irc_and_inter_stage_homogeneity(tmp_path):
    ts_dir = tmp_path / "ts_irc"
    ts_dir.mkdir()
    (ts_dir / "reaction_definition.json").write_text(
        json.dumps({"fingerprint": "fp-irc", "validation_status": "INFO"}),
        encoding="utf-8",
    )
    common_header = "! r2SCAN-3c TightSCF CPCM(Water)\n%pal nprocs 4 end\n%maxcore 2000\n"
    coords_block = "* xyz 0 1\n  O 0.0 0.0 0.0\n  C 0.0 0.0 2.0\n*\n"
    (ts_dir / "01_scan.inp").write_text(
        common_header + "%geom\n Scan\n  B 0 1 = 3.00, 1.50, 2\n end\nend\n" + coords_block,
        encoding="utf-8",
    )
    (ts_dir / "02_optts_template.inp").write_text(
        common_header + "! OptTS Freq\n" + coords_block,
        encoding="utf-8",
    )
    (ts_dir / "02_optts.inp").write_text(
        common_header + "! OptTS Freq\n" + coords_block,
        encoding="utf-8",
    )
    (ts_dir / "02b_irc_template.inp").write_text(
        common_header + "! IRC\n%irc\n MaxIter 40\n Direction both\nend\n" + coords_block,
        encoding="utf-8",
    )
    (ts_dir / "03_adduct_opt_template.inp").write_text(
        common_header + "! Opt\n" + coords_block,
        encoding="utf-8",
    )

    homog = validate_tier4_homogeneity(ts_dir)
    assert homog["homogeneous"] is True

    # Introduce a solvent mismatch and verify homogeneity rejection
    (ts_dir / "03_adduct_opt.inp").write_text(
        "! r2SCAN-3c TightSCF\n* xyz 0 1\n  O 0.0 0.0 0.0\n  C 0.0 0.0 1.5\n*\n",
        encoding="utf-8",
    )
    homog_bad = validate_tier4_homogeneity(ts_dir)
    assert homog_bad["homogeneous"] is False
    assert any("Theory/solvent keyword mismatch" in m for m in homog_bad["mismatches"])
    (ts_dir / "03_adduct_opt.inp").unlink()

    # Test IRC parser
    irc_out = ts_dir / "02b_irc.out"
    irc_out.write_text(
        "FORWARD IRC\n"
        "THE OPTIMIZATION HAS CONVERGED\n"
        "BACKWARD IRC\n"
        "THE OPTIMIZATION HAS CONVERGED\n"
        "IRC PATH SUMMARY\n"
        "----------------\n"
        "  -5   -100.012000   -7.53\n"
        "   0   -100.000000    0.00 <= TS\n"
        "   5   -100.025000  -15.68\n\n\n"
        "ORCA TERMINATED NORMALLY\n",
        encoding="utf-8",
    )
    irc_verif = parse_orca_irc_output(irc_out)
    assert irc_verif.connects_reactant_and_product is True
    assert irc_verif.forward_completed is True
    assert irc_verif.backward_completed is True
    assert irc_verif.delta_e_forward_kcal == pytest.approx(-15.68)
    assert irc_verif.delta_e_backward_kcal == pytest.approx(-7.53)
