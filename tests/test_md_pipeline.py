"""Unit tests for GROMACS MD pipeline orchestrator bridge."""

import json
import os
from pathlib import Path
import zipfile
import pytest

from shark.workflows.md_pipeline import (
    find_gromacs_pipeline,
    setup_and_launch_md,
    run_md_from_session,
    MDRunResult,
)
from shark.core.session import read_poliscreen_session


@pytest.fixture
def mock_pipeline_dir(tmp_path):
    """Creates a mock GROMACS pipeline layout."""
    pipe = tmp_path / "gromacs_pipeline"
    (pipe / "scripts").mkdir(parents=True)
    (pipe / "inputs").mkdir(parents=True)
    (pipe / "mdp_templates").mkdir(parents=True)
    (pipe / "runs").mkdir(parents=True)
    (pipe / "scripts" / "02_prepare_receptor.py").write_text("# mock prepare", encoding="utf-8")
    (pipe / "scripts" / "run_pipeline.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    return pipe


@pytest.fixture
def mock_session_with_receptors(tmp_path):
    """Creates a PoliScreen session containing both raw and ready receptors."""
    archive = tmp_path / "screening.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True}))
        z.writestr("receptors/8HTB.pdb", "HEADER    RAW RECEPTOR PDB FOR MD\nATOM      1  N   MET A   1      10.0  10.0  10.0  1.00 20.00           N\nEND\n")
        z.writestr("receptors/8HTB_ready.pdb", "HEADER    DOCKING READY PDB\nATOM      1  N   MET A   1      10.0  10.0  10.0  1.00 20.00           N\nEND\n")
        z.writestr("poses/docking_8HTB_ready~Pk1_compounds_a_LIG1-model1.pdb", "ATOM      1  C1  LIG A   1      12.0  12.0  12.0  1.00 20.00           C\nEND\n")
        z.writestr("docking_results.csv", "receptor,pose_name,compound_name,docking_score,engine\n8HTB_ready~Pk1,docking_8HTB_ready~Pk1_compounds_a_LIG1-model1,LIG1,-8.5,vina\n")
    return archive


def test_find_gromacs_pipeline(mock_pipeline_dir, monkeypatch):
    # Test explicit directory argument
    found = find_gromacs_pipeline(mock_pipeline_dir)
    assert found == mock_pipeline_dir.resolve()

    # Test via SHARK_GROMACS_PIPELINE env var
    monkeypatch.setenv("SHARK_GROMACS_PIPELINE", str(mock_pipeline_dir))
    found_env = find_gromacs_pipeline()
    assert found_env == mock_pipeline_dir.resolve()


def test_setup_and_launch_md_prepared(mock_pipeline_dir, tmp_path):
    rec_pdb = tmp_path / "receptor_raw.pdb"
    rec_pdb.write_text("ATOM      1  N   MET A   1      10.0  10.0  10.0\nEND\n", encoding="utf-8")
    lig_pdb = tmp_path / "ligand_pose.pdb"
    lig_pdb.write_text("ATOM      1  C1  LIG A   1      12.0  12.0  12.0\nEND\n", encoding="utf-8")

    result = setup_and_launch_md(
        receptor_pdb=rec_pdb,
        ligand_pose_file=lig_pdb,
        ligand_name="TEST_LIG",
        sim_time_ns=5.0,
        pipeline_dir=mock_pipeline_dir,
        run_now=False
    )

    assert isinstance(result, MDRunResult)
    assert result.status == "prepared"
    assert result.sim_time_ns == 5.0
    assert result.run_dir.is_dir()
    assert (result.run_dir / "00_prep" / "receptor_raw.pdb").is_file()
    assert (result.run_dir / "00_prep" / "TEST_LIG_pose.pdb").is_file()
    
    cfg = (result.run_dir / "config.env").read_text()
    assert "SIM_TIME_NS=5.0" in cfg
    assert 'LIGAND_NAME="TEST_LIG"' in cfg


def test_run_md_from_session_selects_raw_receptor(mock_pipeline_dir, mock_session_with_receptors, tmp_path):
    session = read_poliscreen_session(mock_session_with_receptors, tmp_path / "session_unpacked")

    result = run_md_from_session(
        session=session,
        ligand_id="LIG1",
        pose_idx=1,
        sim_time_ns=2.0,
        pipeline_dir=mock_pipeline_dir,
        run_now=False
    )

    assert result.status == "prepared"
    # Verify that the prepared receptor is the RAW one (8HTB.pdb), not the _ready one!
    prep_rec = result.run_dir / "00_prep" / "receptor_raw.pdb"
    assert prep_rec.is_file()
    content = prep_rec.read_text()
    assert "RAW RECEPTOR PDB FOR MD" in content
    assert "DOCKING READY PDB" not in content

    cfg_text = (result.run_dir / "config.env").read_text()
    assert 'PROTEIN_FF="amber99sb-ildn"' in cfg_text
    assert 'WATER_MODEL="spce"' in cfg_text
    assert (mock_pipeline_dir / "inputs" / "run_LIG1_2ns_receptor.pdb").is_file()
    assert (mock_pipeline_dir / "config" / "config.env").is_file()

