"""Unit tests for interactive CLI pickers and --pose filtering."""

import json
from pathlib import Path
import zipfile
import pytest

from shark.core.session import read_poliscreen_session
from shark.cli import (
    interactive_receptor_picker,
    interactive_compound_picker,
    interactive_pose_picker,
    interactive_nucleophile_picker,
    main as cli_main,
)


@pytest.fixture
def mock_session_with_ranking(tmp_path):
    archive = tmp_path / "test_session.poliscreen"
    rec_pdb = (
        "ATOM      1  N   THR A 309      10.000  10.000  10.000  1.00 20.00           N\n"
        "ATOM      2  OG1 THR A 309      10.500  10.500  10.500  1.00 20.00           O\n"
        "ATOM      3  N   CYS A 145      20.000  20.000  20.000  1.00 20.00           N\n"
        "ATOM      4  SG  CYS A 145      20.500  20.500  20.500  1.00 20.00           S\n"
        "END\n"
    )
    lig1_p1 = (
        "HETATM    1  C1  LIG A   1      11.000  11.000  11.000  1.00 20.00           C\n"
        "END\n"
    )
    lig1_p2 = (
        "HETATM    1  C1  LIG A   1      12.000  12.000  12.000  1.00 20.00           C\n"
        "END\n"
    )
    lig2_p1 = (
        "HETATM    1  C1  LIG A   2      13.000  13.000  13.000  1.00 20.00           C\n"
        "END\n"
    )

    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True, "project": "PickerTest"}))
        z.writestr("receptors/targetA_ready.pdb", rec_pdb)
        z.writestr("receptors/targetB_ready.pdb", rec_pdb)
        z.writestr("poses/docking_targetA_ready~Pk1_compounds_a_LIG1-model1.pdb", lig1_p1)
        z.writestr("poses/docking_targetA_ready~Pk1_compounds_a_LIG1-model2.pdb", lig1_p2)
        z.writestr("poses/docking_targetA_ready~Pk1_compounds_a_LIG2-model1.pdb", lig2_p1)
        z.writestr(
            "docking_results.csv",
            "receptor,pose_name,compound_name,docking_score,engine\n"
            "targetA_ready~Pk1,docking_targetA_ready~Pk1_compounds_a_LIG1-model1,LIG1,-8.5,vina\n"
            "targetA_ready~Pk1,docking_targetA_ready~Pk1_compounds_a_LIG1-model2,LIG1,-7.0,vina\n"
            "targetA_ready~Pk1,docking_targetA_ready~Pk1_compounds_a_LIG2-model1,LIG2,-6.2,vina\n"
        )
        z.writestr(
            "ranking.csv",
            "compound,receptor,best_dock,LE,effectiveness_pct\n"
            "LIG1,targetA_ready~Pk1,-8.5,0.45,95.2\n"
            "LIG2,targetA_ready~Pk1,-6.2,0.31,72.4\n"
        )

    return read_poliscreen_session(archive, tmp_path / "unpacked")


def test_interactive_receptor_picker(mock_session_with_ranking, monkeypatch):
    session = mock_session_with_ranking
    # Pick option 1 (targetA_ready~Pk1)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    picked = interactive_receptor_picker(session)
    assert picked == "targetA_ready~Pk1"


def test_interactive_compound_picker(mock_session_with_ranking, monkeypatch):
    session = mock_session_with_ranking
    # Pick Top 1 (LIG1) by entering blank (default)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    picked = interactive_compound_picker(session, receptor_id="targetA_ready~Pk1")
    assert picked == "LIG1"

    # Pick Top 2 (LIG2)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    picked2 = interactive_compound_picker(session, receptor_id="targetA_ready~Pk1")
    assert picked2 == "LIG2"


def test_interactive_pose_picker(mock_session_with_ranking, monkeypatch):
    session = mock_session_with_ranking
    # Default selection -> 1
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    pose = interactive_pose_picker(session, compound_id="LIG1", receptor_id="targetA_ready~Pk1")
    assert pose == 1

    # Pick pose 2
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    pose2 = interactive_pose_picker(session, compound_id="LIG1", receptor_id="targetA_ready~Pk1")
    assert pose2 == 2


def test_interactive_nucleophile_picker(mock_session_with_ranking, monkeypatch):
    session = mock_session_with_ranking
    # Default selection -> None (means All pocket nucleophiles)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    nucl = interactive_nucleophile_picker(session, compound_id="LIG1", receptor_id="targetA_ready~Pk1", pose_idx=1)
    assert nucl is None

    # Custom input
    inputs = iter(["C", "CYS145"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    nucl_custom = interactive_nucleophile_picker(session, compound_id="LIG1", receptor_id="targetA_ready~Pk1", pose_idx=1)
    assert nucl_custom == "CYS145"


def test_cli_pose_argument_filtering(tmp_path):
    archive = tmp_path / "cli_pose_session.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True, "project": "PoseFilterTest"}))
        z.writestr("receptors/target_ready.pdb", "ATOM      1  N   THR A 309      10.000  10.000  10.000  1.00 20.00           N\nEND\n")
        z.writestr("poses/docking_target_ready~Pk1_compounds_a_LIG1-model1.pdb", "ATOM      1  C1  LIG A   1      10.000  14.500  10.000  1.00 20.00           C\nEND\n")
        z.writestr("poses/docking_target_ready~Pk1_compounds_a_LIG1-model2.pdb", "ATOM      1  C1  LIG A   1      10.000  14.500  10.000  1.00 20.00           C\nEND\n")
        z.writestr(
            "docking_results.csv",
            "receptor,pose_name,compound_name,docking_score,engine\n"
            "target_ready~Pk1,docking_target_ready~Pk1_compounds_a_LIG1-model1,LIG1,-8.5,vina\n"
            "target_ready~Pk1,docking_target_ready~Pk1_compounds_a_LIG1-model2,LIG1,-7.0,vina\n"
        )

    out_html = tmp_path / "dossier_pose2.html"
    code = cli_main([
        "--session", str(archive),
        "--fast-analysis",
        "--compound", "LIG1",
        "--pose", "2",
        "--html", str(out_html),
    ])
    assert code == 0
    assert out_html.is_file()
    html_text = out_html.read_text(encoding="utf-8")
    assert "LIG1" in html_text

