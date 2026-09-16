"""Unit tests for raw receptor selection from PoliScreen sessions."""

import json
from pathlib import Path
import zipfile
import pytest

from shark.core.session import read_poliscreen_session


def test_raw_receptor_retrieval(tmp_path):
    archive = tmp_path / "test.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True}))
        # Receptors: both raw 8HTB.pdb and docking ready 8HTB_ready.pdb
        z.writestr("receptors/8HTB.pdb", "REMARK Raw PDB\nATOM      1  N   MET A   1      20.154  12.450  34.120  1.00 20.00           N\nEND\n")
        z.writestr("receptors/8HTB_ready.pdb", "REMARK Vina Ready PDB\nATOM      1  N   MET A   1      20.154  12.450  34.120  1.00 20.00           N\nEND\n")
        z.writestr("poses/docking_8HTB_ready~Pk1_compounds_a_LIG1-model1.pdb", "ATOM      1  C1  LIG A   1      21.000  13.000  35.000  1.00 20.00           C\nEND\n")
        z.writestr("docking_results.csv", "receptor,pose_name,compound_name,docking_score,engine\n8HTB_ready~Pk1,docking_8HTB_ready~Pk1_compounds_a_LIG1-model1,LIG1,-9.2,vina\n")

    session = read_poliscreen_session(archive, tmp_path / "extracted")
    
    # Ready receptor for docking target
    ready_rec = session.receptor_for("8HTB_ready~Pk1", raw=False)
    assert ready_rec.name == "8HTB_ready.pdb"

    # Raw receptor for MD simulation / pdb2gmx
    raw_rec = session.receptor_for("8HTB_ready~Pk1", raw=True)
    assert raw_rec.name == "8HTB.pdb"
    assert "Raw PDB" in raw_rec.read_text()


def test_raw_receptor_fallback_when_no_ready(tmp_path):
    archive = tmp_path / "test2.poliscreen"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "full": True}))
        z.writestr("receptors/3ACX.pdb", "REMARK Raw Only\nEND\n")
        z.writestr("poses/docking_3ACX~Pk1_compounds_a_LIG1-model1.pdb", "END\n")
        z.writestr("docking_results.csv", "receptor,pose_name,compound_name,docking_score,engine\n3ACX~Pk1,docking_3ACX~Pk1_compounds_a_LIG1-model1,LIG1,-8.1,vina\n")

    session = read_poliscreen_session(archive, tmp_path / "extracted2")
    
    rec_raw = session.receptor_for("3ACX~Pk1", raw=True)
    assert rec_raw.name == "3ACX.pdb"
