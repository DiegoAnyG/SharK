"""Unit tests covering audit remediations for Priorities 9, 10, 12, 14, and 15."""

import importlib.resources
import os
from pathlib import Path
import stat
import sys
import tempfile
import pytest

from shark.analysis.qm_cluster import (
    QMCluster,
    ClusterAtom,
    PERIODIC_TABLE_Z,
    get_atomic_number,
)
from shark.core.runner import run_orca_process, ProcessResult, find_orca
from shark.analysis.reactivity import build_reactivity_profile, calculate_condensed_fukui
from shark.workflows.md_bridge import setup_and_launch_md


class TestPriority9MultiplicityAndElements:
    """Tests for QM-cluster element mapping and multiplicity parity validation."""

    def test_periodic_table_z_standard_and_metals(self):
        # Biological and transition metals
        assert get_atomic_number("H") == 1
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("Cl") == 17
        assert get_atomic_number("Fe") == 26
        assert get_atomic_number("Zn") == 30
        assert get_atomic_number("Cu") == 29
        assert get_atomic_number("Br") == 35
        assert get_atomic_number("I") == 53

    def test_unknown_element_raises_never_guesses_carbon(self):
        with pytest.raises(ValueError, match="Unknown element symbol: 'Xx'"):
            get_atomic_number("Xx")

        with pytest.raises(ValueError, match="Unknown element symbol: 'Fake'"):
            get_atomic_number("Fake")

        # In cluster context
        cluster = QMCluster(
            name="UnknownElemCluster",
            atoms=[ClusterAtom(element="Xx", coords=(0.0, 0.0, 0.0))],
            charge=0,
            multiplicity=1,
        )
        with pytest.raises(ValueError, match="Unknown element symbol: 'Xx'"):
            _ = cluster.effective_multiplicity

    def test_closed_shell_singlet(self):
        # Methane CH4: C(6) + 4*H(1) = 10 electrons (even), charge 0, mult 1 -> valid
        cluster = QMCluster(
            name="Methane",
            atoms=[
                ClusterAtom(element="C", coords=(0.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, 1.0)),
                ClusterAtom(element="H", coords=(0.0, 1.0, 0.0)),
                ClusterAtom(element="H", coords=(1.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, -1.0)),
            ],
            charge=0,
            multiplicity=1,
        )
        assert cluster.total_electrons == 10
        assert cluster.effective_multiplicity == 1

    def test_chlorine_containing_cluster(self):
        # Chloromethane CH3Cl: C(6) + 3*H(1) + Cl(17) = 26 electrons (even), charge 0, mult 1 -> valid
        cluster = QMCluster(
            name="Chloromethane",
            atoms=[
                ClusterAtom(element="C", coords=(0.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, 1.0)),
                ClusterAtom(element="H", coords=(0.0, 1.0, 0.0)),
                ClusterAtom(element="H", coords=(1.0, 0.0, 0.0)),
                ClusterAtom(element="Cl", coords=(0.0, 0.0, -1.7)),
            ],
            charge=0,
            multiplicity=1,
        )
        assert cluster.total_electrons == 26
        assert cluster.effective_multiplicity == 1

    def test_zinc_containing_cluster(self):
        # Zn2+ coordinated by 2 water molecules: Zn(30) + 2*(O(8) + 2*H(1)) = 50 electrons
        # With charge +2: 50 - 2 = 48 electrons (even), mult 1 -> valid
        cluster = QMCluster(
            name="ZnCluster",
            atoms=[
                ClusterAtom(element="Zn", coords=(0.0, 0.0, 0.0)),
                ClusterAtom(element="O", coords=(0.0, 0.0, 2.0)),
                ClusterAtom(element="H", coords=(0.0, 0.8, 2.5)),
                ClusterAtom(element="H", coords=(0.0, -0.8, 2.5)),
                ClusterAtom(element="O", coords=(0.0, 0.0, -2.0)),
                ClusterAtom(element="H", coords=(0.0, 0.8, -2.5)),
                ClusterAtom(element="H", coords=(0.0, -0.8, -2.5)),
            ],
            charge=2,
            multiplicity=1,
        )
        assert cluster.total_electrons == 48
        assert cluster.effective_multiplicity == 1

    def test_iron_containing_cluster(self):
        # Fe(III) complex with odd electron count, e.g. Fe(26) - 3 = 23 electrons, mult 2 or 6
        cluster_doublet = QMCluster(
            name="FeClusterDoublet",
            atoms=[ClusterAtom(element="Fe", coords=(0.0, 0.0, 0.0))],
            charge=3,
            multiplicity=2,
        )
        assert cluster_doublet.total_electrons == 23
        assert cluster_doublet.effective_multiplicity == 2

        cluster_sextet = QMCluster(
            name="FeClusterSextet",
            atoms=[ClusterAtom(element="Fe", coords=(0.0, 0.0, 0.0))],
            charge=3,
            multiplicity=6,
        )
        assert cluster_sextet.total_electrons == 23
        assert cluster_sextet.effective_multiplicity == 6

    def test_odd_electron_incompatible_singlet_raises(self):
        # Methyl radical CH3: C(6) + 3*H(1) = 9 electrons (odd), requested mult 1 (singlet)
        cluster = QMCluster(
            name="MethylRadical",
            atoms=[
                ClusterAtom(element="C", coords=(0.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, 1.0)),
                ClusterAtom(element="H", coords=(0.0, 1.0, 0.0)),
                ClusterAtom(element="H", coords=(1.0, 0.0, 0.0)),
            ],
            charge=0,
            multiplicity=1,
        )
        assert cluster.total_electrons == 9
        with pytest.raises(ValueError, match="Spin multiplicity 1 is incompatible with 9 electrons"):
            _ = cluster.effective_multiplicity

    def test_even_electron_incompatible_doublet_raises(self):
        # Methane CH4: 10 electrons (even), requested mult 2 (doublet)
        cluster = QMCluster(
            name="MethaneDoublet",
            atoms=[
                ClusterAtom(element="C", coords=(0.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, 1.0)),
                ClusterAtom(element="H", coords=(0.0, 1.0, 0.0)),
                ClusterAtom(element="H", coords=(1.0, 0.0, 0.0)),
                ClusterAtom(element="H", coords=(0.0, 0.0, -1.0)),
            ],
            charge=0,
            multiplicity=2,
        )
        assert cluster.total_electrons == 10
        with pytest.raises(ValueError, match="Spin multiplicity 2 is incompatible with 10 electrons"):
            _ = cluster.effective_multiplicity


class TestPriority10CentralizedProcessManagement:
    """Tests for run_orca_process helper function in core/runner.py."""

    @pytest.fixture
    def mock_script(self, tmp_path):
        """Creates a mock executable bash script to simulate ORCA behaviors."""
        def make_script(name, body):
            p = tmp_path / name
            p.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            return p
        return make_script

    def test_run_orca_process_success(self, tmp_path, mock_script):
        mock_orca = mock_script("orca_success", """
echo "Starting ORCA 6 calculation..."
echo "FINAL SINGLE POINT ENERGY   -40.123456"
echo "****ORCA TERMINATED NORMALLY****"
exit 0
""")
        out_file = tmp_path / "calc.out"
        inp_file = tmp_path / "calc.inp"
        inp_file.write_text("! r2SCAN-3c\n", encoding="utf-8")

        res = run_orca_process(
            executable=mock_orca,
            input_file=inp_file,
            output_file=out_file,
            cwd=tmp_path,
            check_normal_termination=True,
        )
        assert isinstance(res, ProcessResult)
        assert res.returncode == 0
        assert res.terminated_normally is True
        assert res.timed_out is False
        assert res.interrupted is False
        assert out_file.is_file()
        assert "ORCA TERMINATED NORMALLY" in out_file.read_text(encoding="utf-8")

    def test_run_orca_process_abnormal_termination(self, tmp_path, mock_script):
        mock_orca = mock_script("orca_abnormal", """
echo "Error in SCF iteration"
exit 0
""")
        out_file = tmp_path / "calc.out"
        res = run_orca_process(
            executable=mock_orca,
            input_file="dummy.inp",
            output_file=out_file,
            cwd=tmp_path,
            check_normal_termination=True,
        )
        assert res.returncode == 0
        assert res.terminated_normally is False
        assert "missing 'ORCA TERMINATED NORMALLY'" in (res.error or "")

    def test_run_orca_process_nonzero_exit(self, tmp_path, mock_script):
        mock_orca = mock_script("orca_fail", """
echo "Fatal input error"
exit 2
""")
        out_file = tmp_path / "calc.out"
        res = run_orca_process(
            executable=mock_orca,
            input_file="dummy.inp",
            output_file=out_file,
            cwd=tmp_path,
            check_normal_termination=True,
        )
        assert res.returncode == 2
        assert res.terminated_normally is False
        assert res.error is not None

    def test_run_orca_process_timeout(self, tmp_path, mock_script):
        mock_orca = mock_script("orca_sleep", """
sleep 10
echo "****ORCA TERMINATED NORMALLY****"
exit 0
""")
        out_file = tmp_path / "calc.out"
        res = run_orca_process(
            executable=mock_orca,
            input_file="dummy.inp",
            output_file=out_file,
            cwd=tmp_path,
            timeout=0.5,
            check_normal_termination=True,
        )
        assert res.timed_out is True
        assert res.terminated_normally is False
        assert "timed out" in (res.error or "").lower()


class TestPriority12PackageData:
    """Verify package resources for offline 3Dmol viewer."""

    def test_3dmol_resource_exists(self):
        from shark.reports import adduct_viewer
        js_content = adduct_viewer._get_3dmol_js()
        assert len(js_content) > 1000
        assert "3Dmol" in js_content


class TestPriority14ReactivityProvenance:
    """Verify method provenance and warnings in conceptual DFT reactivity."""

    def test_fukui_proxy_provenance_and_warning(self):
        data = {
            "name": "Acrylamide",
            "homo_ev": -6.5,
            "lumo_ev": -1.2,
            "loewdin_charges": [0.25, -0.30, 0.15],
            "atomic_symbols": ["C", "O", "C"],
        }
        profile = build_reactivity_profile(data, charge_type="loewdin")
        assert profile.fukui_method == "partial_charge_proxy"
        assert profile.charge_scheme == "loewdin"
        assert len(profile.warnings) > 0
        assert any("approximate partial-charge proxies" in w for w in profile.warnings)
        assert all(a.fukui_method == "partial_charge_proxy" for a in profile.atoms)

    def test_fukui_finite_difference_provenance(self):
        data = {
            "name": "Acrylamide",
            "homo_ev": -6.5,
            "lumo_ev": -1.2,
            "loewdin_charges": [0.25, -0.30, 0.15],
            "atomic_symbols": ["C", "O", "C"],
        }
        profile = build_reactivity_profile(
            data,
            anion_charges=[-0.10, -0.60, -0.10],
            cation_charges=[0.50, -0.10, 0.40],
        )
        assert profile.fukui_method == "finite_difference"
        assert len(profile.warnings) == 0
        assert all(a.fukui_method == "finite_difference" for a in profile.atoms)


class TestPriority15MDSDFSemantics:
    """Verify that MD bridge preserves true SDF bond orders and never passes a PDB as SDF."""

    def test_md_bridge_uses_sdf_when_available(self, tmp_path):
        pipe_root = tmp_path / "pipeline"
        (pipe_root / "scripts").mkdir(parents=True)
        (pipe_root / "mdp_templates").mkdir(parents=True)
        (pipe_root / "inputs").mkdir(parents=True)
        (pipe_root / "scripts" / "run_pipeline.sh").write_text("#!/bin/bash\nexit 0\n")

        rec_pdb = tmp_path / "rec.pdb"
        rec_pdb.write_text("ATOM      1  N   ALA A   1      10.000  10.000  10.000\nEND\n")
        lig_pdb = tmp_path / "lig.pdb"
        lig_pdb.write_text("HETATM    1  C1  LIG A   1      12.000  10.000  10.000\nEND\n")
        lig_sdf = tmp_path / "lig.sdf"
        lig_sdf.write_text("Dummy SDF content with bond orders\n$$$$\n")

        res = setup_and_launch_md(
            pipeline_dir=pipe_root,
            receptor_pdb=rec_pdb,
            ligand_pose_file=lig_pdb,
            ligand_ref_file=lig_sdf,
            ligand_name="LIG1",
            sim_time_ns=1.0,
            run_now=False,
        )
        cfg_file = res.run_dir / "config.env"
        assert cfg_file.is_file()
        text = cfg_file.read_text()
        assert 'POLISCREEN_LIGAND_SDF="inputs/run_LIG1_1ns_ref.sdf"' in text

