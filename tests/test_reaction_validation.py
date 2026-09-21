"""Tests for Tier 4 reaction pair validation and reliability spec (SHARK_TIER4_ORCA_RELIABILITY_SPEC.md)."""

import json
from pathlib import Path
import pytest

from shark.analysis.reaction_validation import (
    validate_reaction_pair,
    compute_tier4_fingerprint,
    write_reaction_definition_json,
    KNOWN_MECHANISMS,
)


def test_validate_reaction_pair_pass():
    res = validate_reaction_pair(
        nucleophile_element="O",
        electrophile_element="C",
        mechanism_name="carbonyl_addition",
        nucleophile_label="THR309:OG1",
        electrophile_label="LIG:C7",
        initial_distance_angstrom=3.18,
    )
    assert res.valid is True
    assert res.severity in ("INFO", "WARNING")
    assert res.proposed_bond_type == "O-C"


def test_validate_reaction_pair_blocking_oo_on_carbonyl():
    """PAIR-3: Declared carbonyl-addition mechanism with proposed O-O must produce BLOCKING error."""
    res = validate_reaction_pair(
        nucleophile_element="O",
        electrophile_element="O",
        mechanism_name="carbonyl_addition",
        nucleophile_label="THR309:OG1",
        electrophile_label="LIG:O1",
        initial_distance_angstrom=3.20,
    )
    assert res.valid is False
    assert res.severity == "BLOCKING"
    assert "incompatible" in res.message.lower()


def test_validate_reaction_pair_clash():
    res = validate_reaction_pair(
        nucleophile_element="O",
        electrophile_element="C",
        mechanism_name="carbonyl_addition",
        initial_distance_angstrom=0.85,
    )
    assert res.valid is False
    assert res.severity == "BLOCKING"


def test_validate_reaction_pair_blocks_distant_starting_geometry():
    res = validate_reaction_pair(
        nucleophile_element="O",
        electrophile_element="C",
        mechanism_name="generic_covalent_addition",
        initial_distance_angstrom=6.43,
    )
    assert res.valid is False
    assert res.severity == "BLOCKING"
    assert "reactive-contact region" in res.message


def test_fingerprint_changes_with_chemistry():
    class DummyAtom:
        def __init__(self, element):
            self.element = element

    atoms = [DummyAtom("C"), DummyAtom("O"), DummyAtom("N")]
    fp1 = compute_tier4_fingerprint(atoms, 0, 1, 0, 1, "r2scan-3c", "water", 3.2, 1.45, 18)
    fp2 = compute_tier4_fingerprint(atoms, 0, 1, 0, 1, "r2scan-3c", "water", 3.2, 1.45, 18)
    assert fp1 == fp2

    # Change electrophile
    fp3 = compute_tier4_fingerprint(atoms, 0, 2, 0, 1, "r2scan-3c", "water", 3.2, 1.45, 18)
    assert fp1 != fp3

    # Change charge
    fp4 = compute_tier4_fingerprint(atoms, 0, 1, -1, 1, "r2scan-3c", "water", 3.2, 1.45, 18)
    assert fp1 != fp4


def test_write_reaction_definition(tmp_path):
    out_f = tmp_path / "reaction_definition.json"
    p = write_reaction_definition_json(
        output_path=out_f,
        target_residue="THR309",
        nucleophile_label="THR309:OG1",
        nucleophile_element="O",
        nucleophile_idx=21,
        electrophile_label="LIG:N1",
        electrophile_element="N",
        electrophile_idx=11,
        mechanism="furoxan_heterocycle_attack",
        assignment_source="explicit",
        validation_status="passed",
        initial_distance_angstrom=3.45,
        proposed_bond="O-N",
        fingerprint="dummy_fp",
    )
    assert p.is_file()
    data = json.loads(p.read_text())
    assert data["target_residue"] == "THR309"
    assert data["nucleophile"]["element"] == "O"
    assert data["electrophile"]["element"] == "N"
    assert data["proposed_bond"] == "O-N"


def test_safe_maxcore_available_ram():
    """MEM-1: Available memory governs safe maxcore."""
    from shark.cli import _get_safe_maxcore_mb
    # With 4 procs, should return between 256 and 4000 MB
    core = _get_safe_maxcore_mb(4)
    assert 256 <= core <= 4000
    # Explicit requested maxcore overrides auto-calculation
    assert _get_safe_maxcore_mb(4, requested_maxcore=256) == 256
    with pytest.raises(ValueError):
        _get_safe_maxcore_mb(0)
    with pytest.raises(ValueError):
        _get_safe_maxcore_mb(1, requested_maxcore=0)
    with pytest.raises(ValueError):
        _get_safe_maxcore_mb(1, requested_maxcore=10**9)


def test_cli_orca_completion_helpers_require_scientific_completion_markers():
    from shark.cli import _orca_optimization_completed, _orca_scan_completed, _safe_float

    assert _safe_float("-7.5") == -7.5
    assert _safe_float("not-a-number") is None
    assert not _orca_optimization_completed("ORCA TERMINATED NORMALLY\noptimization did not converge")
    assert _orca_optimization_completed("THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY")
    assert not _orca_scan_completed("ORCA TERMINATED NORMALLY", point_count=1, expected_points=18)
    assert _orca_scan_completed(
        "RELAXED SURFACE SCAN RESULTS\nORCA TERMINATED NORMALLY",
        point_count=18,
        expected_points=18,
    )


def test_recalc_hess_configurable():
    """TS-1: Recalc_Hess is configurable in QMCluster and workflow."""
    from shark.analysis.qm_cluster import QMCluster, ClusterAtom
    cluster = QMCluster(
        name="TestTS",
        atoms=[
            ClusterAtom(element="C", coords=(0.0, 0.0, 0.0)),
            ClusterAtom(element="O", coords=(1.4, 0.0, 0.0)),
        ],
        charge=0,
        multiplicity=1,
    )
    inp_default = cluster.to_orca_input(job_type="optts", recalc_hess=25)
    assert "Recalc_Hess 25" in inp_default

    inp_custom = cluster.to_orca_input(job_type="optts", recalc_hess=10)
    assert "Recalc_Hess 10" in inp_custom
    assert "Recalc_Hess 25" not in inp_custom
