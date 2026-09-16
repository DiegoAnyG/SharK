"""Unit tests for Conceptual DFT reactivity and Fukui warhead identification."""

import os
from pathlib import Path
import pytest

from shark.analysis.reactivity import (
    calculate_cdft_descriptors,
    calculate_condensed_fukui,
    build_reactivity_profile,
    CDFTDescriptors,
    ReactivityProfile,
)
from shark.core.parser import parse_orca_results


def test_cdft_descriptors():
    homo = -6.3417
    lumo = -2.8949
    desc = calculate_cdft_descriptors(homo_ev=homo, lumo_ev=lumo)

    assert isinstance(desc, CDFTDescriptors)
    assert pytest.approx(desc.gap_ev, 1e-4) == 3.4468
    assert pytest.approx(desc.hardness_ev, 1e-4) == 3.4468
    assert pytest.approx(desc.hardness_half_ev, 1e-4) == 1.7234
    assert pytest.approx(desc.chemical_potential_ev, 1e-4) == -4.6183
    assert pytest.approx(desc.electronegativity_ev, 1e-4) == 4.6183
    assert pytest.approx(desc.electrophilicity_ev, 1e-4) == 3.0938
    assert pytest.approx(desc.softness_ev, abs=1e-3) == 0.1451
    assert pytest.approx(desc.nucleophilicity_ev, abs=1e-3) == homo - (-9.13)


def test_cdft_inverted_gap_raises():
    with pytest.raises(ValueError, match="Non-physical or inverted frontier gap"):
        calculate_cdft_descriptors(homo_ev=-2.0, lumo_ev=-5.0)


def test_condensed_fukui_finite_difference():
    # 3 atoms: neutral and anion charges
    q_neutral = [0.25, -0.15, -0.10]
    q_anion = [-0.10, -0.45, -0.45]

    f_plus, f_minus, f_zero = calculate_condensed_fukui(
        neutral_charges=q_neutral,
        anion_charges=q_anion
    )

    # f_k^+ = q_neutral - q_anion
    # atom 0: 0.25 - (-0.10) = 0.35
    # atom 1: -0.15 - (-0.45) = 0.30
    # atom 2: -0.10 - (-0.45) = 0.35
    assert pytest.approx(f_plus[0], 1e-4) == 0.35
    assert pytest.approx(f_plus[1], 1e-4) == 0.30
    assert pytest.approx(f_plus[2], 1e-4) == 0.35
    assert pytest.approx(sum(f_plus), 1e-4) == 1.0


def test_build_reactivity_profile_synthetic():
    data = {
        "name": "SyntheticWarhead",
        "homo_ev": -6.0,
        "lumo_ev": -2.0,
        "loewdin_charges": [0.30, -0.20, 0.10, 0.05],
        "atomic_symbols": ["C", "O", "C", "H"],
    }
    profile = build_reactivity_profile(data)
    assert isinstance(profile, ReactivityProfile)
    assert profile.name == "SyntheticWarhead"
    assert profile.global_descriptors.hardness_ev == 4.0
    assert len(profile.atoms) == 4
    
    # Hydrogen (atom 3) should be excluded from warhead candidates
    assert all(c.element != "H" for c in profile.warhead_candidates)
    
    # Top electrophile should be atom 0 (highest positive charge / local electrophilicity)
    top = profile.top_electrophile
    assert top is not None
    assert top.atom_index == 0
    assert top.element == "C"
    assert top.rank == 1


def test_reactivity_profile_from_orca_file():
    root = Path(__file__).resolve().parent.parent
    dft_path = root / "dft_benzofuroxan" / "tautomer_1_oxide"
    if not (dft_path.parent / "tautomer_1_oxide.property.txt").is_file():
        pytest.skip("Benchmark file tautomer_1_oxide.property.txt not available")

    res = parse_orca_results(dft_path, name="Benzofuroxan_1_oxide")
    profile = build_reactivity_profile(res)

    assert profile.global_descriptors.hardness_ev > 3.0
    assert len(profile.warhead_candidates) > 0
    assert profile.top_electrophile is not None
    assert profile.top_electrophile.element in ("C", "N", "O")
    assert profile.top_electrophile.local_electrophilicity > 0.0
