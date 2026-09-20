"""Reaction-pair validation, mechanism assignment, and Tier 4 preflight verification.

Implements the quality assurance rules specified in SHARK_TIER4_ORCA_RELIABILITY_SPEC.md:
- Explicit validation of nucleophile-electrophile pairs against declared reaction mechanism.
- Prevention of accidental O-O reaction coordinates for carbonyl-addition mechanisms.
- Deterministic input fingerprinting to prevent reusing mismatched calculations.
- Generation of canonical reaction_definition.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ReactionMechanism:
    """Declared reaction mechanism metadata."""
    name: str
    nucleophile_types: list[str]
    electrophile_types: list[str]
    requires_proton_transfer: bool
    reaction_coordinate_type: str
    attack_geometry_model: str
    expected_bond_change: str
    description: str = ""


KNOWN_MECHANISMS: dict[str, ReactionMechanism] = {
    "carbonyl_addition": ReactionMechanism(
        name="carbonyl_addition",
        nucleophile_types=["O", "S", "N"],
        electrophile_types=["C"],
        requires_proton_transfer=True,
        reaction_coordinate_type="distance",
        attack_geometry_model="burgi_dunitz",
        expected_bond_change="Nu-C",
        description="Nucleophilic addition to carbonyl/acyl center (e.g. ester, amide, aldehyde, ketone).",
    ),
    "michael_addition": ReactionMechanism(
        name="michael_addition",
        nucleophile_types=["S", "O", "N"],
        electrophile_types=["C"],
        requires_proton_transfer=False,
        reaction_coordinate_type="distance",
        attack_geometry_model="burgi_dunitz",
        expected_bond_change="Nu-C",
        description="Conjugate 1,4-addition of nucleophile to alpha,beta-unsaturated electrophilic warhead.",
    ),
    "aromatic_substitution_snar": ReactionMechanism(
        name="aromatic_substitution_snar",
        nucleophile_types=["S", "O", "N"],
        electrophile_types=["C"],
        requires_proton_transfer=False,
        reaction_coordinate_type="distance",
        attack_geometry_model="perpendicular",
        expected_bond_change="Nu-C",
        description="Nucleophilic aromatic substitution at electron-deficient heteroaromatic carbon.",
    ),
    "furoxan_heterocycle_attack": ReactionMechanism(
        name="furoxan_heterocycle_attack",
        nucleophile_types=["O", "S", "N"],
        electrophile_types=["N", "C"],
        requires_proton_transfer=True,
        reaction_coordinate_type="distance",
        attack_geometry_model="burgi_dunitz",
        expected_bond_change="Nu-N or Nu-C",
        description="Nucleophilic attack on benzofuroxan / furoxan ring (nitrogen or electrophilic carbon).",
    ),
    "sulfonylation": ReactionMechanism(
        name="sulfonylation",
        nucleophile_types=["O", "N", "S"],
        electrophile_types=["S"],
        requires_proton_transfer=True,
        reaction_coordinate_type="distance",
        attack_geometry_model="trigonal_bipyramidal",
        expected_bond_change="Nu-S",
        description="Attack on sulfonyl fluoride or ester sulfur atom.",
    ),
    "boron_coordination": ReactionMechanism(
        name="boron_coordination",
        nucleophile_types=["O", "S"],
        electrophile_types=["B"],
        requires_proton_transfer=False,
        reaction_coordinate_type="distance",
        attack_geometry_model="tetrahedral_coordination",
        expected_bond_change="Nu-B",
        description="Reversible coordination to boronic acid / ester boron center.",
    ),
    "generic_covalent_addition": ReactionMechanism(
        name="generic_covalent_addition",
        nucleophile_types=["O", "S", "N"],
        electrophile_types=["C", "N", "S", "P", "B"],
        requires_proton_transfer=False,
        reaction_coordinate_type="distance",
        attack_geometry_model="generic",
        expected_bond_change="Nu-E",
        description="Generic covalent bond formation.",
    ),
}


@dataclass
class ReactionCenterAssignment:
    """Reaction center assignment record."""
    nucleophile_idx: int
    electrophile_idx: int
    nucleophile_label: str
    electrophile_label: str
    nucleophile_element: str
    electrophile_element: str
    mechanism: str
    source: str
    confidence: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReactionPairValidation:
    """Preflight validation result of the reaction pair against the reaction mechanism."""
    valid: bool
    severity: str  # 'INFO', 'WARNING', 'ERROR', 'BLOCKING'
    message: str
    warnings: list[str] = field(default_factory=list)
    proposed_bond_type: str = ""
    nucleophile_label: str = ""
    electrophile_label: str = ""


def validate_reaction_pair(
    nucleophile_element: str,
    electrophile_element: str,
    mechanism_name: str = "generic_covalent_addition",
    nucleophile_label: str = "Nu",
    electrophile_label: str = "E",
    initial_distance_angstrom: Optional[float] = None,
    is_same_residue: bool = False,
) -> ReactionPairValidation:
    """Validates whether a nucleophile-electrophile pair is chemically compatible with the declared mechanism.

    Blocks invalid chemical events (such as O-O bonds for carbonyl addition) before ORCA execution.
    """
    nu_el = (nucleophile_element or "").strip().upper()
    el_el = (electrophile_element or "").strip().upper()
    mech = KNOWN_MECHANISMS.get(mechanism_name, KNOWN_MECHANISMS["generic_covalent_addition"])
    proposed_bond = f"{nu_el}-{el_el}"
    warnings: list[str] = []

    # Check 1: Same atom or same residue
    if is_same_residue:
        return ReactionPairValidation(
            valid=False,
            severity="BLOCKING",
            message=f"Nucleophile ({nucleophile_label}) and Electrophile ({electrophile_label}) belong to the same entity.",
            warnings=["Intramolecular self-reaction not supported as target-ligand covalent event."],
            proposed_bond_type=proposed_bond,
            nucleophile_label=nucleophile_label,
            electrophile_label=electrophile_label,
        )

    # Check 2: Initial distance validity
    if initial_distance_angstrom is not None:
        if initial_distance_angstrom > 5.5:
            warnings.append(
                f"Initial distance ({initial_distance_angstrom:.2f} Å) is very distant from reactive contact region (d > 5.5 Å)."
            )
        elif initial_distance_angstrom < 1.1:
            return ReactionPairValidation(
                valid=False,
                severity="BLOCKING",
                message=f"Initial distance ({initial_distance_angstrom:.2f} Å) indicates atomic steric clash or already-formed bond.",
                warnings=["Atoms are severely overlapping."],
                proposed_bond_type=proposed_bond,
                nucleophile_label=nucleophile_label,
                electrophile_label=electrophile_label,
            )

    # Check 3: Check for accidental O-O coordinate under carbonyl or Michael addition
    if nu_el == "O" and el_el == "O":
        if mechanism_name in ("carbonyl_addition", "michael_addition", "aromatic_substitution_snar"):
            return ReactionPairValidation(
                valid=False,
                severity="BLOCKING",
                message=(
                    f"Proposed O–O bond is chemically incompatible with declared mechanism '{mechanism_name}'. "
                    f"Attack must occur at an electrophilic carbon atom, not oxygen."
                ),
                warnings=[
                    "Accidental O-O coordinate rejected (SHARK_TIER4_ORCA_RELIABILITY_SPEC Section 2 & 3).",
                    "Specify electrophilic carbon with --electrophile-atom <NAME>.",
                ],
                proposed_bond_type=proposed_bond,
                nucleophile_label=nucleophile_label,
                electrophile_label=electrophile_label,
            )
        else:
            warnings.append("Peroxide bond (O-O) formation flagged with low confidence.")

    # Check 4: Element compatibility with mechanism
    if el_el not in mech.electrophile_types:
        severity = "BLOCKING" if mechanism_name != "generic_covalent_addition" else "WARNING"
        msg = (
            f"Electrophile element '{el_el}' is not in expected types {mech.electrophile_types} "
            f"for mechanism '{mechanism_name}'."
        )
        return ReactionPairValidation(
            valid=(severity != "BLOCKING"),
            severity=severity,
            message=msg,
            warnings=warnings + [msg],
            proposed_bond_type=proposed_bond,
            nucleophile_label=nucleophile_label,
            electrophile_label=electrophile_label,
        )

    if nu_el not in mech.nucleophile_types:
        warnings.append(
            f"Nucleophile element '{nu_el}' is unconventional for mechanism '{mechanism_name}'."
        )

    return ReactionPairValidation(
        valid=True,
        severity="INFO" if not warnings else "WARNING",
        message=f"Valid {proposed_bond} reaction pair for mechanism '{mechanism_name}'.",
        warnings=warnings,
        proposed_bond_type=proposed_bond,
        nucleophile_label=nucleophile_label,
        electrophile_label=electrophile_label,
    )


def compute_tier4_fingerprint(
    cluster_atoms: Sequence[Any],
    nucleophile_idx: int,
    electrophile_idx: int,
    charge: int,
    multiplicity: int,
    method: str = "r2SCAN-3c",
    solvent: Optional[str] = "Water",
    scan_start: Optional[float] = None,
    scan_end: float = 1.45,
    scan_steps: int = 18,
) -> str:
    """Computes a deterministic SHA-256 fingerprint of the chemical and numerical Tier 4 setup."""
    payload: dict[str, Any] = {
        "num_atoms": len(cluster_atoms),
        "atom_symbols": [getattr(a, "element", str(a)) for a in cluster_atoms],
        "nu_idx": nucleophile_idx,
        "el_idx": electrophile_idx,
        "charge": charge,
        "multiplicity": multiplicity,
        "method": str(method).lower().strip(),
        "solvent": str(solvent).lower().strip() if solvent else "gas",
        "scan_start": round(float(scan_start), 3) if scan_start is not None else None,
        "scan_end": round(float(scan_end), 3),
        "scan_steps": int(scan_steps),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_reaction_definition_json(
    output_path: str | Path,
    target_residue: str,
    nucleophile_label: str,
    nucleophile_element: str,
    nucleophile_idx: int,
    electrophile_label: str,
    electrophile_element: str,
    electrophile_idx: int,
    mechanism: str,
    assignment_source: str,
    validation_status: str,
    initial_distance_angstrom: float,
    proposed_bond: str,
    fingerprint: str,
) -> Path:
    """Writes the canonical reaction_definition.json artifact for Tier 4 TS and adduct calculations."""
    p = Path(output_path)
    data = {
        "target_residue": target_residue,
        "nucleophile": {
            "index": nucleophile_idx,
            "label": nucleophile_label,
            "element": nucleophile_element,
        },
        "electrophile": {
            "index": electrophile_idx,
            "label": electrophile_label,
            "element": electrophile_element,
        },
        "mechanism": mechanism,
        "assignment_source": assignment_source,
        "validation_status": validation_status,
        "initial_distance_angstrom": initial_distance_angstrom,
        "proposed_bond": proposed_bond,
        "fingerprint": fingerprint,
    }
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return p

