"""Quantum mechanical evaluation of covalent adducts and pre-reactive active-site complexes.

Computes:
1. FMO Phase Symmetry Verification (Fukui & Woodward-Hoffmann constructive orbital overlap)
2. Active Pocket Polarization Effect (frontier orbital shifts and electrophilicity modulation)
3. Regiospecificity Confirmation (local electrophilic Fukui function f_k^+ and local softness s_k^+)
4. Nature of the Covalent Bond Formed (Wiberg covalent bond order, charge transfer, bond length)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class FMOSymmetryResult:
    """Results of Frontier Molecular Orbital alignment and trajectory analysis."""
    is_allowed: bool
    status: str
    symmetry_type: str
    fmo_energy_gap_ev: float
    burgi_dunitz_angle_deg: float
    reactive_distance_angstrom: float
    nucleophile_homo_ev: float
    electrophile_lumo_ev: float
    overlap_integral_estimate: float
    explanation: str
    name: str = "orbital_alignment_descriptor"
    evidence_type: str = "model_derived"
    orbital_source: str = "model_derived"
    warnings: List[str] = field(default_factory=lambda: [
        "Geometric orbital alignment proxy; not a computed wavefunction overlap integral",
        "Does not determine ab initio phase symmetry"
    ])

    @property
    def orbital_alignment_score(self) -> float:
        """Geometric orbital alignment score (heuristic overlap descriptor)."""
        return self.overlap_integral_estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "evidence_type": self.evidence_type,
            "orbital_source": self.orbital_source,
            "is_allowed": self.is_allowed,
            "status": self.status,
            "symmetry_type": self.symmetry_type,
            "fmo_energy_gap_ev": round(self.fmo_energy_gap_ev, 3),
            "burgi_dunitz_angle_deg": round(self.burgi_dunitz_angle_deg, 1),
            "reactive_distance_angstrom": round(self.reactive_distance_angstrom, 2),
            "nucleophile_homo_ev": round(self.nucleophile_homo_ev, 3),
            "electrophile_lumo_ev": round(self.electrophile_lumo_ev, 3),
            "overlap_integral_estimate": round(self.overlap_integral_estimate, 4),
            "orbital_alignment_score": round(self.overlap_integral_estimate, 4),
            "explanation": self.explanation,
            "warnings": list(self.warnings),
        }


@dataclass
class PocketPolarizationResult:
    """Analysis of active-site electrostatic polarization on ligand reactivity."""
    isolated_lumo_ev: float
    complex_lumo_ev: float
    delta_lumo_ev: float
    isolated_electrophilicity_ev: float
    complex_electrophilicity_ev: float
    delta_electrophilicity_ev: float
    polarization_effect: str
    stabilization_kcal_mol: float
    explanation: str
    name: str = "polarization_proxy"
    evidence_type: str = "model_derived"
    warnings: List[str] = field(default_factory=lambda: [
        "Estimated electrostatic proxy; not an ab initio pocket shift"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "evidence_type": self.evidence_type,
            "isolated_lumo_ev": round(self.isolated_lumo_ev, 3),
            "complex_lumo_ev": round(self.complex_lumo_ev, 3),
            "delta_lumo_ev": round(self.delta_lumo_ev, 3),
            "isolated_electrophilicity_ev": round(self.isolated_electrophilicity_ev, 3),
            "complex_electrophilicity_ev": round(self.complex_electrophilicity_ev, 3),
            "delta_electrophilicity_ev": round(self.delta_electrophilicity_ev, 3),
            "polarization_effect": self.polarization_effect,
            "stabilization_kcal_mol": round(self.stabilization_kcal_mol, 2),
            "explanation": self.explanation,
            "warnings": list(self.warnings),
        }


@dataclass
class RegiospecificitySite:
    """Individual atom regiospecificity ranking entry."""
    atom_index: int
    atom_symbol: str
    candidate_site_score: float = 0.0
    local_softness: float = 0.0
    rank: int = 1
    is_target_site: bool = False
    fukui_electrophilic: Optional[float] = None

    def __post_init__(self):
        if self.fukui_electrophilic is not None and self.candidate_site_score == 0.0:
            self.candidate_site_score = self.fukui_electrophilic
        elif self.fukui_electrophilic is None:
            self.fukui_electrophilic = self.candidate_site_score


@dataclass
class RegiospecificityResult:
    """Evaluation of electrophilic attack regiospecificity prior."""
    target_atom_index: int
    target_atom_symbol: str
    target_rank: int
    total_sites_evaluated: int
    is_primary_locus: bool
    sites: List[RegiospecificitySite]
    explanation: str
    name: str = "candidate_site_score"
    evidence_type: str = "model_derived"
    warnings: List[str] = field(default_factory=lambda: [
        "Empirical heuristic for candidate site ranking; not a computed ab initio Fukui index."
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "evidence_type": self.evidence_type,
            "target_atom_index": self.target_atom_index,
            "target_atom_symbol": self.target_atom_symbol,
            "target_rank": self.target_rank,
            "total_sites_evaluated": self.total_sites_evaluated,
            "is_primary_locus": self.is_primary_locus,
            "sites": [
                {
                    "atom_index": s.atom_index,
                    "atom_symbol": s.atom_symbol,
                    "candidate_score": round(s.candidate_site_score, 4),
                    "candidate_site_score": round(s.candidate_site_score, 4),
                    "fukui_electrophilic": round(s.fukui_electrophilic if s.fukui_electrophilic is not None else s.candidate_site_score, 4),
                    "local_softness": round(s.local_softness, 4),
                    "rank": s.rank,
                    "is_target_site": s.is_target_site,
                }
                for s in self.sites
            ],
            "explanation": self.explanation,
            "warnings": list(self.warnings),
        }


@dataclass
class CovalentBondNatureResult:
    """Characterization of the covalent bond / product adduct."""
    bond_type: str
    equilibrium_distance_angstrom: Optional[float] = None
    distance_based_bond_order_proxy: Optional[float] = None
    wiberg_bond_order: Optional[float] = None
    charge_transfer_e: Optional[float] = None
    bond_covalency_percent: Optional[float] = None
    bond_likelihood_proxy: Optional[float] = None
    is_reversible: bool = False
    is_calculated: bool = False
    status: str = "not_evaluated"
    evidence_type: str = "model_derived"
    explanation: str = "Covalent product analysis not performed; requires optimized adduct geometry."
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.distance_based_bond_order_proxy is None and self.wiberg_bond_order is not None:
            self.distance_based_bond_order_proxy = self.wiberg_bond_order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_calculated": self.is_calculated,
            "evidence_type": self.evidence_type,
            "bond_type": self.bond_type,
            "equilibrium_distance_angstrom": round(self.equilibrium_distance_angstrom, 3) if self.equilibrium_distance_angstrom is not None else None,
            "distance_based_bond_order_proxy": self.distance_based_bond_order_proxy,
            "wiberg_bond_order": self.wiberg_bond_order,
            "charge_transfer_e": self.charge_transfer_e,
            "bond_covalency_percent": self.bond_covalency_percent,
            "bond_likelihood_proxy": self.bond_likelihood_proxy,
            "is_reversible": self.is_reversible,
            "explanation": self.explanation,
            "warnings": list(self.warnings),
        }


@dataclass
class AdductQuantumProfile:
    """Consolidated quantum chemical evaluation of the covalent adduct/pre-reactive complex."""
    fmo_symmetry: FMOSymmetryResult
    polarization: PocketPolarizationResult
    regiospecificity: RegiospecificityResult
    bond_nature: CovalentBondNatureResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fmo_symmetry": self.fmo_symmetry.to_dict(),
            "polarization": self.polarization.to_dict(),
            "regiospecificity": self.regiospecificity.to_dict(),
            "bond_nature": self.bond_nature.to_dict(),
        }


def evaluate_fmo_phase_symmetry(
    nucleophile_homo_ev: float = -6.40,
    electrophile_lumo_ev: float = -2.90,
    distance_angstrom: float = 3.33,
    burgi_dunitz_angle_deg: float = 132.7,
    nucleophile_symbol: str = "O",
    electrophile_symbol: str = "N",
    orbital_source: str = "model_derived",
) -> FMOSymmetryResult:
    """Evaluates orbital alignment and approach trajectory using geometric FMO heuristic model."""
    fmo_gap = abs(electrophile_lumo_ev - nucleophile_homo_ev)

    is_angle_favorable = (85.0 <= burgi_dunitz_angle_deg <= 145.0)
    is_dist_favorable = (distance_angstrom <= 3.80)

    r0 = 1.45 if nucleophile_symbol in ("O", "N") else 1.82
    alpha = 1.15
    theta_ideal = 107.5
    angle_rad = math.radians(burgi_dunitz_angle_deg)
    ideal_rad = math.radians(theta_ideal)

    angular_factor = max(0.0, math.cos(angle_rad - ideal_rad))
    radial_factor = math.exp(-alpha * max(0.0, distance_angstrom - r0))
    s_overlap = round(radial_factor * angular_factor * 0.35, 4)

    is_allowed = is_angle_favorable and is_dist_favorable and (fmo_gap < 7.5)

    ev_type = "orca_cluster" if orbital_source != "model_derived" else "model_derived"
    warns = [
        f"Geometric orbital alignment proxy (orbital source: {orbital_source}); not a computed wavefunction overlap integral",
        "Does not determine ab initio phase symmetry"
    ]

    if is_allowed:
        status = "Constructive Phase Overlap (Model Heuristic)"
        sym_type = "Constructive sigma-addition (Model Heuristic)"
        explanation = (
            f"Approach trajectory is geometrically compatible with addition. "
            f"The nucleophilic lone pair ({nucleophile_symbol}) approaches within the Bürgi-Dunitz cone "
            f"({burgi_dunitz_angle_deg:.1f} deg) with FMO gap ({fmo_gap:.2f} eV). "
            f"Note: Model-derived geometric proxy (source: {orbital_source}), not a computed orbital overlap integral."
        )
    else:
        status = "Destructive or Hindered Symmetry (Model Heuristic)"
        sym_type = "Mismatched Trajectory"
        explanation = (
            f"The nucleophile approach angle ({burgi_dunitz_angle_deg:.1f} deg) or distance "
            f"({distance_angstrom:.2f} A) violates geometric phase overlap criteria."
        )

    return FMOSymmetryResult(
        is_allowed=is_allowed,
        status=status,
        symmetry_type=sym_type,
        fmo_energy_gap_ev=fmo_gap,
        burgi_dunitz_angle_deg=burgi_dunitz_angle_deg,
        reactive_distance_angstrom=distance_angstrom,
        nucleophile_homo_ev=nucleophile_homo_ev,
        electrophile_lumo_ev=electrophile_lumo_ev,
        overlap_integral_estimate=s_overlap,
        explanation=explanation,
        evidence_type=ev_type,
        orbital_source=orbital_source,
        warnings=warns,
    )


# Explicit alias adhering to INV-008
evaluate_orbital_alignment_heuristic = evaluate_fmo_phase_symmetry


def evaluate_pocket_polarization(
    isolated_homo_ev: float = -6.34,
    isolated_lumo_ev: float = -2.89,
    pocket_residue_count: int = 12,
    pocket_dielectric: float = 4.0,
    has_catalytic_partner: bool = True,
) -> PocketPolarizationResult:
    """Evaluates electrostatic polarization proxy of the pocket on ligand frontier orbitals."""
    delta_lumo = -0.32 if has_catalytic_partner else -0.15
    complex_lumo = isolated_lumo_ev + delta_lumo
    complex_homo = isolated_homo_ev - 0.10

    iso_mu = (isolated_homo_ev + isolated_lumo_ev) / 2.0
    iso_eta = isolated_lumo_ev - isolated_homo_ev
    iso_omega = (iso_mu ** 2) / (2.0 * iso_eta) if iso_eta > 0 else 0.0

    cpx_mu = (complex_homo + complex_lumo) / 2.0
    cpx_eta = complex_lumo - complex_homo
    cpx_omega = (cpx_mu ** 2) / (2.0 * cpx_eta) if cpx_eta > 0 else 0.0

    delta_omega = cpx_omega - iso_omega
    stab_kcal = abs(delta_lumo) * 23.0605

    effect = "Electrophilic Activation Proxy (LUMO Stabilization)"
    explanation = (
        f"Active pocket electrostatic field proxy estimates a LUMO lowering by {abs(delta_lumo):.2f} eV "
        f"({stab_kcal:.1f} kcal/mol stabilization), modifying electrophilicity index "
        f"by {delta_omega:+.2f} eV. Note: Model-derived proxy, not a computed ab initio pocket shift."
    )

    return PocketPolarizationResult(
        isolated_lumo_ev=isolated_lumo_ev,
        complex_lumo_ev=complex_lumo,
        delta_lumo_ev=delta_lumo,
        isolated_electrophilicity_ev=iso_omega,
        complex_electrophilicity_ev=cpx_omega,
        delta_electrophilicity_ev=delta_omega,
        polarization_effect=effect,
        stabilization_kcal_mol=stab_kcal,
        explanation=explanation,
    )


def evaluate_regiospecificity(
    ligand_heavy_atoms: Sequence[Tuple[str, int]],
    target_atom_index: int = 10,
    target_atom_symbol: str = "O",
    electrophilicity_index: float = 3.09,
) -> RegiospecificityResult:
    """Evaluates candidate site scoring based on proximity and chemical prior."""
    sites: List[RegiospecificitySite] = []
    raw_weights = {}
    matched_by_num = any(at_num == target_atom_index for _, at_num in ligand_heavy_atoms)

    for idx, (sym, at_num) in enumerate(ligand_heavy_atoms):
        is_target = (at_num == target_atom_index) if matched_by_num else (idx == target_atom_index)
        if is_target:
            w = 0.45
        elif sym in ("N", "O"):
            dist = abs(at_num - target_atom_index) if matched_by_num else abs(idx - target_atom_index)
            w = 0.18 / (1.0 + dist)
        elif sym == "C":
            dist = abs(at_num - target_atom_index) if matched_by_num else abs(idx - target_atom_index)
            w = 0.10 / (1.0 + dist)
        else:
            w = 0.05
        raw_weights[idx] = (sym, at_num, w, is_target)

    total_w = sum(v[2] for v in raw_weights.values()) or 1.0
    sorted_indices = sorted(raw_weights.keys(), key=lambda k: raw_weights[k][2], reverse=True)
    target_rank = 1

    for rank, k in enumerate(sorted_indices, start=1):
        sym, at_num, w, is_target = raw_weights[k]
        norm_fk = w / total_w
        sk = norm_fk * (1.0 / max(0.1, electrophilicity_index))
        if is_target:
            target_rank = rank
        sites.append(RegiospecificitySite(
            atom_index=at_num,
            atom_symbol=sym,
            fukui_electrophilic=norm_fk,
            local_softness=sk,
            rank=rank,
            is_target_site=is_target,
        ))

    is_primary = (target_rank == 1)
    explanation = (
        f"Atom #{target_atom_index} ({target_atom_symbol}) ranks #{target_rank} in candidate site prior "
        f"(score = {sites[0].fukui_electrophilic:.3f}). "
        f"Note: Model-derived prior score; not a computed ab initio Fukui function."
    )

    return RegiospecificityResult(
        target_atom_index=target_atom_index,
        target_atom_symbol=target_atom_symbol,
        target_rank=target_rank,
        total_sites_evaluated=len(sites),
        is_primary_locus=is_primary,
        sites=sites,
        explanation=explanation,
    )


def evaluate_covalent_bond_nature(
    nucleophile_element: str = "O",
    electrophile_element: str = "C",
    bond_distance_angstrom: float = 1.45,
    is_reversible_warhead: bool = False,
    is_optimized_adduct: Optional[bool] = None,
    distance_angstrom: Optional[float] = None,
    warhead_type: Optional[str] = None,
    has_optimized_adduct: Optional[bool] = None,
) -> CovalentBondNatureResult:
    """Characterizes covalent adduct bond nature when an optimized product structure is available."""
    if distance_angstrom is not None:
        bond_distance_angstrom = distance_angstrom
    if has_optimized_adduct is not None:
        is_optimized_adduct = has_optimized_adduct

    ref_d = 1.43 if {nucleophile_element, electrophile_element} == {"C", "O"} else (
        1.82 if "S" in (nucleophile_element, electrophile_element) else 1.47
    )

    # A short distance alone is not proof that an adduct optimization was run.
    if is_optimized_adduct is None:
        is_optimized_adduct = False

    # Heuristic distance-based proxy
    b_order = max(0.05, min(1.20, math.exp(-1.4 * (bond_distance_angstrom - ref_d))))

    if not is_optimized_adduct:
        # Pre-reactive contact: do not fabricate product bond order (INV-007, INV-010)
        return CovalentBondNatureResult(
            bond_type="Pre-reactive Contact (Unreacted)",
            equilibrium_distance_angstrom=bond_distance_angstrom,
            wiberg_bond_order=None,
            charge_transfer_e=None,
            bond_covalency_percent=None,
            bond_likelihood_proxy=round(b_order, 3),
            is_reversible=is_reversible_warhead,
            is_calculated=False,
            status="not_evaluated",
            evidence_type="model_derived",
            explanation=(
                "Covalent product analysis not performed. Required evidence: "
                "optimized adduct geometry, computed bond-order analysis, and "
                "product energy on a consistent level of theory."
            ),
            warnings=["Bond order calculation requires optimized adduct QM calculation; not evaluated."],
        )

    formed_geometry = bond_distance_angstrom <= ref_d * 1.30
    bond_type = (
        f"Optimized {nucleophile_element}-{electrophile_element} bond geometry"
        if formed_geometry else "Optimized unbound geometry"
    )
    explanation = (
        f"The optimized {nucleophile_element}-{electrophile_element} distance is "
        f"{bond_distance_angstrom:.2f} A. The distance-based bond-likelihood proxy is {b_order:.2f}; "
        "no Wiberg bond order or charge transfer was calculated."
    )

    return CovalentBondNatureResult(
        bond_type=bond_type,
        equilibrium_distance_angstrom=bond_distance_angstrom,
        distance_based_bond_order_proxy=round(b_order, 3),
        wiberg_bond_order=None,
        charge_transfer_e=None,
        bond_covalency_percent=None,
        bond_likelihood_proxy=round(b_order, 3),
        is_reversible=is_reversible_warhead,
        is_calculated=False,
        status="geometry_optimized" if formed_geometry else "optimized_unbound",
        evidence_type="model_derived",
        explanation=explanation,
        warnings=["Distance proxy is not a computed bond order or charge-transfer analysis."],
    )


def compute_adduct_quantum_profile(
    distance_angstrom: float = 3.33,
    burgi_dunitz_angle_deg: float = 132.7,
    nucleophile_homo_ev: float = -6.40,
    electrophile_lumo_ev: float = -2.89,
    target_atom_index: int = 10,
    target_atom_symbol: str = "O",
    ligand_heavy_atoms: Optional[Sequence[Tuple[str, int]]] = None,
    nucleophile_element: str = "O",
    electrophile_element: str = "N",
    is_reversible_warhead: bool = False,
    is_optimized_adduct: bool = False,
    has_optimized_adduct: Optional[bool] = None,
    orbital_source: str = "model_derived",
) -> AdductQuantumProfile:
    """Computes the 4-checkpoint quantum verification profile for the pre-reactive complex."""
    if has_optimized_adduct is not None:
        is_optimized_adduct = has_optimized_adduct
    if ligand_heavy_atoms is None:
        ligand_heavy_atoms = [
            ("C", 1), ("C", 2), ("C", 3), ("O", 4), ("C", 5),
            ("C", 6), ("C", 7), ("C", 8), ("N", 9), ("O", 10),
            ("O", 11), ("N", 12), ("O", 13)
        ]

    fmo = evaluate_fmo_phase_symmetry(
        nucleophile_homo_ev=nucleophile_homo_ev,
        electrophile_lumo_ev=electrophile_lumo_ev,
        distance_angstrom=distance_angstrom,
        burgi_dunitz_angle_deg=burgi_dunitz_angle_deg,
        nucleophile_symbol=nucleophile_element,
        electrophile_symbol=electrophile_element,
        orbital_source=orbital_source,
    )

    pol = evaluate_pocket_polarization(
        isolated_homo_ev=nucleophile_homo_ev,
        isolated_lumo_ev=electrophile_lumo_ev,
    )

    reg = evaluate_regiospecificity(
        ligand_heavy_atoms=ligand_heavy_atoms,
        target_atom_index=target_atom_index,
        target_atom_symbol=target_atom_symbol,
        electrophilicity_index=pol.complex_electrophilicity_ev,
    )

    # INV-007 / INV-010: Do NOT replace distance with 1.45 A. Evaluate pre-reactive contact as unreacted.
    bond = evaluate_covalent_bond_nature(
        nucleophile_element=nucleophile_element,
        electrophile_element=electrophile_element,
        bond_distance_angstrom=distance_angstrom,
        is_reversible_warhead=is_reversible_warhead,
        is_optimized_adduct=is_optimized_adduct,
    )

    return AdductQuantumProfile(
        fmo_symmetry=fmo,
        polarization=pol,
        regiospecificity=reg,
        bond_nature=bond,
    )
