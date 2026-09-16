"""Conceptual Density Functional Theory (CDFT) and warhead reactivity engine.

Calculates global electronic reactivity descriptors (hardness, chemical potential,
global electrophilicity index, softness, nucleophilicity index) and local atomic
descriptors (condensed Fukui functions f_k^+, local electrophilicity omega_k)
to identify and rank reactive covalent warhead centers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple, Union

# Fundamental conversion constants
HARTREE_TO_EV = 27.211386
EV_TO_KCAL = 23.060548

# Tetracyanoethylene (TCE) HOMO reference energy in eV (Domingo et al. scale)
TCE_HOMO_REF_EV = -9.13

# Non-reactive or solvent-exchangeable elements generally excluded as primary warhead centers
NON_WARHEAD_ELEMENTS = {"H", "HE", "LI", "BE", "NA", "MG", "K", "CA"}


@dataclass
class CDFTDescriptors:
    """Global Conceptual DFT descriptors."""
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    hardness_ev: float             # eta = lumo - homo (fundamental gap)
    chemical_potential_ev: float    # mu = (homo + lumo) / 2
    electronegativity_ev: float     # chi = -mu
    electrophilicity_ev: float      # omega = mu^2 / (2 * eta)
    softness_ev: float              # S = 1 / (2 * eta)
    nucleophilicity_ev: Optional[float] = None  # N = homo - homo(TCE)

    @property
    def hardness_half_ev(self) -> float:
        """Parr-Pearson operational half-gap hardness: (lumo - homo) / 2."""
        return self.hardness_ev / 2.0


@dataclass
class LocalReactivityAtom:
    """Local atomic reactivity and Fukui indices for a specific atom."""
    atom_index: int
    element: str
    charge: float
    coordinates: Optional[Tuple[float, float, float]] = None
    fukui_plus: Optional[float] = None       # f_k^+ : susceptibility to nucleophilic attack
    fukui_minus: Optional[float] = None      # f_k^- : susceptibility to electrophilic attack
    fukui_zero: Optional[float] = None       # f_k^0 : radical attack susceptibility
    local_electrophilicity: Optional[float] = None  # omega_k = omega * f_k^+ (eV)
    is_warhead_candidate: bool = False
    rank: Optional[int] = None


@dataclass
class ReactivityProfile:
    """Complete global and local CDFT profile for a molecule."""
    name: str
    global_descriptors: CDFTDescriptors
    atoms: List[LocalReactivityAtom] = field(default_factory=list)
    warhead_candidates: List[LocalReactivityAtom] = field(default_factory=list)

    @property
    def top_electrophile(self) -> Optional[LocalReactivityAtom]:
        """Returns the primary electrophilic warhead center (rank 1)."""
        return self.warhead_candidates[0] if self.warhead_candidates else None


def calculate_cdft_descriptors(
    homo_ev: float,
    lumo_ev: float,
    tce_homo_ref: float = TCE_HOMO_REF_EV
) -> CDFTDescriptors:
    """Calculates global Conceptual DFT descriptors from frontier orbital eigenvalues.
    
    Parameters
    ----------
    homo_ev : float
        Highest Occupied Molecular Orbital energy in eV.
    lumo_ev : float
        Lowest Unoccupied Molecular Orbital energy in eV.
    tce_homo_ref : float
        Reference TCE HOMO energy in eV for Domingo nucleophilicity scale.
    """
    gap = lumo_ev - homo_ev
    if gap <= 0.0:
        raise ValueError(f"Non-physical or inverted frontier gap: HOMO={homo_ev:.3f} eV, LUMO={lumo_ev:.3f} eV")

    hardness = gap
    chemical_potential = (homo_ev + lumo_ev) / 2.0
    electronegativity = -chemical_potential
    electrophilicity = (chemical_potential ** 2) / (2.0 * hardness)
    softness = 1.0 / (2.0 * hardness)
    nucleophilicity = homo_ev - tce_homo_ref

    return CDFTDescriptors(
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        gap_ev=gap,
        hardness_ev=hardness,
        chemical_potential_ev=chemical_potential,
        electronegativity_ev=electronegativity,
        electrophilicity_ev=electrophilicity,
        softness_ev=softness,
        nucleophilicity_ev=nucleophilicity
    )


def calculate_condensed_fukui(
    neutral_charges: List[float],
    anion_charges: Optional[List[float]] = None,
    cation_charges: Optional[List[float]] = None
) -> Tuple[List[float], List[float], List[float]]:
    """Calculates condensed Fukui functions (f+, f-, f0) per atom.
    
    If anion/cation states are provided, uses the finite difference formulation:
        f_k^+ = q_k(N) - q_k(N+1)   (susceptibility to nucleophilic attack)
        f_k^- = q_k(N-1) - q_k(N)   (susceptibility to electrophilic attack)
        f_k^0 = (f_k^+ + f_k^-) / 2 (radical attack susceptibility)
        
    When only ground-state neutral charges are provided, approximates f_k^+ from
    the normalized positive partial charge density on atoms.
    """
    n_atoms = len(neutral_charges)
    if n_atoms == 0:
        return [], [], []

    # 1. Nucleophilic attack susceptibility (f+)
    if anion_charges is not None and len(anion_charges) == n_atoms:
        f_plus = [q_n - q_a for q_n, q_a in zip(neutral_charges, anion_charges)]
    else:
        # Ground-state population approximation: electrophiles have positive partial charge
        pos_charges = [max(0.0, q) for q in neutral_charges]
        total_pos = sum(pos_charges)
        if total_pos > 1e-6:
            f_plus = [q / total_pos for q in pos_charges]
        else:
            f_plus = [1.0 / n_atoms] * n_atoms

    # 2. Electrophilic attack susceptibility (f-)
    if cation_charges is not None and len(cation_charges) == n_atoms:
        f_minus = [q_c - q_n for q_c, q_n in zip(cation_charges, neutral_charges)]
    else:
        neg_charges = [max(0.0, -q) for q in neutral_charges]
        total_neg = sum(neg_charges)
        if total_neg > 1e-6:
            f_minus = [q / total_neg for q in neg_charges]
        else:
            f_minus = [1.0 / n_atoms] * n_atoms

    # 3. Radical susceptibility (f0)
    f_zero = [(fp + fm) / 2.0 for fp, fm in zip(f_plus, f_minus)]

    return f_plus, f_minus, f_zero


def build_reactivity_profile(
    calc_or_dict: Union[dict, object],
    name: str = "",
    charge_type: str = "loewdin",
    anion_charges: Optional[List[float]] = None,
    cation_charges: Optional[List[float]] = None
) -> ReactivityProfile:
    """Builds a complete Conceptual DFT ReactivityProfile from parsed calculation results."""
    # Extract HOMO and LUMO
    if isinstance(calc_or_dict, dict):
        homo = calc_or_dict.get("homo_ev")
        lumo = calc_or_dict.get("lumo_ev")
        loewdin = calc_or_dict.get("loewdin_charges", [])
        mulliken = calc_or_dict.get("mulliken_charges", [])
        symbols = calc_or_dict.get("atomic_symbols", [])
        coords = calc_or_dict.get("coordinates", [])
        mol_name = name or str(calc_or_dict.get("name", "molecule"))
    else:
        homo = getattr(calc_or_dict, "homo_energy", None)
        lumo = getattr(calc_or_dict, "lumo_energy", None)
        loewdin = getattr(calc_or_dict, "loewdin_charges", [])
        mulliken = getattr(calc_or_dict, "mulliken_charges", [])
        symbols = getattr(calc_or_dict, "atomic_symbols", [])
        coords = getattr(calc_or_dict, "coordinates_angstrom", [])
        mol_name = name or getattr(calc_or_dict, "name", "molecule")

    if homo is None or lumo is None:
        raise ValueError("Cannot calculate CDFT descriptors: HOMO or LUMO eigenvalue missing.")

    global_desc = calculate_cdft_descriptors(homo_ev=homo, lumo_ev=lumo)

    # Select atomic charges
    charges = loewdin if charge_type.lower() == "loewdin" and loewdin else mulliken
    if not charges and loewdin:
        charges = loewdin
    elif not charges and mulliken:
        charges = mulliken

    n_atoms = len(charges)
    if not symbols:
        symbols = ["X"] * n_atoms
    if not coords:
        coords = [None] * n_atoms

    f_plus, f_minus, f_zero = calculate_condensed_fukui(
        neutral_charges=charges,
        anion_charges=anion_charges,
        cation_charges=cation_charges
    )

    atom_list: List[LocalReactivityAtom] = []
    candidates: List[LocalReactivityAtom] = []

    for i in range(n_atoms):
        elem = symbols[i] if i < len(symbols) else "X"
        chg = charges[i]
        fp = f_plus[i] if i < len(f_plus) else 0.0
        fm = f_minus[i] if i < len(f_minus) else 0.0
        fz = f_zero[i] if i < len(f_zero) else 0.0
        crd = coords[i] if i < len(coords) else None

        # Local electrophilicity: omega_k = omega * f_k^+
        omega_k = global_desc.electrophilicity_ev * fp

        is_cand = elem.upper() not in NON_WARHEAD_ELEMENTS and (omega_k > 0.01 or chg > 0.02)

        atom_obj = LocalReactivityAtom(
            atom_index=i,
            element=elem,
            charge=chg,
            coordinates=crd,
            fukui_plus=fp,
            fukui_minus=fm,
            fukui_zero=fz,
            local_electrophilicity=omega_k,
            is_warhead_candidate=is_cand
        )
        atom_list.append(atom_obj)
        if is_cand:
            candidates.append(atom_obj)

    # Rank candidates by local electrophilicity descending
    candidates.sort(key=lambda a: (a.local_electrophilicity or 0.0, a.charge), reverse=True)
    for rank_idx, cand in enumerate(candidates, 1):
        cand.rank = rank_idx

    return ReactivityProfile(
        name=mol_name,
        global_descriptors=global_desc,
        atoms=atom_list,
        warhead_candidates=candidates
    )
