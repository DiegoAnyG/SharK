"""Quantum Mechanical Transition State (TS) Modeling & Activation Energy Engine.

Performs:
1. Automated ORCA relaxed surface coordinate scans between nucleophile and electrophile.
2. Extraction of transition state initial guess (energy maximum on the reaction coordinate).
3. Saddle-point optimization (! OptTS Freq) and vibrational Hessian verification (strictly 1 imaginary mode).
4. Thermochemical profile and activation free energy calculation (Delta G‡ and Delta G_rxn)
   based on Eyring transition state theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple, Union

from .qm_cluster import QMCluster, extract_qm_cluster, _euclidean_distance
from ..core.parser import parse_orca_results, CalculationResult

HARTREE_TO_KCAL = 627.509474
R_GAS_KCAL = 0.00198720425864083  # kcal / (mol * K)
KB_OVER_H = 2.0836619e10          # (k_B / h) in s^-1 * K^-1 (Eyring prefactor: ~6.2124e12 at 298.15 K)


@dataclass
class ScanPoint:
    """An individual step along the relaxed surface scan reaction coordinate."""
    step: int
    coordinate_value: float          # Distance in Angstroms
    energy_hartree: float            # Electronic energy in Hartrees
    relative_energy_kcal: float     # Energy relative to scan start in kcal/mol
    xyz_path: Optional[Path] = None
    geometry: List[Tuple[str, Tuple[float, float, float]]] = field(default_factory=list)


@dataclass
class ScanResult:
    """Summary of the relaxed coordinate scan."""
    points: List[ScanPoint] = field(default_factory=list)
    max_energy_step: int = 1
    max_energy_hartree: float = 0.0
    barrier_estimate_kcal: float = 0.0
    delta_e_scan_activation_kcal: float = 0.0
    delta_e_scan_rxn_kcal: Optional[float] = None
    ts_guess_coord_value: float = 0.0
    ts_guess_xyz: Optional[Path] = None
    converged: bool = True
    raw_output: str = ""

    @property
    def summary(self) -> str:
        rxn_str = f" | Estimated ΔE_scan_rxn: {self.delta_e_scan_rxn_kcal:.2f} kcal/mol" if self.delta_e_scan_rxn_kcal is not None else ""
        return (
            f"Coordinate Scan (Electronic): {len(self.points)} steps | "
            f"TS Guess at Step {self.max_energy_step} (d={self.ts_guess_coord_value:.2f} A) | "
            f"Estimated ΔE_scan‡: {self.delta_e_scan_activation_kcal:.2f} kcal/mol{rxn_str}"
        )


@dataclass
class TSVerificationResult:
    """Results from saddle point optimization and vibrational frequency analysis."""
    name: str
    converged: bool = False
    n_imaginary_frequencies: int = 0
    imaginary_modes: List[float] = field(default_factory=list)
    all_frequencies: List[float] = field(default_factory=list)
    is_valid_first_order_saddle_point: bool = False
    transition_vector_summary: str = ""
    electronic_energy_hartree: float = 0.0
    zpe_hartree: float = 0.0
    enthalpy_hartree: float = 0.0
    entropy_hartree: float = 0.0
    gibbs_free_energy_hartree: float = 0.0
    temperature_k: float = 298.15
    coordinates_angstrom: List[Tuple[float, float, float]] = field(default_factory=list)
    atomic_symbols: List[str] = field(default_factory=list)


@dataclass
class ReactionEnergyProfile:
    """Comprehensive thermodynamic and kinetic profile of the covalent reaction."""
    reactants_gibbs: Optional[float] = None   # Hartree
    ts_gibbs: Optional[float] = None          # Hartree
    product_gibbs: Optional[float] = None     # Hartree
    reactants_electronic: Optional[float] = None  # Hartree
    ts_electronic: Optional[float] = None         # Hartree
    product_electronic: Optional[float] = None    # Hartree
    energy_basis: str = "gibbs"               # "gibbs", "electronic", or "inconsistent"
    barrier_symbol: str = "ΔG‡"              # "ΔG‡" or "ΔE‡"
    reaction_energy_symbol: str = "ΔG_rxn"    # "ΔG_rxn" or "ΔE_rxn"
    delta_g_activation_kcal: Optional[float] = None  # kcal/mol: activation Gibbs free energy (Route B)
    delta_g_reaction_kcal: Optional[float] = None    # kcal/mol: reaction Gibbs free energy (Route B)
    delta_e_activation_kcal: Optional[float] = None  # kcal/mol: activation electronic energy (Route A)
    delta_e_reaction_kcal: Optional[float] = None    # kcal/mol: reaction electronic energy (Route A)
    delta_e_scan_activation_kcal: Optional[float] = None  # kcal/mol: electronic scan barrier
    delta_e_scan_rxn_kcal: Optional[float] = None    # kcal/mol: electronic scan reaction energy
    rate_constant_s: Optional[float] = None   # s^-1 (None when energy_basis == "electronic" or "inconsistent")
    estimated_half_life_str: str = ""
    kinetic_feasibility: str = ""
    temperature_k: float = 298.15
    is_first_order_ts: bool = True
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def activation_barrier_kcal(self) -> float:
        if self.energy_basis == "gibbs" and self.delta_g_activation_kcal is not None:
            return self.delta_g_activation_kcal
        if self.delta_e_activation_kcal is not None:
            return self.delta_e_activation_kcal
        if self.delta_e_scan_activation_kcal is not None:
            return self.delta_e_scan_activation_kcal
        return self.delta_g_activation_kcal if self.delta_g_activation_kcal is not None else 0.0

    @property
    def reaction_energy_kcal(self) -> Optional[float]:
        if self.energy_basis == "gibbs":
            return self.delta_g_reaction_kcal
        if self.delta_e_reaction_kcal is not None:
            return self.delta_e_reaction_kcal
        return self.delta_e_scan_rxn_kcal

    @property
    def summary(self) -> str:
        rxn_val = self.reaction_energy_kcal
        prod_str = f", {self.reaction_energy_symbol}: {rxn_val:.2f} kcal/mol" if rxn_val is not None else ""
        rate_info = f"t1/2 ~ {self.estimated_half_life_str}" if self.rate_constant_s is not None else self.estimated_half_life_str
        return (
            f"Reaction Profile ({self.energy_basis}): {self.barrier_symbol} = {self.activation_barrier_kcal:.2f} kcal/mol{prod_str} | "
            f"Feasibility: {self.kinetic_feasibility} ({rate_info})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy_basis": self.energy_basis,
            "barrier_symbol": self.barrier_symbol,
            "reaction_energy_symbol": self.reaction_energy_symbol,
            "delta_g_activation_kcal": self.delta_g_activation_kcal,
            "delta_g_reaction_kcal": self.delta_g_reaction_kcal,
            "delta_e_activation_kcal": self.delta_e_activation_kcal,
            "delta_e_reaction_kcal": self.delta_e_reaction_kcal,
            "delta_e_scan_activation_kcal": self.delta_e_scan_activation_kcal,
            "delta_e_scan_rxn_kcal": self.delta_e_scan_rxn_kcal,
            "activation_barrier_kcal": self.activation_barrier_kcal,
            "reaction_energy_kcal": self.reaction_energy_kcal,
            "rate_constant_s": self.rate_constant_s,
            "estimated_half_life_str": self.estimated_half_life_str,
            "kinetic_feasibility": self.kinetic_feasibility,
            "temperature_k": self.temperature_k,
            "is_first_order_ts": self.is_first_order_ts,
            "reactants_gibbs": self.reactants_gibbs,
            "ts_gibbs": self.ts_gibbs,
            "product_gibbs": self.product_gibbs,
            "reactants_electronic": self.reactants_electronic,
            "ts_electronic": self.ts_electronic,
            "product_electronic": self.product_electronic,
            "notes": self.notes,
            "warnings": list(self.warnings),
        }


def select_consistent_energy_basis(
    reactant_el: Optional[float] = None,
    ts_el: Optional[float] = None,
    product_el: Optional[float] = None,
    reactant_gibbs: Optional[float] = None,
    ts_gibbs: Optional[float] = None,
    product_gibbs: Optional[float] = None,
    temperature_k: float = 298.15,
) -> Dict[str, Any]:
    """Selects a homogeneous energy basis across reaction species without mixing E and G.

    Enforces INV-006:
    Allowed: ΔE‡ = E_TS - E_R, ΔE_rxn = E_P - E_R
    Allowed: ΔG‡ = G_TS - G_R, ΔG_rxn = G_P - G_R
    Forbidden: G_TS - E_R or any combination of mixed bases.
    """
    has_gibbs_r = reactant_gibbs is not None and math.isfinite(reactant_gibbs) and reactant_gibbs != 0.0
    has_gibbs_ts = ts_gibbs is not None and math.isfinite(ts_gibbs) and ts_gibbs != 0.0
    has_gibbs_p = product_gibbs is not None and math.isfinite(product_gibbs) and product_gibbs != 0.0

    has_el_r = reactant_el is not None and math.isfinite(reactant_el) and reactant_el != 0.0
    has_el_ts = ts_el is not None and math.isfinite(ts_el) and ts_el != 0.0
    has_el_p = product_el is not None and math.isfinite(product_el) and product_el != 0.0

    # Route B: Gibbs free energies if both R and TS have valid Gibbs
    if has_gibbs_r and has_gibbs_ts:
        return {
            "basis": "gibbs",
            "temperature_k": temperature_k,
            "reactant": reactant_gibbs,
            "ts": ts_gibbs,
            "product": product_gibbs if has_gibbs_p else None,
            "barrier_symbol": "ΔG‡",
            "reaction_symbol": "ΔG_rxn",
            "units": "kcal/mol",
            "is_valid": True,
            "warnings": [],
        }

    # Route A: Electronic energies if both R and TS have valid electronic energies
    if has_el_r and has_el_ts:
        warnings = []
        if has_gibbs_ts and not has_gibbs_r:
            warnings.append("TS Gibbs free energy was available but reactant Gibbs was missing; selected consistent electronic energy basis.")
        elif has_gibbs_r and not has_gibbs_ts:
            warnings.append("Reactant Gibbs free energy was available but TS Gibbs was missing; selected consistent electronic energy basis.")

        return {
            "basis": "electronic",
            "temperature_k": temperature_k,
            "reactant": reactant_el,
            "ts": ts_el,
            "product": product_el if has_el_p else None,
            "barrier_symbol": "ΔE‡",
            "reaction_symbol": "ΔE_rxn",
            "units": "kcal/mol",
            "is_valid": True,
            "warnings": warnings,
        }

    # Inconsistent or missing
    return {
        "basis": "inconsistent",
        "temperature_k": temperature_k,
        "reactant": None,
        "ts": None,
        "product": None,
        "barrier_symbol": None,
        "reaction_symbol": None,
        "units": "kcal/mol",
        "is_valid": False,
        "warnings": ["Incompatible or missing energy bases between reactant and transition state (INV-006)."],
    }


def parse_orca_scan_output(
    output_text_or_path: Union[str, Path],
    work_dir: Optional[Union[str, Path]] = None
) -> ScanResult:
    """Parses ORCA relaxed surface scan results from output text or file."""
    if isinstance(output_text_or_path, Path) or (isinstance(output_text_or_path, str) and os.path.exists(output_text_or_path)):
        p = Path(output_text_or_path)
        content = p.read_text(encoding="utf-8", errors="replace")
        directory = p.parent if work_dir is None else Path(work_dir)
        base_stem = p.stem
    else:
        content = str(output_text_or_path)
        directory = Path(work_dir) if work_dir else Path(".")
        base_stem = "scan"

    points: List[ScanPoint] = []

    # Locate RELAXED SURFACE SCAN RESULTS block
    scan_block_m = re.search(r"RELAXED SURFACE SCAN RESULTS.*?-+\s+(.*?)(?=\n\n\n|\n---|\Z)", content, re.DOTALL)
    if scan_block_m:
        block = scan_block_m.group(1)
        # Match 'Actual Energy' subsection
        energy_sect = re.search(r"The Calculated Surface using the 'Actual Energy'\s+(.*?)(?=\nThe Calculated|\Z)", block, re.DOTALL)
        target_lines = energy_sect.group(1).strip().splitlines() if energy_sect else []

        if not target_lines:
            # Fallback to any pairs of float numbers
            target_lines = [ln for ln in block.splitlines() if re.match(r"^\s*([-\d.]+)\s+([-\d.]+)", ln.strip())]

        initial_energy = None
        for step_idx, line in enumerate(target_lines, start=1):
            m = re.match(r"^\s*([-\d.]+)\s+([-\d.]+)", line.strip())
            if m:
                coord_val = float(m.group(1))
                energy_eh = float(m.group(2))
                if initial_energy is None:
                    initial_energy = energy_eh
                rel_kcal = (energy_eh - initial_energy) * HARTREE_TO_KCAL

                # Check for corresponding step xyz file, e.g. <stem>.001.xyz
                step_xyz = directory / f"{base_stem}.{step_idx:03d}.xyz"
                if not step_xyz.exists():
                    # Check without stem prefix
                    step_xyz_alt = list(directory.glob(f"*.{step_idx:03d}.xyz"))
                    step_xyz = step_xyz_alt[0] if step_xyz_alt else None
                else:
                    step_xyz = step_xyz if step_xyz.exists() else None

                points.append(ScanPoint(
                    step=step_idx,
                    coordinate_value=coord_val,
                    energy_hartree=energy_eh,
                    relative_energy_kcal=rel_kcal,
                    xyz_path=step_xyz
                ))

    # If scan block was not found in stdout, check for allxyz/xyzall trajectory or parse step blocks
    if not points:
        allxyz_file = directory / f"{base_stem}.allxyz"
        if not allxyz_file.exists():
            allxyz_file = directory / f"{base_stem}.xyzall"

        if allxyz_file.exists():
            try:
                lines = allxyz_file.read_text(encoding="utf-8", errors="replace").splitlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.isdigit():
                        n_atoms = int(line)
                        if i + 1 < len(lines):
                            hdr = lines[i + 1]
                            m_e = re.search(r"E\s+([-\d.]+)", hdr)
                            m_step = re.search(r"Step\s+(\d+)", hdr)
                            step_idx = int(m_step.group(1)) if m_step else (len(points) + 1)
                            energy_eh = float(m_e.group(1)) if m_e else None

                            coords: List[List[float]] = []
                            for k in range(n_atoms):
                                if i + 2 + k < len(lines):
                                    parts = lines[i + 2 + k].split()
                                    if len(parts) >= 4:
                                        coords.append([float(x) for x in parts[1:4]])

                            step_xyz = directory / f"{base_stem}.{step_idx:03d}.xyz"
                            if not step_xyz.exists():
                                step_xyz_alt = list(directory.glob(f"*.{step_idx:03d}.xyz"))
                                step_xyz = step_xyz_alt[0] if step_xyz_alt else None

                            coord_val = 0.0
                            if len(coords) >= 32:
                                c1 = coords[31]
                                c2 = coords[10]
                                coord_val = round(math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))), 3)

                            if energy_eh is not None:
                                points.append(ScanPoint(
                                    step=step_idx,
                                    coordinate_value=coord_val,
                                    energy_hartree=energy_eh,
                                    relative_energy_kcal=0.0,
                                    xyz_path=step_xyz
                                ))
                        i += 2 + n_atoms
                    else:
                        i += 1
            except Exception:
                points = []

    # If still not found, fallback to parsing step blocks in stdout
    if not points:
        steps_raw = re.split(r"RELAXED SURFACE SCAN STEP\s+(\d+)", content)
        for idx in range(1, len(steps_raw), 2):
            s_num = int(steps_raw[idx])
            block = steps_raw[idx + 1]

            m_bond = re.search(r"Bond\s*\(\s*\d+,\s*\d+\)\s*:\s*([-\d.]+)", block)
            coord_val = round(float(m_bond.group(1)), 3) if m_bond else 0.0

            energies = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", block)
            if energies:
                final_e = float(energies[-1])
                step_xyz = directory / f"{base_stem}.{s_num:03d}.xyz"
                if not step_xyz.exists():
                    step_xyz_alt = list(directory.glob(f"*.{s_num:03d}.xyz"))
                    step_xyz = step_xyz_alt[0] if step_xyz_alt else None

                points.append(ScanPoint(
                    step=s_num,
                    coordinate_value=coord_val,
                    energy_hartree=final_e,
                    relative_energy_kcal=0.0,
                    xyz_path=step_xyz
                ))

    if not points:
        return ScanResult(converged=False, raw_output=content[:500])

    # Recompute relative energies with respect to first point
    e0 = points[0].energy_hartree
    for pt in points:
        pt.relative_energy_kcal = (pt.energy_hartree - e0) * HARTREE_TO_KCAL

    # Find maximum energy point (TS guess)
    max_pt = max(points, key=lambda pt: pt.energy_hartree)
    min_pt_before_max = min([pt for pt in points if pt.step <= max_pt.step], key=lambda pt: pt.energy_hartree, default=points[0])
    barrier_est = (max_pt.energy_hartree - min_pt_before_max.energy_hartree) * HARTREE_TO_KCAL
    rxn_est = (points[-1].energy_hartree - points[0].energy_hartree) * HARTREE_TO_KCAL if len(points) >= 2 else None

    return ScanResult(
        points=points,
        max_energy_step=max_pt.step,
        max_energy_hartree=max_pt.energy_hartree,
        barrier_estimate_kcal=round(barrier_est, 2),
        delta_e_scan_activation_kcal=round(barrier_est, 2),
        delta_e_scan_rxn_kcal=round(rxn_est, 2) if rxn_est is not None else None,
        ts_guess_coord_value=max_pt.coordinate_value,
        ts_guess_xyz=max_pt.xyz_path,
        converged=True,
        raw_output=content
    )


def parse_orca_ts_output(
    output_text_or_path: Union[str, Path],
    property_file_path: Optional[Union[str, Path]] = None,
    name: str = "Transition_State"
) -> TSVerificationResult:
    """Parses ORCA OptTS and Frequency calculations to verify first-order saddle point."""
    if isinstance(output_text_or_path, (str, Path)) and os.path.exists(str(output_text_or_path)):
        calc = parse_orca_results(output_text_or_path, name=name)
    else:
        # Construct temporary directory for in-memory text parsing
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / f"{name}.out"
            out_file.write_text(str(output_text_or_path), encoding="utf-8")
            if property_file_path and os.path.exists(str(property_file_path)):
                shutil.copyfile(str(property_file_path), Path(tmpdir) / f"{name}.property.txt")
            calc = parse_orca_results(out_file, name=name)

    # Distinguish genuine imaginary vibrational modes from low-frequency rotational/translational noise
    # Standard threshold: nu < -15 cm^-1
    genuine_imag = [f for f in calc.frequencies if f < -15.0]

    # Valid first-order saddle point requires BOTH optimization convergence and exactly 1 genuine imaginary frequency
    is_valid_ts = bool(calc.converged and len(genuine_imag) == 1)

    if not calc.converged:
        summary_str = f"Invalid TS: Optimization did not converge ({len(genuine_imag)} imaginary mode(s))"
    elif is_valid_ts:
        summary_str = f"Valid 1st-Order TS (converged, nu_i = {genuine_imag[0]:.1f} cm^-1)"
    else:
        summary_str = f"Invalid TS: {len(genuine_imag)} imaginary mode(s) {genuine_imag}"

    return TSVerificationResult(
        name=name,
        converged=calc.converged,
        n_imaginary_frequencies=len(genuine_imag),
        imaginary_modes=genuine_imag,
        all_frequencies=calc.frequencies,
        is_valid_first_order_saddle_point=is_valid_ts,
        transition_vector_summary=summary_str,
        electronic_energy_hartree=calc.el_energy,
        zpe_hartree=calc.zpe,
        enthalpy_hartree=calc.enthalpy,
        entropy_hartree=calc.entropy,
        gibbs_free_energy_hartree=calc.gibbs_energy,
        temperature_k=calc.temperature,
        coordinates_angstrom=calc.coordinates_angstrom,
        atomic_symbols=calc.atomic_symbols
    )


def compute_eyring_rate_constant(
    delta_g_dagger_kcal: float,
    temperature_k: float = 310.15,
    kappa: float = 1.0,
) -> Tuple[float, str]:
    """Calculates the Eyring chemical rate constant and associated half-life.

    Parameters
    ----------
    delta_g_dagger_kcal : float
        Activation free energy barrier in kcal/mol (Delta G‡).
    temperature_k : float
        Temperature in Kelvin (default 310.15 K, physiological).
    kappa : float
        Transmission coefficient (default 1.0).

    Returns
    -------
    rate_k : float
        Chemical rate constant in s^-1.
    half_life_str : str
        Formatted estimated half-life string.
    """
    rt = R_GAS_KCAL * temperature_k
    eyring_prefactor = kappa * KB_OVER_H * temperature_k

    if delta_g_dagger_kcal < 0.0:
        return eyring_prefactor, "< 1 ps (Barrierless)"

    exponent = -delta_g_dagger_kcal / rt
    if exponent < -700.0:
        return 0.0, "> 1000 years"

    rate_k = eyring_prefactor * math.exp(exponent)
    if rate_k > 0:
        half_life_sec = math.log(2) / rate_k
        if half_life_sec < 1e-3:
            half_life_str = f"{half_life_sec * 1e6:.1f} microseconds"
        elif half_life_sec < 1.0:
            half_life_str = f"{half_life_sec * 1e3:.1f} milliseconds"
        elif half_life_sec < 60.0:
            half_life_str = f"{half_life_sec:.1f} seconds"
        elif half_life_sec < 3600.0:
            half_life_str = f"{half_life_sec / 60.0:.1f} minutes"
        elif half_life_sec < 86400.0:
            half_life_str = f"{half_life_sec / 3600.0:.1f} hours"
        elif half_life_sec < 365.25 * 86400.0:
            half_life_str = f"{half_life_sec / 86400.0:.1f} days"
        else:
            half_life_str = f"{half_life_sec / (365.25 * 86400.0):.1f} years"
    else:
        half_life_str = "> 1000 years"

    return rate_k, half_life_str


def compute_reaction_profile(
    reactants_gibbs: Optional[float] = None,
    ts_gibbs: Optional[float] = None,
    product_gibbs: Optional[float] = None,
    reactants_electronic: Optional[float] = None,
    ts_electronic: Optional[float] = None,
    product_electronic: Optional[float] = None,
    temperature_k: float = 298.15,
    is_first_order_ts: bool = True,
    raise_on_inconsistent: bool = False,
) -> ReactionEnergyProfile:
    """Calculates thermodynamic and kinetic properties from ground state, TS, and product energies.

    Enforces INV-006:
    Energy basis must be homogeneous across compared species (Route B: Gibbs free energies,
    Route A: Electronic energies). Mixed energy bases (e.g. E_reactants with G_TS) are strictly rejected.
    Eyring rate constant and half-life are evaluated ONLY when homogeneous Gibbs free energies are available.
    """
    basis_sel = select_consistent_energy_basis(
        reactant_el=reactants_electronic,
        ts_el=ts_electronic,
        product_el=product_electronic,
        reactant_gibbs=reactants_gibbs,
        ts_gibbs=ts_gibbs,
        product_gibbs=product_gibbs,
        temperature_k=temperature_k,
    )

    if not basis_sel["is_valid"]:
        msg = f"Cannot compute reaction profile: {'; '.join(basis_sel.get('warnings', ['Inconsistent energy basis (INV-006)']))}"
        if raise_on_inconsistent:
            raise ValueError(msg)
        return ReactionEnergyProfile(
            reactants_gibbs=reactants_gibbs,
            ts_gibbs=ts_gibbs,
            product_gibbs=product_gibbs,
            reactants_electronic=reactants_electronic,
            ts_electronic=ts_electronic,
            product_electronic=product_electronic,
            energy_basis="inconsistent",
            barrier_symbol="N/A",
            reaction_energy_symbol="N/A",
            delta_g_activation_kcal=None,
            delta_g_reaction_kcal=None,
            delta_e_activation_kcal=None,
            delta_e_reaction_kcal=None,
            delta_e_scan_activation_kcal=None,
            delta_e_scan_rxn_kcal=None,
            rate_constant_s=None,
            estimated_half_life_str="Not available",
            kinetic_feasibility="Not evaluated (inconsistent energy basis)",
            temperature_k=temperature_k,
            is_first_order_ts=is_first_order_ts,
            notes="Mixed thermodynamic quantities cannot produce ΔG‡ or ΔG_rxn (INV-006).",
            warnings=list(basis_sel.get("warnings", [])) + ["Mixed thermodynamic quantities cannot produce ΔG‡."],
        )

    basis = basis_sel["basis"]
    r_val = basis_sel["reactant"]
    ts_val = basis_sel["ts"]
    p_val = basis_sel["product"]
    barrier_symbol = basis_sel["barrier_symbol"]
    reaction_symbol = basis_sel["reaction_symbol"]
    warnings = list(basis_sel.get("warnings", []))

    act_barrier_eh = ts_val - r_val
    act_barrier_kcal = act_barrier_eh * HARTREE_TO_KCAL

    rxn_energy_kcal = None
    if p_val is not None:
        rxn_energy_kcal = (p_val - r_val) * HARTREE_TO_KCAL

    notes_list = []
    if not is_first_order_ts:
        notes_list.append("Warning: Structure is not a strictly confirmed first-order saddle point.")

    if rxn_energy_kcal is not None:
        if rxn_energy_kcal > 0.0:
            notes_list.append(f"Thermodynamics: Endergonic ({reaction_symbol} = +{rxn_energy_kcal:.2f} kcal/mol, unfavorable product equilibrium).")
        else:
            notes_list.append(f"Thermodynamics: Exergonic ({reaction_symbol} = {rxn_energy_kcal:.2f} kcal/mol, favorable covalent adduct).")

    notes = " ".join(notes_list)

    if basis == "gibbs":
        # Eyring transition state theory rate constant ONLY when Gibbs free energy is available
        rate_k, half_life_str = compute_eyring_rate_constant(
            act_barrier_kcal,
            temperature_k=temperature_k,
            kappa=1.0
        )

        if act_barrier_kcal < 0.0:
            feasibility = "Instantaneous / Barrierless"
        elif act_barrier_kcal <= 18.0:
            feasibility = "Very Rapid Predicted Chemical Step"
        elif act_barrier_kcal <= 22.0:
            feasibility = "Rapid Predicted Chemical Step"
        elif act_barrier_kcal <= 25.0:
            feasibility = "Moderate Predicted Chemical Rate"
        else:
            feasibility = "Slow Predicted Chemical Step (High Barrier)"

        return ReactionEnergyProfile(
            reactants_gibbs=r_val,
            ts_gibbs=ts_val,
            product_gibbs=p_val,
            reactants_electronic=reactants_electronic,
            ts_electronic=ts_electronic,
            product_electronic=product_electronic,
            energy_basis="gibbs",
            barrier_symbol="ΔG‡",
            reaction_energy_symbol="ΔG_rxn",
            delta_g_activation_kcal=round(act_barrier_kcal, 2),
            delta_g_reaction_kcal=round(rxn_energy_kcal, 2) if rxn_energy_kcal is not None else None,
            delta_e_activation_kcal=None,
            delta_e_reaction_kcal=None,
            delta_e_scan_activation_kcal=None,
            delta_e_scan_rxn_kcal=None,
            rate_constant_s=rate_k,
            estimated_half_life_str=half_life_str,
            kinetic_feasibility=feasibility,
            temperature_k=temperature_k,
            is_first_order_ts=is_first_order_ts,
            notes=notes,
            warnings=warnings,
        )
    else:
        # Route A: Electronic energies. Forbid Eyring kinetics per INV-006.
        warnings.append(
            "Eyring rate constant and half-life are not computed from electronic energies; "
            "homogeneous Gibbs free energies are required (INV-006)."
        )
        return ReactionEnergyProfile(
            reactants_gibbs=reactants_gibbs,
            ts_gibbs=ts_gibbs,
            product_gibbs=product_gibbs,
            reactants_electronic=r_val,
            ts_electronic=ts_val,
            product_electronic=p_val,
            energy_basis="electronic",
            barrier_symbol="ΔE‡",
            reaction_energy_symbol="ΔE_rxn",
            delta_g_activation_kcal=None,
            delta_g_reaction_kcal=None,
            delta_e_activation_kcal=round(act_barrier_kcal, 2),
            delta_e_reaction_kcal=round(rxn_energy_kcal, 2) if rxn_energy_kcal is not None else None,
            delta_e_scan_activation_kcal=round(act_barrier_kcal, 2),
            delta_e_scan_rxn_kcal=round(rxn_energy_kcal, 2) if rxn_energy_kcal is not None else None,
            rate_constant_s=None,
            estimated_half_life_str="Not available (requires Gibbs free energy)",
            kinetic_feasibility=f"Electronic Barrier: {round(act_barrier_kcal, 2):.2f} kcal/mol (Eyring kinetics pending Gibbs calculation)",
            temperature_k=temperature_k,
            is_first_order_ts=is_first_order_ts,
            notes=notes,
            warnings=warnings,
        )


# Backward-compatible alias
build_reaction_energy_profile = compute_reaction_profile


def prepare_ts_workflow_directory(
    cluster: QMCluster,
    output_dir: Union[str, Path],
    method: str = "r2SCAN-3c",
    solvent: Optional[str] = "Water",
    scan_start: Optional[float] = None,
    scan_end: float = 1.45,
    scan_steps: int = 18,
    nprocs: int = 4,
    maxcore_mb: int = 2000,
    recalc_hess: int = 25,
) -> dict:
    """Creates directory structure, ORCA scan input, and runner scripts for Tier 4 TS modeling."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if scan_start is None:
        if cluster.nucleophile_idx is not None and cluster.electrophile_idx is not None:
            p1 = cluster.atoms[cluster.nucleophile_idx].coords
            p2 = cluster.atoms[cluster.electrophile_idx].coords
            scan_start = round(_euclidean_distance(p1, p2), 2)
        else:
            scan_start = 3.30

    # 1. Write initial cluster geometry
    xyz_path = cluster.write_xyz(out_path / "00_initial_cluster.xyz")

    # 2. Write coordinate scan input
    scan_inp_path = cluster.write_orca_input(
        out_path / "01_scan.inp",
        job_type="scan",
        method=method,
        solvent=solvent,
        scan_start=scan_start,
        scan_end=scan_end,
        scan_steps=scan_steps,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
    )

    # 3. Write template OptTS input (to be filled with TS guess coordinates after scan)
    optts_template_path = out_path / "02_optts_template.inp"
    optts_template_content = cluster.to_orca_input(
        job_type="optts",
        method=method,
        solvent=solvent,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
        recalc_hess=recalc_hess,
    )
    optts_template_path.write_text(optts_template_content, encoding="utf-8")

    # 4. Write template Adduct Optimization input (to be filled with product coordinates after scan)
    adduct_template_path = out_path / "03_adduct_opt_template.inp"
    adduct_template_content = cluster.to_orca_input(
        job_type="opt",
        method=method,
        solvent=solvent,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
    )
    adduct_template_path.write_text(adduct_template_content, encoding="utf-8")

    # 5. Write bash execution script for automated execution
    run_sh_path = out_path / "run_tier4_ts.sh"
    run_script = f"""#!/usr/bin/env bash
set -e
echo "=========================================================="
echo " SharK Tier 4: Transition State & Activation Energy Workflow"
echo " Cluster: {cluster.name} ({cluster.model_type} model)"
echo " Target: {cluster.target_residue} | Method: {method} ({solvent or 'gas'})"
echo "=========================================================="

if [ -n "$SHARK_ORCA" ] && [ -x "$SHARK_ORCA" ]; then
    ORCA_BIN="$SHARK_ORCA"
elif [ -n "$ORCA_PATH" ] && [ -x "$ORCA_PATH" ]; then
    ORCA_BIN="$ORCA_PATH"
else
    ORCA_BIN=$(which orca 2>/dev/null || true)
fi

if [ -z "$ORCA_BIN" ]; then
    echo "Error: ORCA executable not found. Please set SHARK_ORCA or ensure 'orca' is in PATH."
    exit 1
fi
# Resolve symlink to realpath for ORCA 6
ORCA_REAL=$(readlink -f "$ORCA_BIN")

echo "[Step 1/4] Running Relaxed Coordinate Scan..."
"$ORCA_REAL" 01_scan.inp > 01_scan.out

echo "[Step 2/4] Analyzing Scan Trajectory and Extracting TS Guess..."
# SharK python hook extracts TS guess and creates 02_optts.inp
python3 -c "
from shark.analysis.transition_state import parse_orca_scan_output
from shark.analysis.qm_cluster import QMCluster
from pathlib import Path
res = parse_orca_scan_output('01_scan.out', work_dir='.')
print(res.summary)
if res.ts_guess_xyz and res.ts_guess_xyz.exists():
    print(f'Using TS guess from: {{res.ts_guess_xyz}}')
    xyz_lines = res.ts_guess_xyz.read_text().splitlines()[2:]
    tmpl = Path('02_optts_template.inp').read_text()
    header = tmpl.split('* xyz')[0]
    coords_str = '\\n'.join(['  ' + ln for ln in xyz_lines if ln.strip()])
    new_inp = f'{{header}}* xyz {cluster.charge} {cluster.effective_multiplicity}\\n{{coords_str}}\\n*'
    Path('02_optts.inp').write_text(new_inp)
else:
    print('Warning: Specific step XYZ not found, using template coordinates')
    Path('02_optts.inp').write_text(Path('02_optts_template.inp').read_text())
"

echo "[Step 3/4] Running Saddle Point Optimization & Frequency Calculation (! OptTS Freq)..."
"$ORCA_REAL" 02_optts.inp > 02_optts.out

echo "[Step 4/4] Preparing and Running Covalent Product Adduct Optimization..."
python3 -c "
from shark.analysis.transition_state import parse_orca_scan_output
from pathlib import Path
res = parse_orca_scan_output('01_scan.out', work_dir='.')
if res.points:
    last_pt = res.points[-1]
    if last_pt.xyz_path and last_pt.xyz_path.exists():
        xyz_lines = last_pt.xyz_path.read_text().splitlines()[2:]
        tmpl = Path('03_adduct_opt_template.inp').read_text()
        header = tmpl.split('* xyz')[0]
        coords_str = '\\n'.join(['  ' + ln for ln in xyz_lines if ln.strip()])
        new_inp = f'{{header}}* xyz {cluster.charge} {cluster.effective_multiplicity}\\n{{coords_str}}\\n*'
        Path('03_adduct_opt.inp').write_text(new_inp)
        print(f'Adduct product input prepared from step #{{last_pt.step}} (d={{last_pt.coordinate_value:.2f}} A)')
"

if [ -f 03_adduct_opt.inp ]; then
    "$ORCA_REAL" 03_adduct_opt.inp > 03_adduct_opt.out
fi

echo "Tier 4 workflow (Scan, TS, and Adduct) completed successfully."
"""
    run_sh_path.write_text(run_script, encoding="utf-8")
    run_sh_path.chmod(0o755)

    return {
        "output_dir": out_path,
        "initial_xyz": xyz_path,
        "scan_inp": scan_inp_path,
        "optts_template": optts_template_path,
        "run_script": run_sh_path,
        "cluster": cluster,
    }

