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

from .qm_cluster import QMCluster, extract_qm_cluster
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
    ts_guess_coord_value: float = 0.0
    ts_guess_xyz: Optional[Path] = None
    converged: bool = True
    raw_output: str = ""

    @property
    def summary(self) -> str:
        return (
            f"Coordinate Scan: {len(self.points)} steps | "
            f"TS Guess at Step {self.max_energy_step} (d={self.ts_guess_coord_value:.2f} A) | "
            f"Estimated Barrier: {self.barrier_estimate_kcal:.2f} kcal/mol"
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
    reactants_gibbs: float                  # Hartree
    ts_gibbs: float                         # Hartree
    product_gibbs: Optional[float] = None   # Hartree
    delta_g_activation_kcal: float = 0.0    # kcal/mol: G(TS) - G(Reactants)
    delta_g_reaction_kcal: Optional[float] = None  # kcal/mol: G(Product) - G(Reactants)
    rate_constant_s: float = 0.0            # s^-1
    estimated_half_life_str: str = ""
    kinetic_feasibility: str = ""
    temperature_k: float = 298.15
    is_first_order_ts: bool = True
    notes: str = ""

    @property
    def summary(self) -> str:
        prod_str = f", Delta G_rxn: {self.delta_g_reaction_kcal:.2f} kcal/mol" if self.delta_g_reaction_kcal is not None else ""
        return (
            f"Reaction Profile: Delta G‡ = {self.delta_g_activation_kcal:.2f} kcal/mol{prod_str} | "
            f"Feasibility: {self.kinetic_feasibility} (t1/2 ~ {self.estimated_half_life_str})"
        )


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

    # If scan block was not found, fallback to parsing step lines in stdout
    if not points:
        step_pattern = re.compile(r"RELAXED SURFACE SCAN STEP\s+(\d+).*?FINAL SINGLE POINT ENERGY\s+([-\d.]+)", re.DOTALL)
        matches = step_pattern.findall(content)
        initial_e = None
        for m in matches:
            s_num = int(m[0])
            s_eh = float(m[1])
            if initial_e is None:
                initial_e = s_eh
            points.append(ScanPoint(
                step=s_num,
                coordinate_value=0.0,
                energy_hartree=s_eh,
                relative_energy_kcal=(s_eh - initial_e) * HARTREE_TO_KCAL
            ))

    if not points:
        return ScanResult(converged=False, raw_output=content[:500])

    # Find maximum energy point (TS guess)
    max_pt = max(points, key=lambda pt: pt.energy_hartree)
    min_pt_before_max = min([pt for pt in points if pt.step <= max_pt.step], key=lambda pt: pt.energy_hartree, default=points[0])
    barrier_est = (max_pt.energy_hartree - min_pt_before_max.energy_hartree) * HARTREE_TO_KCAL

    return ScanResult(
        points=points,
        max_energy_step=max_pt.step,
        max_energy_hartree=max_pt.energy_hartree,
        barrier_estimate_kcal=round(barrier_est, 2),
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

    is_valid_ts = (len(genuine_imag) == 1)

    summary_str = (
        f"Valid 1st-Order TS (nu_i = {genuine_imag[0]:.1f} cm^-1)"
        if is_valid_ts
        else f"Invalid TS: {len(genuine_imag)} imaginary mode(s) {genuine_imag}"
    )

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


def compute_reaction_profile(
    reactants_gibbs: float,
    ts_gibbs: float,
    product_gibbs: Optional[float] = None,
    temperature_k: float = 298.15,
    is_first_order_ts: bool = True
) -> ReactionEnergyProfile:
    """Computes Eyring activation free energy and reaction kinetics.

    Parameters
    ----------
    reactants_gibbs : float
        Gibbs free energy of ground state reactants in Hartrees.
    ts_gibbs : float
        Gibbs free energy of the transition state in Hartrees.
    product_gibbs : float or None
        Gibbs free energy of the covalent adduct product in Hartrees.
    temperature_k : float
        Temperature in Kelvin (default 298.15 K).
    is_first_order_ts : bool
        Whether the TS Hessian was confirmed to have strictly 1 imaginary mode.
    """
    delta_g_act_eh = ts_gibbs - reactants_gibbs
    delta_g_act_kcal = delta_g_act_eh * HARTREE_TO_KCAL

    delta_g_rxn_kcal = None
    if product_gibbs is not None:
        delta_g_rxn_kcal = (product_gibbs - reactants_gibbs) * HARTREE_TO_KCAL

    # Eyring transition state theory rate constant
    # k = (k_B * T / h) * exp(-Delta G‡ / (R * T))
    rt = R_GAS_KCAL * temperature_k
    eyring_prefactor = KB_OVER_H * temperature_k

    if delta_g_act_kcal < 0.0:
        # Barrierless or ground state slightly above TS guess
        rate_k = eyring_prefactor
        half_life_str = "< 1 ps (Barrierless)"
        feasibility = "Instantaneous / Barrierless"
    else:
        exponent = -delta_g_act_kcal / rt
        if exponent < -700:
            rate_k = 0.0
            half_life_str = "> 1000 years"
            feasibility = "Infeasible (Extremely High Barrier)"
        else:
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

            # Categorize kinetic feasibility at physiological temperature
            if delta_g_act_kcal <= 18.0:
                feasibility = "Spontaneous / Rapid Reaction"
            elif delta_g_act_kcal <= 22.0:
                feasibility = "High Covalent Feasibility"
            elif delta_g_act_kcal <= 25.0:
                feasibility = "Moderate / Physiologically Feasible"
            else:
                feasibility = "Infeasible / High Barrier (Requires Acid/Base Catalysis)"

    notes = ""
    if not is_first_order_ts:
        notes = "Warning: Structure is not a strictly confirmed first-order saddle point."

    return ReactionEnergyProfile(
        reactants_gibbs=reactants_gibbs,
        ts_gibbs=ts_gibbs,
        product_gibbs=product_gibbs,
        delta_g_activation_kcal=round(delta_g_act_kcal, 2),
        delta_g_reaction_kcal=round(delta_g_rxn_kcal, 2) if delta_g_rxn_kcal is not None else None,
        rate_constant_s=rate_k,
        estimated_half_life_str=half_life_str,
        kinetic_feasibility=feasibility,
        temperature_k=temperature_k,
        is_first_order_ts=is_first_order_ts,
        notes=notes
    )


def prepare_ts_workflow_directory(
    cluster: QMCluster,
    output_dir: Union[str, Path],
    method: str = "r2SCAN-3c",
    solvent: Optional[str] = "Water",
    scan_start: float = 3.30,
    scan_end: float = 1.45,
    scan_steps: int = 18,
    nprocs: int = 4,
    maxcore_mb: int = 2000,
) -> dict:
    """Creates directory structure, ORCA scan input, and runner scripts for Tier 4 TS modeling."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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
    )
    optts_template_path.write_text(optts_template_content, encoding="utf-8")

    # 4. Write bash execution script for automated execution
    run_sh_path = out_path / "run_tier4_ts.sh"
    run_script = f"""#!/usr/bin/env bash
set -e
echo "=========================================================="
echo " SharK Tier 4: Transition State & Activation Energy Workflow"
echo " Cluster: {cluster.name} ({cluster.model_type} model)"
echo " Target: {cluster.target_residue} | Method: {method} ({solvent or 'gas'})"
echo "=========================================================="

ORCA_BIN=$(which orca 2>/dev/null || true)
if [ -z "$ORCA_BIN" ]; then
    if [ -f "/home/diego/bioinformatics/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca" ]; then
        ORCA_BIN="/home/diego/bioinformatics/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
    else
        echo "Error: ORCA executable not found in PATH."
        exit 1
    fi
fi
# Resolve symlink to realpath for ORCA 6
ORCA_REAL=$(readlink -f "$ORCA_BIN")

echo "[Step 1/3] Running Relaxed Coordinate Scan..."
"$ORCA_REAL" 01_scan.inp > 01_scan.out

echo "[Step 2/3] Analyzing Scan Trajectory and Extracting TS Guess..."
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
    # Replace coordinates
    header = tmpl.split('* xyz')[0]
    coords_str = '\n'.join(['  ' + ln for ln in xyz_lines if ln.strip()])
    new_inp = f'{{header}}* xyz {cluster.charge} {cluster.multiplicity}\n{{coords_str}}\n*'
    Path('02_optts.inp').write_text(new_inp)
else:
    print('Warning: Specific step XYZ not found, using template coordinates')
    Path('02_optts.inp').write_text(Path('02_optts_template.inp').read_text())
"

echo "[Step 3/3] Running Saddle Point Optimization & Frequency Calculation (! OptTS Freq)..."
"$ORCA_REAL" 02_optts.inp > 02_optts.out

echo "Transition state workflow completed successfully."
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
