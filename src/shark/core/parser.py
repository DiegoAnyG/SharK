"""Parser for ORCA 6 calculation outputs (.property.txt and .out)."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class CalculationResult:
    """Structured container for quantum chemical calculation results."""
    name: str
    converged: bool = False
    optimization_converged: Optional[bool] = None
    el_energy: float = 0.0          # Electronic energy in Hartrees (Eh)
    zpe: float = 0.0                # Zero-point vibrational energy in Hartrees
    enthalpy: float = 0.0           # Enthalpy (H) in Hartrees
    gibbs_energy: float = 0.0       # Gibbs free energy (G) in Hartrees
    entropy: float = 0.0            # Entropy (S) in Hartrees
    temperature: float = 298.15     # Temperature in Kelvin
    frequencies: List[float] = field(default_factory=list)      # All frequencies (cm^-1)
    ir_frequencies: List[float] = field(default_factory=list)   # Frequencies corresponding to IR intensities
    ir_intensities: List[float] = field(default_factory=list)   # km/mol
    imaginary_frequencies: List[float] = field(default_factory=list)
    dipole_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0) # a.u.
    dipole_magnitude: float = 0.0   # Debye
    homo_energy: Optional[float] = None # eV
    lumo_energy: Optional[float] = None # eV
    homo_lumo_gap: Optional[float] = None # eV
    loewdin_charges: List[float] = field(default_factory=list)
    mulliken_charges: List[float] = field(default_factory=list)
    atomic_numbers: List[int] = field(default_factory=list)
    atomic_symbols: List[str] = field(default_factory=list)
    coordinates_angstrom: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def is_stationary_minimum(self) -> bool:
        """True if the structure converged and has zero imaginary vibrational frequencies."""
        return (
            self.converged
            and self.optimization_converged is True
            and bool(self.frequencies)
            and len(self.imaginary_frequencies) == 0
        )


def _parse_property_file(prop_path: Path, result: CalculationResult) -> None:
    """Extract structured thermochemistry and properties from ORCA .property.txt."""
    content = prop_path.read_text(encoding="utf-8", errors="ignore")

    # Check termination status
    if "NORMAL TERMINATION" in content:
        result.converged = True

    # Parse Thermochemistry section
    thermo_match = re.search(r"\$THERMOCHEMISTRY_Energies\s+(.*?)\$End", content, re.DOTALL)
    if thermo_match:
        block = thermo_match.group(1)
        
        # Temperature
        temp_m = re.search(r"&temperature\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if temp_m:
            result.temperature = float(temp_m.group(1))

        # Electronic Energy
        el_m = re.search(r"&elEnergy\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if el_m:
            result.el_energy = float(el_m.group(1))

        # ZPE
        zpe_m = re.search(r"&zpe\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if zpe_m:
            result.zpe = float(zpe_m.group(1))

        # Enthalpy H
        h_m = re.search(r"&enthalpyH\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if h_m:
            result.enthalpy = float(h_m.group(1))

        # Entropy S
        s_m = re.search(r"&entropyS\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if s_m:
            result.entropy = float(s_m.group(1))

        # Gibbs Free Energy G
        g_m = re.search(r"&freeEnergyG\s+\[.*?\]\s+([-\d.eE+]+)", block)
        if g_m:
            result.gibbs_energy = float(g_m.group(1))

        # Frequencies array
        freq_block_m = re.search(r"&FREQ\s+\[.*?\]\s+(.*?)(?=&[a-zA-Z]|\$End)", block, re.DOTALL)
        if freq_block_m:
            freq_lines = freq_block_m.group(1).strip().splitlines()
            freqs = []
            for line in freq_lines:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        freqs.append(float(parts[1]))
                    except ValueError:
                        pass
            if freqs:
                result.frequencies = freqs
                result.imaginary_frequencies = [f for f in freqs if f < -1e-3]

    # Geometry & Coordinates
    geom_m = re.search(r"\$Geometry\s+(.*?)\$End", content, re.DOTALL)
    if geom_m:
        coords_match = re.search(r"&CartesianCoordinates.*?\[.*?Units\s+\"([^\"]+)\"\s*\]\s+(.*?)(?=&|\Z)", geom_m.group(1), re.DOTALL)
        if coords_match:
            unit = coords_match.group(1).strip().lower()
            scale = 0.529177210903 if "bohr" in unit else 1.0
            symbols = []
            coords = []
            for line in coords_match.group(2).strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].isalpha():
                    symbols.append(parts[0])
                    try:
                        coords.append((float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale))
                    except ValueError:
                        pass
            if symbols:
                result.atomic_symbols = symbols
                result.coordinates_angstrom = coords

    # Loewdin Population Analysis
    loewdin_m = re.search(r"\$SCF_Loewdin_Population_Analysis\s+(.*?)\$End", content, re.DOTALL)
    if loewdin_m:
        block = loewdin_m.group(1)
        atno_m = re.search(r"&ATNO\s+\[.*?\]\s*(.*?)(?=&|\$End)", block, re.DOTALL)
        if atno_m:
            atnos = []
            for line in atno_m.group(1).strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    atnos.append(int(parts[1]))
            if atnos:
                result.atomic_numbers = atnos

        chg_m = re.search(r"&AtomicCharges\s+\[.*?\]\s*(.*?)(?=&|\$End)", block, re.DOTALL)
        if chg_m:
            charges = []
            for line in chg_m.group(1).strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        charges.append(float(parts[1]))
                    except ValueError:
                        pass
            if charges:
                result.loewdin_charges = charges

    # Mulliken Population Analysis
    mulliken_m = re.search(r"\$SCF_Mulliken_Population_Analysis\s+(.*?)\$End", content, re.DOTALL)
    if mulliken_m:
        block = mulliken_m.group(1)
        chg_m = re.search(r"&AtomicCharges\s+\[.*?\]\s*(.*?)(?=&|\$End)", block, re.DOTALL)
        if chg_m:
            charges = []
            for line in chg_m.group(1).strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        charges.append(float(parts[1]))
                    except ValueError:
                        pass
            if charges:
                result.mulliken_charges = charges


def _parse_out_file(out_path: Path, result: CalculationResult) -> None:
    """Extract IR intensities, dipole magnitude, frontier orbitals, and thermochemistry from ORCA .out."""
    content = out_path.read_text(encoding="utf-8", errors="ignore")

    # Check termination status in .out
    if "ORCA TERMINATED NORMALLY" in content or "NORMAL TERMINATION" in content:
        result.converged = True

    if "THE OPTIMIZATION HAS CONVERGED" in content:
        result.optimization_converged = True
    elif "optimization did not converge" in content.lower():
        result.optimization_converged = False

    # Parse Electronic Energy fallback from .out
    if result.el_energy == 0.0:
        sp_m = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", content)
        if sp_m:
            result.el_energy = float(sp_m[-1])

    # Parse Gibbs Free Energy fallback from .out
    if result.gibbs_energy == 0.0:
        gibbs_m = re.search(r"Final Gibbs free energy\s+\.\.\.\s+([-\d.]+)\s+Eh", content)
        if gibbs_m:
            result.gibbs_energy = float(gibbs_m.group(1))

    # Parse VIBRATIONAL FREQUENCIES block
    if not result.frequencies:
        vf_match = re.search(r"VIBRATIONAL FREQUENCIES\s+-+\s+(.*?)(?=\n\n[A-Z]|\n---|\Z)", content, re.DOTALL)
        if vf_match:
            vf_freqs = []
            for line in vf_match.group(1).strip().splitlines():
                m = re.match(r"^\s*\d+:\s+([-\d.]+)\s+cm\*\*-1", line.strip())
                if m:
                    vf_freqs.append(float(m.group(1)))
            if vf_freqs:
                result.frequencies = vf_freqs
                result.imaginary_frequencies = [f for f in vf_freqs if f < -1e-3]

    # Parse IR Spectrum table
    # Example format:
    #   Mode   freq       eps      Int      T**2         TX        TY        TZ
    #   6:     54.57   0.000760    3.84  ...
    ir_match = re.search(r"IR SPECTRUM\s+-+\s+Mode\s+freq.*?-+\s+(.*?)(?=\*|\n\n|\Z)", content, re.DOTALL)
    if ir_match:
        intensities = []
        parsed_freqs = []
        for line in ir_match.group(1).strip().splitlines():
            line = line.strip()
            # Look for lines starting with mode: e.g. "6: 54.57 ... 3.84"
            m = re.match(r"^\s*(\d+):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                freq = float(m.group(2))
                intensity = float(m.group(4))
                parsed_freqs.append(freq)
                intensities.append(intensity)
        if intensities:
            result.ir_intensities = intensities
            result.ir_frequencies = parsed_freqs
            # If frequencies were not set yet, set them from the IR table
            if not result.frequencies:
                result.frequencies = parsed_freqs
                result.imaginary_frequencies = [f for f in parsed_freqs if f < -1e-3]

    # Parse Dipole Moment (Debye)
    # Total Dipole Moment    :     -1.647830419      -0.244214854       0.153205970
    # Magnitude (Debye)      :      4.252070284
    dip_vec_m = re.search(r"Total Dipole Moment\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", content)
    if dip_vec_m:
        result.dipole_vector = (
            float(dip_vec_m.group(1)),
            float(dip_vec_m.group(2)),
            float(dip_vec_m.group(3)),
        )

    dip_mag_m = re.search(r"Magnitude \(Debye\)\s+:\s+([-\d.]+)", content)
    if dip_mag_m:
        result.dipole_magnitude = float(dip_mag_m.group(1))

    # Parse Frontier Orbitals (last ORBITAL ENERGIES block)
    # NO   OCC          E(Eh)            E(eV)
    # 45   2.0000      -0.233052        -6.3417
    # 46   0.0000      -0.106384        -2.8949
    orbital_blocks = list(re.finditer(r"ORBITAL ENERGIES\s+-+\s+NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s+(.*?)(?=\*|\n\n|MULLIKEN)", content, re.DOTALL))
    if orbital_blocks:
        last_orb_block = orbital_blocks[-1].group(1)
        homo_ev = None
        lumo_ev = None
        for line in last_orb_block.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    occ = float(parts[1])
                    ev = float(parts[3])
                    if occ > 0.5:
                        homo_ev = ev  # Last occupied orbital
                    elif occ <= 0.5 and lumo_ev is None:
                        lumo_ev = ev  # First unoccupied orbital
                except ValueError:
                    continue
        if homo_ev is not None:
            result.homo_energy = homo_ev
        if lumo_ev is not None:
            result.lumo_energy = lumo_ev
        if homo_ev is not None and lumo_ev is not None:
            result.homo_lumo_gap = lumo_ev - homo_ev

    # Fallback atomic charges from .out if not found in property file
    if not result.mulliken_charges:
        mull_m = re.search(r"MULLIKEN ATOMIC CHARGES\s+-+\s+(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
        if mull_m:
            m_charges = []
            for line in mull_m.group(1).strip().splitlines():
                m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-\d.eE+]+)", line)
                if m:
                    m_charges.append(float(m.group(3)))
            if m_charges:
                result.mulliken_charges = m_charges

    if not result.loewdin_charges:
        loew_m = re.search(r"LOEWDIN ATOMIC CHARGES\s+-+\s+(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
        if loew_m:
            l_charges = []
            for line in loew_m.group(1).strip().splitlines():
                m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-\d.eE+]+)", line)
                if m:
                    l_charges.append(float(m.group(3)))
            if l_charges:
                result.loewdin_charges = l_charges


def parse_orca_results(base_path: str | Path, name: Optional[str] = None) -> CalculationResult:
    """
    Parse ORCA calculation files associated with a given base path.
    Supports supplying either the base stem, the .out file, or the .property.txt file.
    """
    p = Path(base_path)
    if p.suffix in [".out", ".txt", ".property"]:
        stem = p.stem
        if stem.endswith(".property"):
            stem = stem[:-9]
        directory = p.parent
    else:
        stem = p.name
        directory = p.parent if str(p.parent) != "" else Path(".")

    mol_name = name or stem
    result = CalculationResult(name=mol_name)

    prop_file = directory / f"{stem}.property.txt"
    out_file = directory / f"{stem}.out"

    if prop_file.exists():
        _parse_property_file(prop_file, result)

    if out_file.exists():
        _parse_out_file(out_file, result)

    if not prop_file.exists() and not out_file.exists():
        raise FileNotFoundError(f"No ORCA output files found for stem: {directory / stem}")

    return result


def parse_orca_output(base_path: str | Path) -> dict:
    """Convenience dictionary parser for ORCA results."""
    res = parse_orca_results(base_path)
    return {
        "energy_scf": res.el_energy,
        "dipole_total": res.dipole_magnitude,
        "homo_ev": res.homo_energy,
        "lumo_ev": res.lumo_energy,
        "gap_ev": res.homo_lumo_gap,
        "converged": res.converged,
        "optimization_converged": res.optimization_converged,
        "loewdin_charges": res.loewdin_charges,
        "mulliken_charges": res.mulliken_charges,
        "atomic_symbols": res.atomic_symbols,
        "atomic_numbers": res.atomic_numbers,
    }
