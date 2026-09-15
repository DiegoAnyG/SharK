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

    @property
    def is_stationary_minimum(self) -> bool:
        """True if the structure converged and has zero imaginary vibrational frequencies."""
        return self.converged and len(self.imaginary_frequencies) == 0


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


def _parse_out_file(out_path: Path, result: CalculationResult) -> None:
    """Extract IR intensities, dipole magnitude, and frontier orbitals from ORCA .out."""
    content = out_path.read_text(encoding="utf-8", errors="ignore")

    # Parse IR Spectrum table
    # Example format:
    #   Mode   freq       eps      Int      T**2         TX        TY        TZ
    #   6:     54.57   0.000760    3.84  ...
    ir_match = re.search(r"IR SPECTRUM\s+-+\s+Mode\s+freq.*?km/mol.*?-+\s+(.*?)(?=\*|\n\n|\Z)", content, re.DOTALL)
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
        "converged": res.converged
    }
