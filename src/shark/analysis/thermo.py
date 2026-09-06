"""Thermochemical and Boltzmann population analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List
from ..core.parser import CalculationResult

# Physical constants
HARTREE_TO_KCAL_MOL = 627.509474  # 1 Eh in kcal/mol
GAS_CONSTANT_R = 1.98720425864083e-3  # kcal / (mol * K)


def calculate_relative_thermo(
    results: List[CalculationResult],
    temperature: float = 298.15
) -> pd.DataFrame:
    """
    Compute relative energetics (dE, dH, dG in kcal/mol) and Boltzmann populations
    at the given temperature for a series of structures/tautomers/conformers.
    """
    if not results:
        raise ValueError("Results list cannot be empty.")

    # Sort results by Gibbs free energy ascending (lowest energy first)
    sorted_results = sorted(results, key=lambda r: r.gibbs_energy)
    min_g = sorted_results[0].gibbs_energy

    rows = []
    delta_g_list = []

    for r in sorted_results:
        d_g = (r.gibbs_energy - min_g) * HARTREE_TO_KCAL_MOL
        delta_g_list.append(d_g)

    # Compute Boltzmann factors and populations
    # P_i = exp(-dG_i / (R*T)) / sum(exp(-dG_j / (R*T)))
    rt = GAS_CONSTANT_R * temperature
    boltzmann_factors = [np.exp(-dg / rt) for dg in delta_g_list]
    z_partition = sum(boltzmann_factors)
    populations = [(bf / z_partition) * 100.0 for bf in boltzmann_factors]

    # Reference minimum electronic energy and enthalpy
    min_e = sorted_results[0].el_energy
    min_h = sorted_results[0].enthalpy

    for i, r in enumerate(sorted_results):
        d_e = (r.el_energy - min_e) * HARTREE_TO_KCAL_MOL
        d_h = (r.enthalpy - min_h) * HARTREE_TO_KCAL_MOL
        d_g = delta_g_list[i]
        pop = populations[i]

        rows.append({
            "Name": r.name,
            "Converged": r.converged,
            "N_imag": len(r.imaginary_frequencies),
            "E_el (Eh)": r.el_energy,
            "ZPE (Eh)": r.zpe,
            "H (Eh)": r.enthalpy,
            "G (Eh)": r.gibbs_energy,
            "dE_el (kcal/mol)": round(d_e, 3),
            "dH (kcal/mol)": round(d_h, 3),
            "dG (kcal/mol)": round(d_g, 3),
            "Population (%)": round(pop, 2),
            "Dipole (D)": round(r.dipole_magnitude, 3),
            "HOMO (eV)": round(r.homo_energy, 3) if r.homo_energy is not None else None,
            "LUMO (eV)": round(r.lumo_energy, 3) if r.lumo_energy is not None else None,
            "Gap (eV)": round(r.homo_lumo_gap, 3) if r.homo_lumo_gap is not None else None,
        })

    df = pd.DataFrame(rows)
    return df

