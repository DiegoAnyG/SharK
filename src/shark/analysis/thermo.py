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

    Enforces INV-006 / TEST-011 / TEST-012:
    - If valid Gibbs free energies exist for all structures:
      computes Boltzmann populations from ΔG, labeled "Boltzmann population from Gibbs free energy (ΔG)".
    - If only electronic energies exist:
      computes populations using electronic energies as an explicit proxy,
      labeled "Electronic-energy population proxy (ΔE_el)", without fabricating ΔG.
    """
    if not results:
        raise ValueError("Results list cannot be empty.")

    all_have_gibbs = all(
        r.gibbs_energy is not None and np.isfinite(r.gibbs_energy) and r.gibbs_energy != 0.0
        for r in results
    )
    all_have_el = all(
        r.el_energy is not None and np.isfinite(r.el_energy) and r.el_energy != 0.0
        for r in results
    )

    rt = GAS_CONSTANT_R * temperature

    if all_have_gibbs:
        sorted_results = sorted(results, key=lambda r: r.gibbs_energy)
        min_g = sorted_results[0].gibbs_energy
        min_e = sorted_results[0].el_energy
        min_h = sorted_results[0].enthalpy

        delta_g_list = [(r.gibbs_energy - min_g) * HARTREE_TO_KCAL_MOL for r in sorted_results]
        boltzmann_factors = [np.exp(-dg / rt) for dg in delta_g_list]
        z_partition = sum(boltzmann_factors) if sum(boltzmann_factors) > 0 else 1.0
        populations = [(bf / z_partition) * 100.0 for bf in boltzmann_factors]

        rows = []
        for i, r in enumerate(sorted_results):
            d_e = (r.el_energy - min_e) * HARTREE_TO_KCAL_MOL if r.el_energy != 0.0 else 0.0
            d_h = (r.enthalpy - min_h) * HARTREE_TO_KCAL_MOL if r.enthalpy != 0.0 else 0.0
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
                "Population_Basis": "Gibbs free energy (ΔG)",
                "Dipole (D)": round(r.dipole_magnitude, 3),
                "HOMO (eV)": round(r.homo_energy, 3) if r.homo_energy is not None else None,
                "LUMO (eV)": round(r.lumo_energy, 3) if r.lumo_energy is not None else None,
                "Gap (eV)": round(r.homo_lumo_gap, 3) if r.homo_lumo_gap is not None else None,
            })

        df = pd.DataFrame(rows)
        df.attrs["energy_basis"] = "gibbs"
        df.attrs["population_label"] = "Boltzmann population from Gibbs free energy (ΔG)"
        df.attrs["delta_energy_column"] = "dG (kcal/mol)"
        return df

    elif all_have_el:
        sorted_results = sorted(results, key=lambda r: r.el_energy)
        min_e = sorted_results[0].el_energy

        delta_e_list = [(r.el_energy - min_e) * HARTREE_TO_KCAL_MOL for r in sorted_results]
        boltzmann_factors = [np.exp(-de / rt) for de in delta_e_list]
        z_partition = sum(boltzmann_factors) if sum(boltzmann_factors) > 0 else 1.0
        populations = [(bf / z_partition) * 100.0 for bf in boltzmann_factors]

        rows = []
        for i, r in enumerate(sorted_results):
            d_e = delta_e_list[i]
            pop = populations[i]

            rows.append({
                "Name": r.name,
                "Converged": r.converged,
                "N_imag": len(r.imaginary_frequencies),
                "E_el (Eh)": r.el_energy,
                "ZPE (Eh)": r.zpe if r.zpe != 0.0 else None,
                "H (Eh)": r.enthalpy if r.enthalpy != 0.0 else None,
                "G (Eh)": None,
                "dE_el (kcal/mol)": round(d_e, 3),
                "dH (kcal/mol)": None,
                "dG (kcal/mol)": None,
                "Population (%)": round(pop, 2),
                "Population_Basis": "Electronic-energy population proxy (ΔE_el)",
                "Dipole (D)": round(r.dipole_magnitude, 3),
                "HOMO (eV)": round(r.homo_energy, 3) if r.homo_energy is not None else None,
                "LUMO (eV)": round(r.lumo_energy, 3) if r.lumo_energy is not None else None,
                "Gap (eV)": round(r.homo_lumo_gap, 3) if r.homo_lumo_gap is not None else None,
            })

        df = pd.DataFrame(rows)
        df.attrs["energy_basis"] = "electronic"
        df.attrs["population_label"] = "Electronic-energy population proxy (ΔE_el)"
        df.attrs["delta_energy_column"] = "dE_el (kcal/mol)"
        return df
    else:
        raise ValueError("Cannot calculate relative thermodynamics: inconsistent or missing energies.")
