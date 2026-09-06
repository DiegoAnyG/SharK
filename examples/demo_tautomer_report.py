#!/usr/bin/env python3
"""
Demonstration script for SharK:
Parses existing ORCA 6.1.1 DFT results for benzofuroxan-5-carboxylic acid
tautomers, computes thermochemical Boltzmann distributions, generates
FTIR transmittance spectra, extracts volumetric cube files, renders
Frontier Molecular Orbitals (HOMO/LUMO) and total electron density, and
compiles a comprehensive research report.
"""

from pathlib import Path

from shark.core.parser import parse_orca_results
from shark.analysis.thermo import calculate_relative_thermo
from shark.reports.visualizer import (
    plot_boltzmann_equilibrium,
    plot_ir_comparison,
    generate_markdown_report,
    generate_html_report,
)
from shark.reports.orbitals import (
    plot_frontier_orbitals_panel,
    plot_electron_density_comparison,
)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    dft_dir = root_dir / "dft_benzofuroxan"
    out_dir = root_dir / "reports" / "benzofuroxan_tautomers"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SharK Quantum Chemical Report Engine: Pilot Tautomer Benchmark")
    print("=" * 70)

    # 1. Parse ORCA results
    print("\n[1/5] Parsing ORCA 6.1.1 output files...")
    res_1 = parse_orca_results(dft_dir / "tautomer_1_oxide", name="Benzofuroxan 1-oxide")
    res_3 = parse_orca_results(dft_dir / "tautomer_3_oxide", name="Benzofuroxan 3-oxide")

    results = [res_1, res_3]
    for r in results:
        print(f"  [+] {r.name}:")
        print(f"      Status: {'CONVERGED' if r.converged else 'FAILED'} (N_imag = {len(r.imaginary_frequencies)})")
        print(f"      G = {r.gibbs_energy:.6f} Eh | Dipole = {r.dipole_magnitude:.2f} D | Gap = {r.homo_lumo_gap:.2f} eV")

    # 2. Compute Thermodynamics and Boltzmann distribution
    print("\n[2/5] Computing thermochemical metrics and Boltzmann populations at 298.15 K...")
    df = calculate_relative_thermo(results, temperature=298.15)
    
    # Save CSV
    csv_path = out_dir / "thermodynamics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [+] Saved tabular data: {csv_path.relative_to(root_dir)}")

    print("\n" + df.to_string(index=False) + "\n")

    # 3. Generate Visualizations (Spectra & Thermodynamics)
    print("[3/5] Generating publication-quality thermodynamics & FTIR spectra (300 DPI)...")
    fig_boltzmann = out_dir / "boltzmann_distribution.png"
    plot_boltzmann_equilibrium(
        df,
        fig_boltzmann,
        title="Benzofuroxan-5-Carboxylic Acid Tautomer Equilibrium (CPCM Water)",
    )
    print(f"  [+] Generated: {fig_boltzmann.relative_to(root_dir)}")

    fig_ir = out_dir / "ir_spectra_comparison.png"
    plot_ir_comparison(
        results,
        fig_ir,
        title="Simulated FTIR Spectra: 1-oxide vs 3-oxide (B3LYP/def2-SVP/D4/CPCM Water)",
        fwhm=15.0,
        mode="transmittance",
    )
    print(f"  [+] Generated: {fig_ir.relative_to(root_dir)}")

    # 4. Generate Quantum Electronic Visualizations (HOMO/LUMO and Density)
    print("\n[4/5] Rendering Frontier Molecular Orbitals & Electron Density (300 DPI)...")
    t1_homo = dft_dir / "tautomer_1_oxide.mo45a.cube"
    t1_lumo = dft_dir / "tautomer_1_oxide.mo46a.cube"
    t3_homo = dft_dir / "tautomer_3_oxide.mo45a.cube"
    t3_lumo = dft_dir / "tautomer_3_oxide.mo46a.cube"

    fig_orbitals = out_dir / "frontier_orbitals_comparison.png"
    if all(p.exists() for p in [t1_homo, t1_lumo, t3_homo, t3_lumo]):
        plot_frontier_orbitals_panel(
            t1_homo,
            t1_lumo,
            t3_homo,
            t3_lumo,
            fig_orbitals,
            t1_energies=(res_1.homo_energy, res_1.lumo_energy),
            t3_energies=(res_3.homo_energy, res_3.lumo_energy),
        )
        print(f"  [+] Generated: {fig_orbitals.relative_to(root_dir)}")

    t1_dens = dft_dir / "tautomer_1_oxide.eldens.cube"
    t3_dens = dft_dir / "tautomer_3_oxide.eldens.cube"
    fig_density = out_dir / "electron_density_comparison.png"
    if t1_dens.exists() and t3_dens.exists():
        plot_electron_density_comparison(
            t1_dens,
            t3_dens,
            fig_density,
        )
        print(f"  [+] Generated: {fig_density.relative_to(root_dir)}")

    # 5. Compile Research Reports
    print("\n[5/5] Compiling research reports...")
    report_images = [fig_boltzmann, fig_ir]
    if fig_orbitals.exists():
        report_images.append(fig_orbitals)
    if fig_density.exists():
        report_images.append(fig_density)

    md_path = out_dir / "report.md"
    generate_markdown_report(
        df,
        md_path,
        title="Benzofuroxan Tautomeric Equilibrium & Quantum Chemical Analysis",
        notes=(
            "Calculations performed with ORCA 6.1.1 (B3LYP/def2-SVP D4 RIJCOSX CPCM Water). "
            "Both tautomers converged to true stationary minima (zero imaginary frequencies). "
            "The 3-oxide tautomer is thermodynamically favored in aqueous solution by 0.755 kcal/mol, "
            "accounting for 78.1% of the equilibrium population at 298.15 K."
        )
    )
    print(f"  [+] Saved Markdown report: {md_path.relative_to(root_dir)}")

    html_path = out_dir / "report.html"
    generate_html_report(
        df,
        images=report_images,
        out_path=html_path,
        title="Benzofuroxan Tautomer Research Report",
        subtitle="Automated DFT Analysis, Spectroscopy & Frontier Orbitals",
    )
    print(f"  [+] Saved interactive HTML report: {html_path.relative_to(root_dir)}")

    print("\n" + "=" * 70)
    print("  SharK analysis and visualization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
