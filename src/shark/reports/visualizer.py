"""Publication-quality visualization and research report generation."""

from __future__ import annotations
import os
import tempfile
import html
from pathlib import Path
from typing import List, Optional

# Ensure matplotlib uses a non-GUI backend and a safe writable cache directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..core.parser import CalculationResult
from ..analysis.spectra import simulate_ir_spectrum


# Publication styling constants
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_boltzmann_equilibrium(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Tautomeric Thermodynamic Equilibrium",
    dpi: int = 300,
) -> Path:
    """
    Generate a two-panel publication figure:
    - Left: Relative Gibbs free energy (dG in kcal/mol)
    - Right: Boltzmann population percentage (%) at given temperature
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=dpi)

    names = df["Name"].tolist()
    d_g = df["dG (kcal/mol)"].tolist()
    pops = df["Population (%)"].tolist()
    x = np.arange(len(names))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    # Left: dG (kcal/mol)
    bars1 = ax1.bar(x, d_g, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title(r"Relative Free Energy ($\Delta G$)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel(r"$\Delta G$ (kcal/mol)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, max(max(d_g) * 1.35, 1.0))

    for bar, val in zip(bars1, d_g):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f} kcal/mol",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Population (%)
    bars2 = ax2.bar(x, pops, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax2.set_title(r"Boltzmann Population ($P_i$)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Population (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 105)

    for bar, val in zip(bars2, pops):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_ir_comparison(
    results: List[CalculationResult],
    out_path: str | Path,
    title: str = "Simulated Infrared (IR) Spectra",
    fwhm: float = 15.0,
    mode: str = "transmittance",
    dpi: int = 300,
) -> Path:
    """
    Generate an overlaid simulated IR spectrum comparing calculation results.
    Standard FTIR convention: Wavenumber axis is inverted (4000 cm^-1 to 400 cm^-1).
    Supports mode='transmittance' (default, baseline at 100%, peaks dip down) or mode='absorbance'.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=dpi)

    for idx, r in enumerate(results):
        if not r.ir_intensities:
            continue

        freqs_to_use = r.ir_frequencies if r.ir_frequencies else r.frequencies[-len(r.ir_intensities):]
        if len(freqs_to_use) != len(r.ir_intensities):
            continue

        wn, absorp = simulate_ir_spectrum(
            freqs_to_use,
            r.ir_intensities,
            fwhm=fwhm,
            wavenumber_range=(400.0, 4000.0),
        )
        max_abs = np.max(absorp) if np.max(absorp) > 0 else 1.0
        norm_absorp = absorp / max_abs
        color = PALETTE[idx % len(PALETTE)]

        if mode == "transmittance":
            # Convert normalized absorbance to transmittance % (FTIR standard)
            # Baseline = 100%, strongest absorption dips down to 10%
            transmittance = 100.0 * (10.0 ** (-norm_absorp))
            ax.plot(wn, transmittance, label=f"{r.name}", color=color, linewidth=1.6)
        else:
            ax.plot(wn, norm_absorp, label=f"{r.name}", color=color, linewidth=1.6)

    # Invert x-axis to match infrared spectroscopy standard (high to low wavenumber)
    ax.set_xlim(4000, 400)
    ax.set_xlabel(r"Wavenumber ($\mathrm{cm^{-1}}$)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    if mode == "transmittance":
        ax.set_ylim(0, 115)
        ax.set_ylabel("Transmittance (%)", fontsize=11, fontweight="bold")
        ax.axhline(100, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95, fontsize=10, loc="lower center")

        # Annotate key spectroscopic zones in header band (above 100% baseline) to prevent overlap with peaks or legend
        ax.axvspan(3100, 3650, color="#e0e0e0", alpha=0.3, zorder=0)
        ax.text(3375, 107, "O-H / N-H Stretch", fontsize=9, fontweight="semibold", color="#333333", ha="center")

        ax.axvspan(1650, 1850, color="#e0e0e0", alpha=0.3, zorder=0)
        ax.text(1750, 107, "C=O / N=O Stretch", fontsize=9, fontweight="semibold", color="#333333", ha="center")

        ax.axvspan(400, 1500, color="#f5f5f5", alpha=0.3, zorder=0)
        ax.text(950, 107, "Fingerprint Region", fontsize=9, fontweight="semibold", color="#555555", ha="center")
    else:
        ax.set_ylim(-0.02, 1.15)
        ax.set_ylabel("Normalized Absorbance (a.u.)", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper right")

        # Annotate key spectroscopic zones near top
        ax.axvspan(3100, 3650, color="#e0e0e0", alpha=0.3, zorder=0)
        ax.text(3375, 1.07, "O-H / N-H Stretch", fontsize=9, fontweight="semibold", color="#333333", ha="center")

        ax.axvspan(1650, 1850, color="#e0e0e0", alpha=0.3, zorder=0)
        ax.text(1750, 1.07, "C=O / N=O Stretch", fontsize=9, fontweight="semibold", color="#333333", ha="center")

        ax.axvspan(400, 1500, color="#f5f5f5", alpha=0.3, zorder=0)
        ax.text(950, 1.07, "Fingerprint Region", fontsize=9, fontweight="semibold", color="#555555", ha="center")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as GitHub-Flavored Markdown table without external tabulate dependency."""
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.values]
    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def generate_markdown_report(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "SharK Quantum Chemical Report",
    notes: Optional[str] = None,
) -> Path:
    """Export summary metrics as a clean GitHub-Flavored Markdown report."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {title}",
        "",
        "## Summary of Thermodynamics & Electronic Properties",
        "",
        _df_to_markdown(df),
        "",
    ]

    if notes:
        lines.extend(["## Analysis Notes", "", notes, ""])

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def generate_html_report(
    df: pd.DataFrame,
    images: List[Path],
    out_path: str | Path,
    title: str = "SharK Quantum Chemical Analysis Report",
    subtitle: str = "Tautomer & Conformer Evaluation",
    interactive_reports: Optional[List[Path]] = None,
) -> Path:
    """Export an interactive, self-contained HTML research report."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    table_html = df.to_html(classes="data-table", index=False, border=0)

    image_cards = []
    for img in images:
        rel_img = os.path.relpath(img, out_file.parent)
        image_cards.append(f"""
        <div class="card">
            <h3>{img.stem.replace('_', ' ').title()}</h3>
            <img src="{rel_img}" alt="{img.name}" />
        </div>
        """)

    images_section = "\n".join(image_cards)
    for report in interactive_reports or []:
        relative = html.escape(os.path.relpath(report, out_file.parent), quote=True)
        images_section += f'<div class="card"><a href="{relative}">Open interactive orbital viewer</a></div>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: auto;
            background: white;
            padding: 35px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        }}
        h1 {{ color: #0d47a1; margin-bottom: 5px; }}
        h2 {{ color: #1565c0; border-bottom: 2px solid #e3f2fd; padding-bottom: 8px; margin-top: 30px; }}
        .subtitle {{ color: #666; font-size: 1.1em; margin-bottom: 25px; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        .data-table th, .data-table td {{
            padding: 10px 14px;
            text-align: center;
        }}
        .data-table th {{
            background-color: #0d47a1;
            color: white;
            font-weight: 600;
        }}
        .data-table tr:nth-child(even) {{ background-color: #f2f7fb; }}
        .data-table tr:hover {{ background-color: #e8f0fe; }}
        .gallery {{
            display: flex;
            flex-direction: column;
            gap: 30px;
            margin-top: 25px;
        }}
        .card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            background: #ffffff;
        }}
        .card h3 {{
            margin-top: 0;
            color: #333;
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 40px;
            font-size: 0.85em;
            color: #888;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle} — Generated by <strong>SharK</strong></div>
        
        <h2>Thermodynamic Metrics & Frontier Orbitals</h2>
        {table_html}
        
        <h2>Graphical Analysis & Spectra</h2>
        <div class="gallery">
            {images_section}
        </div>

        <div class="footer">
            SharK Quantum Chemistry Workflow Engine &bull; Author: Diego Cesar Anaya Guerrero
        </div>
    </div>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return out_file
