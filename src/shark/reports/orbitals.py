"""Parser and visualization module for volumetric Gaussian .cube files (Orbitals, Density, MEP)."""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ..core.cube import BOHR_TO_ANGSTROM, parse_cube_file


def _draw_molecule_2d_projection(ax, atoms: List[Dict[str, Any]], plane: str = "xy") -> None:
    """Draw molecular bonds and atoms projected onto the chosen Cartesian plane."""
    idx_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    i1, i2 = idx_map[plane]

    positions = np.array([a["pos_ang"] for a in atoms])

    # Draw bonds based on distance heuristic (sum of covalent radii * 1.3)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            max_dist = (atoms[i]["radius"] + atoms[j]["radius"]) * 1.25
            if dist <= max_dist:
                ax.plot(
                    [positions[i][i1], positions[j][i1]],
                    [positions[i][i2], positions[j][i2]],
                    color="#2b2b2b",
                    linewidth=2.4,
                    zorder=2,
                    solid_capstyle="round",
                )

    # Draw atom nodes
    for a in atoms:
        pos = a["pos_ang"]
        ax.scatter(
            pos[i1],
            pos[i2],
            s=180 if a["element"] != "H" else 90,
            color=a["color"],
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        if a["element"] not in ["H", "C"]:
            ax.text(
                pos[i1],
                pos[i2],
                a["element"],
                color="white" if a["element"] in ["N", "C"] else "black",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )


def plot_frontier_orbitals_panel(
    t1_homo_cube: str | Path,
    t1_lumo_cube: str | Path,
    t3_homo_cube: str | Path,
    t3_lumo_cube: str | Path,
    out_path: str | Path,
    t1_energies: Tuple[float, float] | None = None,
    t3_energies: Tuple[float, float] | None = None,
    dpi: int = 300,
) -> Path:
    """Compatibility adapter for the original two-calculation API.

    New code should use discover_frontier_fields and export_orbital_report,
    which accept any number of calculations and configurable view settings.
    """
    from ..core.frontier import OrbitalField, read_orbital_metadata
    from .orbital_viewer import ViewSettings, build_orbital_figure

    fields = []
    for paths, energies in (((t1_homo_cube, t1_lumo_cube), t1_energies),
                            ((t3_homo_cube, t3_lumo_cube), t3_energies)):
        for position, (cube_path, role) in enumerate(zip(paths, ('HOMO', 'LUMO'))):
            path = Path(cube_path)
            cube = parse_cube_file(path)
            index, operator = cube['orbital_index'], cube['operator']
            if index is None or operator is None:
                raise ValueError('Frontier plotting requires an identified orbital CUBE')
            energy = energies[position] if energies is not None else None
            metadata, occupation = {}, None
            outputs = [p for p in path.parent.glob('*.out') if path.name.startswith(p.stem + '.')]
            label = path.stem
            if len(outputs) == 1:
                label = outputs[0].stem
                metadata = read_orbital_metadata(outputs[0])
                entries = metadata['channels'].get(operator, [])
                occupied = [r for r in entries if r['occupation'] > 0]
                virtual = [r for r in entries if r['occupation'] == 0]
                expected = (max(occupied, key=lambda r: (r['energy_eV'], r['index'])) if role == 'HOMO'
                            else min(virtual, key=lambda r: (r['energy_eV'], r['index'])))
                if index != expected['index']:
                    raise ValueError(f'{path.name} is not the {role} in the associated ORCA output')
                occupation = expected['occupation']
                if energy is None:
                    energy = expected['energy_eV']
            if occupation is None:
                raise ValueError('Use the generic OrbitalField API with explicit metadata when ORCA output is unavailable')
            fields.append(OrbitalField(label, role, index, operator, occupation, energy, path, cube, metadata))
    settings = ViewSettings(export_scale=dpi / 96)
    fig, _, _ = build_orbital_figure(fields, settings)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_file, scale=settings.export_scale)
    return out_file


def plot_electron_density_comparison(
    t1_density_cube: str | Path,
    t3_density_cube: str | Path,
    out_path: str | Path,
    dpi: int = 300,
) -> Path:
    """
    Generate a 2-panel figure showing total electron density distribution
    and isodensity contours (van der Waals envelope) for both tautomers.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    c1 = parse_cube_file(t1_density_cube)
    c3 = parse_cube_file(t3_density_cube)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    for ax, data, title in [
        (ax1, c1, "Benzofuroxan 1-oxide: Total Electron Density $\\rho(\\mathbf{r})$"),
        (ax2, c3, "Benzofuroxan 3-oxide: Total Electron Density $\\rho(\\mathbf{r})$"),
    ]:
        atoms = data["atoms"]
        grid = data["grid"]
        shape = data["shape"]
        origin_ang = data["origin_ang"]
        vx, vy, vz = data["voxel_vectors_bohr"]

        z_mid = shape[2] // 2
        slice_2d = grid[:, :, z_mid]

        extent = [
            origin_ang[0],
            origin_ang[0] + shape[0] * vx[0] * BOHR_TO_ANGSTROM,
            origin_ang[1],
            origin_ang[1] + shape[1] * vy[1] * BOHR_TO_ANGSTROM,
        ]

        levels = np.logspace(-3, 0, 30)
        cf = ax.contourf(
            slice_2d.T,
            levels=levels,
            extent=extent,
            cmap="inferno",
            alpha=0.85,
            zorder=1,
            locator=matplotlib.ticker.LogLocator(),
        )
        ax.contour(
            slice_2d.T,
            levels=[0.002, 0.05, 0.2],
            extent=extent,
            colors="white",
            linewidths=0.8,
            linestyles=["dashed", "solid", "solid"],
            zorder=1,
        )

        _draw_molecule_2d_projection(ax, atoms, plane="xy")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("X (Å)", fontsize=10)
        ax.set_ylabel("Y (Å)", fontsize=10)
        ax.set_aspect("equal")

        cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.04)
        cbar.set_label(r"Electron Density $\rho$ ($\mathrm{e/Bohr^3}$)", fontsize=9)

    fig.suptitle(
        "SharK: Total Molecular Electron Density Contours",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file

