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


PERIODIC_TABLE = {
    1: ("H", "#FFFFFF", 0.31),
    6: ("C", "#505050", 0.76),
    7: ("N", "#3050F8", 0.71),
    8: ("O", "#FF0D0D", 0.66),
    9: ("F", "#90E050", 0.57),
    15: ("P", "#FF8000", 1.07),
    16: ("S", "#FFFF30", 1.05),
    17: ("Cl", "#1FF01F", 1.02),
}

BOHR_TO_ANGSTROM = 0.529177249


def parse_cube_file(cube_path: str | Path) -> Dict[str, Any]:
    """
    Parse a standard Gaussian .cube file containing 3D volumetric scalar fields
    (molecular orbitals, electron density, or electrostatic potential).
    """
    path = Path(cube_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        title = f.readline().strip()
        comment = f.readline().strip()
        line3 = f.readline().split()
        natoms = abs(int(line3[0]))
        origin = np.array([float(x) for x in line3[1:4]])

        nx, vx_x, vx_y, vx_z = f.readline().split()
        ny, vy_x, vy_y, vy_z = f.readline().split()
        nz, vz_x, vz_y, vz_z = f.readline().split()

        nx, ny, nz = int(nx), int(ny), int(nz)
        vx = np.array([float(vx_x), float(vx_y), float(vx_z)])
        vy = np.array([float(vy_x), float(vy_y), float(vy_z)])
        vz = np.array([float(vz_x), float(vz_y), float(vz_z)])

        atoms = []
        for _ in range(natoms):
            aline = f.readline().split()
            atno = int(aline[0])
            charge = float(aline[1])
            pos = np.array([float(aline[2]), float(aline[3]), float(aline[4])])
            elem, color, radius = PERIODIC_TABLE.get(atno, (f"X{atno}", "#888888", 0.7))
            atoms.append({
                "atno": atno,
                "element": elem,
                "color": color,
                "radius": radius,
                "charge": charge,
                "pos_bohr": pos,
                "pos_ang": pos * BOHR_TO_ANGSTROM,
            })

        # Check for optional MO line (if natoms was negative in line 3)
        pos_after_atoms = f.tell()
        line_next = f.readline()
        if len(line_next.split()) <= 2 and line_next.strip():
            pass  # Orbital header line
        else:
            f.seek(pos_after_atoms)

        # Read remaining scalar data
        data_str = f.read()
        raw_vals = np.fromstring(data_str, sep=" ")
        grid = raw_vals.reshape((nx, ny, nz))

    return {
        "title": title,
        "comment": comment,
        "natoms": natoms,
        "origin_bohr": origin,
        "origin_ang": origin * BOHR_TO_ANGSTROM,
        "voxel_vectors_bohr": (vx, vy, vz),
        "shape": (nx, ny, nz),
        "atoms": atoms,
        "grid": grid,
    }


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
    t1_energies: Tuple[float, float] = (-6.342, -2.895),
    t3_energies: Tuple[float, float] = (-6.412, -2.917),
    dpi: int = 300,
) -> Path:
    """
    Generate a 4-panel publication-grade figure of Frontier Molecular Orbitals (HOMO & LUMO)
    for both benzofuroxan tautomers, showing wave-function phases and molecular structures.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cubes = {
        "T1_HOMO": (parse_cube_file(t1_homo_cube), t1_energies[0], "Benzofuroxan 1-oxide — HOMO"),
        "T1_LUMO": (parse_cube_file(t1_lumo_cube), t1_energies[1], "Benzofuroxan 1-oxide — LUMO"),
        "T3_HOMO": (parse_cube_file(t3_homo_cube), t3_energies[0], "Benzofuroxan 3-oxide — HOMO"),
        "T3_LUMO": (parse_cube_file(t3_lumo_cube), t3_energies[1], "Benzofuroxan 3-oxide — LUMO"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=dpi)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    keys = ["T1_HOMO", "T1_LUMO", "T3_HOMO", "T3_LUMO"]

    for ax, key in zip(ax_list, keys):
        data, ev, title = cubes[key]
        atoms = data["atoms"]
        grid = data["grid"]
        shape = data["shape"]
        origin_ang = data["origin_ang"]
        vx, vy, vz = data["voxel_vectors_bohr"]

        # Project volumetric orbital slice across the molecular plane
        # Aromatic/heterocyclic HOMO and LUMO are pi orbitals with a nodal plane (psi = 0)
        # coinciding with the molecular ring plane. Slicing at ~0.70 Å above the plane
        # captures the peak amplitude of 2p_z lobes without destructive cancellation across the nodal plane.
        z_atoms = float(np.mean([a["pos_ang"][2] for a in atoms]))
        dz = vz[2] * BOHR_TO_ANGSTROM
        z_atom_idx = int(round((z_atoms - origin_ang[2]) / dz))
        offset_voxels = max(1, int(round(0.70 / dz)))
        slice_idx = min(shape[2] - 1, max(0, z_atom_idx + offset_voxels))
        slice_2d = grid[:, :, slice_idx]

        extent = [
            origin_ang[0],
            origin_ang[0] + shape[0] * vx[0] * BOHR_TO_ANGSTROM,
            origin_ang[1],
            origin_ang[1] + shape[1] * vy[1] * BOHR_TO_ANGSTROM,
        ]

        # Draw filled orbital contour lobes (Diverging Colormap: Red = negative, Blue = positive)
        levels = np.linspace(-0.08, 0.08, 31)
        cf = ax.contourf(
            slice_2d.T,
            levels=levels,
            extent=extent,
            cmap="RdBu",
            alpha=0.75,
            extend="both",
            zorder=1,
        )
        ax.contour(
            slice_2d.T,
            levels=[-0.02, 0.02],
            extent=extent,
            colors=["#b2182b", "#2166ac"],
            linewidths=1.2,
            zorder=1,
        )

        _draw_molecule_2d_projection(ax, atoms, plane="xy")

        ax.set_title(f"{title}\n$E = {ev:.3f}$ eV", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("X (Å)", fontsize=10)
        ax.set_ylabel("Y (Å)", fontsize=10)
        ax.set_aspect("equal")

        # Colorbar inside / beside
        cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.04)
        cbar.set_label(r"Wavefunction Amplitude $\psi$", fontsize=9)

    fig.suptitle(
        "SharK: Frontier Molecular Orbital (FMO) Spatial Profiles (ORCA 6 DFT / B3LYP)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
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

