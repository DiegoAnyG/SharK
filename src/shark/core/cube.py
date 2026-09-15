"""Validated scalar Gaussian CUBE reader, retaining legacy dictionary keys."""

from pathlib import Path
import re

import numpy as np
from scipy.constants import physical_constants

BOHR_TO_ANGSTROM = physical_constants["Bohr radius"][0] * 1e10


def parse_cube_file(cube_path: str | Path, dataset_id: int | None = None) -> dict:
    """Read scalar or explicitly selected multi-orbital data in atomic units.

    Positive grid counts denote bohr; the negative-count angstrom convention
    is accepted only when all three counts are negative. Scalar amplitudes
    retain their source units. No field normalization or coordinate alignment
    is performed here.
    """
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    path = Path(cube_path)
    number = lambda token: float(token.replace("D", "E").replace("d", "e"))
    try:
        with path.open(encoding="utf-8") as stream:
            title, comment = stream.readline().strip(), stream.readline().strip()
            header = stream.readline().split()
            signed_natoms = int(header[0])
            origin = np.array([number(t) for t in header[1:4]])
            nvalues = int(header[4]) if len(header) > 4 else 1
            axes = [stream.readline().split() for _ in range(3)]
            counts = np.array([int(row[0]) for row in axes])
            vectors = np.array([[number(t) for t in row[1:4]] for row in axes])
            if origin.shape != (3,) or vectors.shape != (3, 3):
                raise ValueError("Incomplete grid geometry")
            if np.any(counts == 0) or not (np.all(counts > 0) or np.all(counts < 0)):
                raise ValueError("Grid counts must have consistent, nonzero signs")
            factor = 1.0 / BOHR_TO_ANGSTROM if np.all(counts < 0) else 1.0
            origin *= factor
            vectors *= factor
            atoms = []
            for _ in range(abs(signed_natoms)):
                row = stream.readline().split()
                atno = int(row[0])
                position = np.array([number(t) for t in row[2:5]]) * factor
                if atno < 0 or atno > table.GetMaxAtomicNumber() or position.shape != (3,):
                    raise ValueError("Invalid atom record")
                atoms.append(dict(atno=atno, element=table.GetElementSymbol(atno),
                                  charge=number(row[1]), pos_bohr=position,
                                  pos_ang=position * BOHR_TO_ANGSTROM,
                                  radius=table.GetRcovalent(atno), color="#777777"))
            tokens = stream.read().replace("D", "E").replace("d", "e").split()
        ids = []
        if signed_natoms < 0:
            nsets = int(tokens[0])
            if nsets < 1 or nvalues != 1:
                raise ValueError("Invalid orbital dataset header")
            ids = [int(t) for t in tokens[1:1 + nsets]]
            tokens = tokens[1 + nsets:]
        else:
            nsets = nvalues
        if nsets < 1:
            raise ValueError("Dataset count must be positive")
        if nsets > 1 and dataset_id is None:
            raise ValueError("Multiple datasets: select dataset_id explicitly")
        if dataset_id is not None:
            if not ids or dataset_id not in ids:
                raise ValueError("Requested dataset_id is not present")
            selected = ids.index(dataset_id)
        else:
            selected = 0
        shape = tuple(abs(int(n)) for n in counts)
        raw = np.array(tokens, dtype=float)
        if raw.size != int(np.prod(shape)) * nsets:
            raise ValueError("Scalar count does not match grid dimensions")
        grid = raw.reshape((*shape, nsets))[..., selected].copy()
        if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(vectors)):
            raise ValueError("Non-finite scalar or grid values")
        if not np.all(np.isfinite(origin)) or any(not np.all(np.isfinite(a['pos_bohr'])) for a in atoms):
            raise ValueError("Non-finite coordinates")
        if abs(np.linalg.det(vectors)) < np.finfo(float).eps:
            raise ValueError("Singular voxel transform")
        match = re.search(r"Molecular orbital\s+(\d+)\s+of operator\s+(\d+)", comment, re.I)
        mo_index = int(match[1]) if match else (ids[selected] if ids else None)
        if match and ids and nsets == 1 and ids[0] != mo_index:
            raise ValueError("Orbital identifiers disagree between cube headers")
        return dict(title=title, comment=comment, natoms=abs(signed_natoms),
                    origin_bohr=origin, origin_ang=origin * BOHR_TO_ANGSTROM,
                    voxel_vectors_bohr=tuple(vectors), shape=shape, atoms=atoms, grid=grid,
                    dataset_ids=ids, orbital_index=mo_index,
                    operator=int(match[2]) if match else None,
                    coordinate_units="bohr", source_coordinate_units="angstrom" if factor != 1 else "bohr")
    except (ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"Invalid CUBE {path.name}: {exc}") from exc
