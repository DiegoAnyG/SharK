"""Discover frontier orbital fields from ORCA occupations and CUBE headers."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

import numpy as np

from .cube import parse_cube_file


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_orbital_metadata(output: str | Path) -> dict:
    """Read the final closed-shell or spin-resolved ORCA orbital table.

    Fractional occupations are rejected: their HOMO/LUMO boundary needs an
    explicit scientific definition rather than an arbitrary threshold.
    """
    path = Path(output)
    text = path.read_text(encoding='utf-8', errors='replace')
    if 'ORBITAL ENERGIES' not in text:
        raise ValueError(f'{path.name}: no orbital-energy table')
    block = text.rsplit('ORBITAL ENERGIES', 1)[1]
    channels, operator, reading = {}, 0, False
    row_pattern = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*$')
    for line in block.splitlines():
        upper = line.upper()
        if 'SPIN UP ORBITALS' in upper or 'SPIN DOWN ORBITALS' in upper:
            operator = int('DOWN' in upper)
            reading = False
            continue
        if re.search(r'NO\s+OCC\s+E\(EH\)', upper):
            reading = True
            channels.setdefault(operator, [])
            continue
        if reading:
            match = row_pattern.match(line)
            if match:
                index, occ, eh, ev = match.groups()
                values = [float(occ), float(eh), float(ev)]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f'{path.name}: non-finite orbital data')
                if not any(abs(values[0] - n) < 1e-6 for n in (0, 1, 2)):
                    raise ValueError(f'{path.name}: fractional occupations require explicit orbital selection')
                channels[operator].append(dict(index=int(index), occupation=values[0], energy_eV=values[2]))
            elif channels.get(operator):
                reading = False
                # A second spin table can follow; unrelated property rows cannot.
        if channels and ('MULLIKEN' in upper or 'LOEWDIN' in upper or 'LOWDIN' in upper):
            break
    if not channels or any(not rows for rows in channels.values()):
        raise ValueError(f'{path.name}: incomplete orbital-energy table')
    for rows in channels.values():
        if len({r['index'] for r in rows}) != len(rows):
            raise ValueError(f'{path.name}: duplicate orbital indices')
    version = re.search(r'Program Version\s+(\S+)', text)
    commands = re.findall(r'^\s*\|\s*\d+>\s*(![^\n]+)', text, re.M)
    charge = re.search(r'Total Charge\s+Charge\s+\.+\s+(-?\d+)', text)
    mult = re.search(r'Multiplicity\s+Mult\s+\.+\s+(\d+)', text)
    return dict(channels=channels, orca_version=version[1] if version else None,
                input_keywords=' '.join(commands) or None,
                charge=int(charge[1]) if charge else None,
                multiplicity=int(mult[1]) if mult else None,
                normal_termination='ORCA TERMINATED NORMALLY' in text)


@dataclass
class OrbitalField:
    label: str
    role: str
    index: int
    operator: int
    occupation: float
    energy_eV: float | None
    cube_path: Path
    cube: dict
    metadata: dict


def discover_frontier_fields(directory: str | Path, calculations: list[dict] | None = None) -> list[OrbitalField]:
    """Discover every ORCA output, or use explicit output/cube paths in a manifest.

    Default cube association uses the calculation stem, but orbital identities
    come from headers. A manifest can override naming and display labels.
    Paths in the manifest are relative to directory.
    """
    root = Path(directory)
    specs = calculations if calculations is not None else [dict(output=p.name) for p in sorted(root.glob('*.out'))]
    if not specs:
        raise ValueError('No ORCA output files found; provide an input directory or calculation manifest')
    fields = []
    for spec in specs:
        output = root / spec['output']
        metadata = read_orbital_metadata(output)
        candidates = [root / p for p in spec['cubes']] if 'cubes' in spec else sorted(output.parent.glob(output.stem + '.*.cube'))
        available = {}
        for path in candidates:
            # Density fields cannot be used as orbital amplitudes.
            with path.open(encoding='utf-8') as stream:
                stream.readline()
                comment = stream.readline()
            if not re.search(r'Molecular orbital\s+\d+\s+of operator\s+\d+', comment, re.I):
                continue
            cube = parse_cube_file(path)
            key = (cube['operator'], cube['orbital_index'])
            if key in available:
                raise ValueError(f'{output.name}: ambiguous cube files for orbital {key}')
            available[key] = (path, cube)
        provenance = dict(metadata)
        provenance.pop('channels')
        provenance['output_file'] = output.name
        provenance['output_sha256'] = file_hash(output)
        binary = output.with_suffix('.gbw')
        provenance['gbw_sha256'] = file_hash(binary) if binary.exists() else None
        geometry = None
        xyz = output.with_suffix('.xyz')
        if xyz.exists():
            from rdkit import Chem
            molecule = Chem.MolFromXYZBlock(xyz.read_text())
            if molecule is None:
                raise ValueError(f'{xyz.name}: invalid XYZ geometry')
            geometry = ([a.GetAtomicNum() for a in molecule.GetAtoms()], molecule.GetConformer().GetPositions())
            provenance['xyz_sha256'] = file_hash(xyz)
        for operator, rows in metadata['channels'].items():
            occupied = [r for r in rows if r['occupation'] > 0]
            virtual = [r for r in rows if r['occupation'] == 0]
            if not occupied or not virtual:
                raise ValueError(f'{output.name}: missing occupied or virtual orbitals for operator {operator}')
            frontier = [('HOMO', max(occupied, key=lambda r: (r['energy_eV'], r['index']))),
                        ('LUMO', min(virtual, key=lambda r: (r['energy_eV'], r['index'])))]
            for role, row in frontier:
                key = (operator, row['index'])
                if key not in available:
                    raise FileNotFoundError(f'{output.name}: missing {role} cube for orbital {row["index"]}, operator {operator}; export it with orca_plot')
                path, cube = available[key]
                identity = [a['atno'] for a in cube['atoms']]
                positions = np.array([a['pos_ang'] for a in cube['atoms']])
                if geometry is None:
                    geometry = identity, positions
                if identity != geometry[0] or not np.allclose(positions, geometry[1], atol=2e-5, rtol=0):
                    raise ValueError(f'{path.name}: geometry does not match the associated calculation')
                fields.append(OrbitalField(spec.get('label', output.stem.replace('_', ' ')), role, row['index'], operator,
                                           row['occupation'], row['energy_eV'], path, cube, dict(provenance)))
    return fields
