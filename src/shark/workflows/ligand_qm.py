"""Reproducible isolated-ligand ORCA jobs from PoliScreen molecular inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from ..core.frontier import file_hash
from ..core.session import PoliScreenSession, match_receptor_id, split_receptor_pocket


def _flag(value) -> bool:
    return str(value).strip().lower() in ('true', '1', '1.0')


def select_ligands(session: PoliScreenSession, *, top: int = 1, target: str | None = None,
                   compounds: list[str] | None = None, pareto: bool = False,
                   include_controls: bool = False) -> list[dict]:
    """Select unique compounds, retaining all selected target/pocket associations."""
    if top < 1:
        raise ValueError('top must be positive')
    rows = []
    if session.ranking_df is not None:
        table = session.ranking_df
        name_col = next((c for c in ('compound', 'ligand', 'name') if c in table), None)
        if name_col is None:
            raise ValueError('Ranking has no compound identity column')
        rows = [dict(row, ligand_id=str(row[name_col])) for row in table.to_dict('records')]
    elif session.poses:
        rows = [dict(p.metadata, ligand_id=p.ligand_id, receptor=p.receptor_id,
                     best_dock=p.score) for p in session.poses]
    else:
        rows = [dict(ligand_id=name) for name in sorted(set(session.ligand_files) | set(session.ligand_metadata))]
    wanted = {n.casefold() for n in compounds or []}
    filtered = []
    for row in rows:
        name = row['ligand_id']
        receptor = str(row.get('receptor', ''))
        if target and not match_receptor_id(receptor, target):
            continue
        if wanted and name.casefold() not in wanted:
            continue
        if not include_controls and any(_flag(row.get(c)) for c in ('is_control', 'is_target_control')):
            continue
        if pareto and not (_flag(row.get('is_pareto')) or row.get('pareto_rank') == 1):
            continue
        value = row.get('best_dock', row.get('score'))
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = math.inf
        filtered.append((score if math.isfinite(score) else math.inf, name.casefold(), receptor, row))
    unique = {}
    pocket_scores: dict[tuple[str, str], list[float]] = {}
    for score, key, receptor, row in sorted(filtered, key=lambda item: item[:3]):
        if key not in unique:
            unique[key] = dict(ligand_id=row['ligand_id'], targets=[],
                               docking_score_kcal_mol=score if math.isfinite(score) else None,
                               pocket_breakdown=[])
        if receptor and receptor not in unique[key]['targets']:
            unique[key]['targets'].append(receptor)
        if receptor and math.isfinite(score):
            pocket_scores.setdefault((key, receptor), []).append(score)
    for key, item in unique.items():
        pose_summaries = {
            s['receptor_id']: s
            for s in session.summarize_by_target_and_pocket(ligand_id=item['ligand_id'], target=target)
        } if hasattr(session, 'summarize_by_target_and_pocket') else {}
        for rec in item['targets']:
            if rec in pose_summaries:
                item['pocket_breakdown'].append(pose_summaries[rec])
            else:
                tgt_id, pock_id = split_receptor_pocket(rec)
                scores = sorted(pocket_scores.get((key, rec), []))
                best_s = scores[0] if scores else None
                mean_s = (sum(scores) / len(scores)) if scores else None
                std_s = (
                    math.sqrt(sum((x - mean_s) ** 2 for x in scores) / (len(scores) - 1))
                    if len(scores) > 1 and mean_s is not None else (0.0 if scores else None)
                )
                item['pocket_breakdown'].append({
                    'ligand_id': item['ligand_id'],
                    'target_id': tgt_id,
                    'pocket_id': pock_id,
                    'receptor_id': rec,
                    'pose_count': len(scores),
                    'best_score': round(best_s, 3) if best_s is not None else None,
                    'mean_score': round(mean_s, 3) if mean_s is not None else None,
                    'std_score': round(std_s, 3) if std_s is not None else None,
                    'min_score': round(best_s, 3) if best_s is not None else None,
                    'max_score': round(scores[-1], 3) if scores else None,
                })
    if wanted - unique.keys():
        raise ValueError('Requested compounds absent after target/control/Pareto filtering: '
                         + ', '.join(sorted(wanted - unique.keys())))
    selected = list(unique.values()) if wanted else list(unique.values())[:top]
    if not selected:
        raise ValueError('No eligible ligands found in the session')
    return selected


def _molecule(session, ligand_id, seed, geometry):
    key = ligand_id.casefold()
    source = session.ligand_files.get(key)
    if geometry == 'smiles':
        source = None
    metadata = session.ligand_metadata.get(key, {})
    smiles = metadata.get('smiles')
    template = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if isinstance(smiles, str) and smiles.strip() and template is None:
        raise ValueError(f'Invalid recorded SMILES for {ligand_id}')
    if source:
        if source.suffix.lower() == '.sdf':
            supplier = Chem.SDMolSupplier(str(source), removeHs=False)
            if len(supplier) != 1:
                raise ValueError(f'Expected one molecule in input for {ligand_id}')
            mol = supplier[0]
        elif source.suffix.lower() == '.mol2':
            mol = Chem.MolFromMol2File(str(source), removeHs=False)
        else:
            mol = Chem.MolFromMolFile(str(source), removeHs=False)
        if mol is None:
            raise ValueError(f'Cannot read chemical structure for {ligand_id}; use --geometry smiles if recorded SMILES is available')
        source_info = dict(kind='input_structure', file=source.relative_to(session.extract_dir).as_posix(),
                           sha256=file_hash(source))
    elif template is not None:
        mol = Chem.Mol(template)
        source_info = dict(kind='recorded_smiles', sha256=hashlib.sha256(smiles.encode()).hexdigest())
    else:
        raise ValueError(f'No molecular input or recorded SMILES for {ligand_id}; PDB/PDBQT alone is insufficient')
    Chem.SanitizeMol(mol)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    if template is not None and canonical != Chem.MolToSmiles(Chem.RemoveHs(template), isomericSmiles=True):
        raise ValueError(f'Input structure and recorded SMILES disagree for {ligand_id}')
    has_geometry = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    mol = Chem.AddHs(mol, addCoords=has_geometry)
    if not has_geometry:
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f'3D embedding failed for {ligand_id}')
    coordinates = mol.GetConformer().GetPositions()
    if not np.isfinite(coordinates).all():
        raise ValueError(f'Non-finite coordinates for {ligand_id}')
    if mol.GetNumAtoms() > 1:
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < 0.2:
            raise ValueError(f'Overlapping atoms in {ligand_id}')
    source_info.update(geometry='input_3d' if has_geometry else 'ETKDGv3',
                       smiles=canonical, hydrogen_placement='RDKit AddHs', seed=seed)
    return mol, source_info


def _resources(nprocs, maxcore):
    nprocs = int(nprocs if nprocs is not None else os.environ.get('SHARK_NPROCS', '1'))
    maxcore = int(maxcore if maxcore is not None else os.environ.get('SHARK_MAXCORE', '1024'))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    if not 1 <= nprocs <= cpus or maxcore < 1:
        raise ValueError('Requested CPU or memory allocation is invalid for this host')
    available_mb = None
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    if available_mb is None and hasattr(os, 'sysconf'):
        try:
            available_mb = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1024**2
        except (ValueError, OSError):
            available_mb = None
    if available_mb is None and os.name == 'nt':
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            kernel32 = getattr(ctypes, 'windll', None)
            if kernel32 is not None and hasattr(kernel32, 'kernel32'):
                if kernel32.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    available_mb = float(stat.ullAvailPhys) / (1024.0 ** 2)
        except Exception:
            available_mb = None
    if available_mb is not None and nprocs * maxcore > available_mb * 0.75:
        raise ValueError('ORCA maxcore times nprocs exceeds 75% of available host memory')
    return nprocs, maxcore


def prepare_ligand_jobs(session: PoliScreenSession, output_dir: str | Path | None = None, *,
                        top: int = 1, target: str | None = None, compounds: list[str] | None = None,
                        pareto: bool = False, include_controls: bool = False,
                        method: str = 'r2SCAN-3c', solvent: str | None = 'Water',
                        optimize: bool = True, frequencies: bool = False,
                        multiplicity: int = 1, charge: int | None = None,
                        seed: int = 42, nprocs: int | None = None, maxcore: int | None = None,
                        geometry: str = 'input', orbital_grid: int | None = 60,
                        fukui: bool = False) -> list[Path]:
    """Prepare input-only jobs. No electronic results exist until ORCA completes."""
    if not re.fullmatch(r'[A-Za-z0-9+(),*/._ -]+', method) or not method.strip():
        raise ValueError('Method must be an ORCA keyword line, without blocks or newlines')
    if any(word.lower() in ('opt', 'sp', 'freq', 'numfreq') or word.lower().startswith('pal')
           for word in method.split()):
        raise ValueError('Use job options for optimization, frequencies and parallelism')
    if solvent is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', solvent):
        raise ValueError('Invalid solvent keyword')
    if not 0 <= seed < 2**31 or multiplicity < 1:
        raise ValueError('Seed or multiplicity out of range')
    if fukui and multiplicity != 1:
        raise ValueError('Finite-difference Fukui analysis currently requires a closed-shell reference state')
    if geometry not in ('input', 'smiles'):
        raise ValueError('Geometry source must be input or smiles')
    if orbital_grid is not None and not 10 <= orbital_grid <= 256:
        raise ValueError('Orbital grid must contain 10 to 256 points per axis')
    nprocs, maxcore = _resources(nprocs, maxcore)
    selections = select_ligands(session, top=top, target=target, compounds=compounds,
                                pareto=pareto, include_controls=include_controls)
    # Validate all chemistry before creating any job directories.
    prepared = []
    for item in selections:
        mol, source = _molecule(session, item['ligand_id'], seed, geometry)
        formal_charge = Chem.GetFormalCharge(mol)
        if charge is not None and charge != formal_charge:
            raise ValueError('Charge disagrees with molecular input; supply the intended protonation state')
        electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - formal_charge
        if electrons < multiplicity - 1 or (electrons - multiplicity + 1) % 2:
            raise ValueError(f'Charge and multiplicity are incompatible for {item["ligand_id"]}')
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) and multiplicity == 1:
            raise ValueError('Radical input requires an explicit non-singlet multiplicity')
        prepared.append((item, mol, source, formal_charge))
    if output_dir is None:
        scratch = Path(os.environ.get('SHARK_SCRATCH', tempfile.gettempdir())).expanduser()
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='shark-dft-', dir=scratch))
    else:
        root = Path(output_dir).expanduser().resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError('QM output directory must be empty; existing jobs are never overwritten')
        root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, (selection, mol, source, formal_charge) in enumerate(prepared, 1):
        job = root / f'job-{index:03d}'
        job.mkdir()
        xyz = Chem.MolToXYZBlock(mol)
        (job / 'geometry.xyz').write_text(xyz, encoding='utf-8')
        keywords = f'{method.strip()} {"Opt" if optimize else "SP"} TightSCF'
        if frequencies:
            keywords += ' Freq'
        if solvent:
            keywords += f' CPCM({solvent})'
        deck = (f'! {keywords}\n%pal nprocs {nprocs} end\n%maxcore {maxcore}\n'
                f'* xyz {formal_charge} {multiplicity}\n' + '\n'.join(xyz.splitlines()[2:]) + '\n*\n')
        (job / 'calculation.inp').write_text(deck, encoding='utf-8')
        try:
            version = importlib.metadata.version('shark-qc')
        except importlib.metadata.PackageNotFoundError:
            version = 'source-checkout'
        record = dict(schema_version=1, status='prepared', calculation='isolated_ligand',
                      selection=selection, source=source, session_sha256=session.archive_sha256,
                      poliscreen_version=session.manifest.get('poliscreen'),
                      parameters=dict(method=method.strip(), solvent=solvent, optimize=optimize,
                                      frequencies=frequencies, charge=formal_charge, multiplicity=multiplicity,
                                      nprocs=nprocs, maxcore_mb=maxcore, seed=seed, geometry=geometry,
                                      orbital_grid=orbital_grid, fukui=fukui),
                      software=dict(shark=version, rdkit=rdBase.rdkitVersion, numpy=np.__version__),
                      inputs={name: file_hash(job / name) for name in ('calculation.inp', 'geometry.xyz')},
                      results=None)
        (job / 'job.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        jobs.append(job)
    return jobs
