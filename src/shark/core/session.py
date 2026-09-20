"""Read PoliScreen archives without guessing compound or target identities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
import zipfile

import pandas as pd

from .frontier import file_hash


@dataclass
class DockingPose:
    ligand_id: str
    pose_idx: int
    score: float
    receptor_id: str
    pose_file: Path
    complex_file: Path | None = None
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0
    efficiency: float | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


def match_receptor_id(candidate: str | None, target: str | None) -> bool:
    """Robust matcher between receptor IDs and user target queries.

    Handles:
    - None or empty target (matches anything)
    - Exact case-insensitive match (e.g. '8HTB_ready' == '8htb_ready')
    - Pocket stripping (e.g. '8HTB_ready~Pk1' matches '8HTB_ready' or '8HTB')
    - Preparation suffix stripping (e.g. '8HTB_ready', '8HTB_prep', '8HTB_clean' match '8HTB')
    """
    if target is None:
        return True
    if candidate is None:
        return False
    cand_str = str(candidate).strip()
    tgt_str = str(target).strip()
    if not tgt_str:
        return True
    if cand_str.casefold() == tgt_str.casefold():
        return True
    # If target explicitly specifies a pocket (~...), candidate must match that pocket
    cand_has_pocket = "~" in cand_str
    tgt_has_pocket = "~" in tgt_str
    if tgt_has_pocket:
        if not cand_has_pocket:
            return False
        cand_pocket = cand_str.split("~", 1)[1].strip().casefold()
        tgt_pocket = tgt_str.split("~", 1)[1].strip().casefold()
        if cand_pocket != tgt_pocket:
            return False

    cand_stem = cand_str.split("~", 1)[0].casefold()
    tgt_stem = tgt_str.split("~", 1)[0].casefold()
    if cand_stem == tgt_stem:
        return True
    cand_base = re.sub(r"_(?:ready|prep|docking|prepared|clean)$", "", cand_stem, flags=re.IGNORECASE)
    tgt_base = re.sub(r"_(?:ready|prep|docking|prepared|clean)$", "", tgt_stem, flags=re.IGNORECASE)
    return cand_base == tgt_base


@dataclass
class PoliScreenSession:
    session_file: Path
    project_name: str
    manifest: dict
    extract_dir: Path
    receptors: dict[str, Path] = field(default_factory=dict)
    poses: list[DockingPose] = field(default_factory=list)
    ranking_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None
    ligand_files: dict[str, Path] = field(default_factory=dict)
    ligand_metadata: dict[str, dict] = field(default_factory=dict)
    archive_sha256: str = ""
    warnings: list[str] = field(default_factory=list)

    def list_top_poses(self, top_n: int = 10) -> list[DockingPose]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        return sorted(self.poses, key=lambda p: (
            p.score if math.isfinite(p.score) else math.inf,
            p.receptor_id, p.ligand_id, p.pose_idx,
        ))[:top_n]

    def get_pose(self, ligand_id: str, pose_idx: int = 1,
                 receptor_id: str | None = None) -> DockingPose | None:
        matches = [p for p in self.poses if p.ligand_id.casefold() == ligand_id.casefold() and p.pose_idx == pose_idx
                   and match_receptor_id(p.receptor_id, receptor_id)]
        if len(matches) > 1:
            raise ValueError("Ambiguous pose; specify receptor_id including its pocket identifier")
        return matches[0] if matches else None

    def receptor_for(self, receptor_id: str, raw: bool = False) -> Path:
        """Return receptor PDB path for a given receptor ID or target.
        
        If raw=True, seeks the original/raw unprocessed receptor structure
        (e.g., stripping '_ready' or '~PkN' suffixes like '8HTB_ready~Pk1' -> '8HTB'),
        which is required for classical force-field parameterization (pdb2gmx in MD).
        """
        stem = receptor_id.split("~", 1)[0]
        if raw:
            raw_stem = re.sub(r"_(?:ready|prep|docking|prepared)$", "", stem, flags=re.IGNORECASE)
            if raw_stem in self.receptors:
                return self.receptors[raw_stem]
            for r_name, r_path in self.receptors.items():
                if r_name.casefold() == raw_stem.casefold():
                    return r_path
            for folder in ("receptors", "receptores", ""):
                cand = self.extract_dir / folder / f"{raw_stem}.pdb"
                if cand.is_file():
                    return cand

        if stem in self.receptors:
            return self.receptors[stem]

        for r_name, r_path in self.receptors.items():
            if r_name.casefold() == stem.casefold():
                return r_path

        raise ValueError(f"No prepared receptor for target {receptor_id}")


def _table(root: Path, *names: str) -> pd.DataFrame | None:
    for name in names:
        path = root / name
        if path.is_file():
            try:
                return pd.read_csv(path)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
                raise ValueError(f"Invalid session table: {name}") from exc
    return None


def _number(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _unique_stems(paths) -> dict[str, Path]:
    indexed = {}
    for path in sorted(paths):
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(f"Ambiguous molecular input: {path.stem}")
        indexed[key] = path
    return indexed


def read_poliscreen_session(session_path: str | Path, target_dir: str | Path | None = None,
                           *, max_unpacked_bytes: int = 4 * 1024**3) -> PoliScreenSession:
    """Extract into a fresh directory; accept modern and legacy PoliScreen layouts.

    An explicit destination must be empty. Invalid manifests, escaping paths,
    links, duplicate members and oversized archives are rejected before extraction.
    Callers own the extracted directory and can delete it after use.
    """
    path = Path(session_path).expanduser().resolve()
    if path.is_dir():
        manifest = {}
        if (path / 'manifest.json').is_file():
            try:
                manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
            except (ValueError, UnicodeError):
                pass
        elif (path / 'run.json').is_file():
            try:
                manifest = json.loads((path / 'run.json').read_text(encoding='utf-8'))
            except (ValueError, UnicodeError):
                pass
        return _index_session(path, path, manifest, "")
    if not path.is_file():
        raise FileNotFoundError(f"Session file not found: {path.name}")
    if max_unpacked_bytes < 1:
        raise ValueError("max_unpacked_bytes must be positive")
    archive_hash = file_hash(path)
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        if sum(m.file_size for m in members) > max_unpacked_bytes:
            raise ValueError("Session exceeds maximum unpacked size")
        seen = set()
        for member in members:
            name = member.filename
            parts = PurePosixPath(name).parts
            if (not parts or '\\' in name or PurePosixPath(name).is_absolute()
                    or PureWindowsPath(name).drive or '..' in parts
                    or any(':' in part for part in parts)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Insecure path detected in archive: {name}")
            normalized = '/'.join(parts).casefold()
            if normalized in seen:
                raise ValueError("Duplicate archive member")
            seen.add(normalized)
        manifest = {}
        if 'manifest.json' in zf.namelist():
            try:
                manifest = json.loads(zf.read('manifest.json'))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Invalid session manifest") from exc
            if not isinstance(manifest, dict) or manifest.get('format', 1) != 1:
                raise ValueError("Unsupported PoliScreen session format")
        if target_dir is None:
            scratch = Path(os.environ.get('SHARK_SCRATCH', tempfile.gettempdir())).expanduser()
            scratch.mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix='shark-session-', dir=scratch))
        else:
            root = Path(target_dir).expanduser().resolve()
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise ValueError("Session destination must be an empty directory")
            root.mkdir(parents=True, exist_ok=True)
        try:
            zf.extractall(root)
            return _index_session(path, root, manifest, archive_hash)
        except BaseException:
            shutil.rmtree(root)
            raise


def _index_session(path, root, manifest, archive_hash):
    ranking = _table(root, 'ranking.csv', 'results.csv')
    docking = _table(root, 'docking_results.csv', 'resultados_docking.csv')
    summary = _table(root, 'summary.csv', 'resumen.csv')
    metadata = _table(root, 'ligands_meta.csv')
    rec_paths = [p for folder in ('receptors', 'receptores')
                 for p in (root / folder).glob('*.pdb')]
    if not rec_paths:
        rec_paths = [p for p in root.glob('*.pdb')
                     if 'pose' not in p.stem.lower() and 'complex' not in p.stem.lower()]
    receptors = {p.stem: p for p in rec_paths}
    ligand_files = _unique_stems(
        p for folder in ('input_ligands', 'ligandos_entrada')
        if (root / folder).is_dir()
        for p in (root / folder).iterdir() if p.suffix.lower() in ('.sdf', '.mol', '.mol2')
    ) if any((root / f).is_dir() for f in ('input_ligands', 'ligandos_entrada')) else {}
    ligand_metadata = {}
    if metadata is not None and 'name' in metadata:
        for row in metadata.to_dict('records'):
            key = str(row['name']).casefold()
            if key in ligand_metadata:
                raise ValueError(f"Duplicate ligand metadata: {row['name']}")
            ligand_metadata[key] = row
    session = PoliScreenSession(path, str(manifest.get('project', path.stem)), manifest, root,
                                receptors=receptors, ranking_df=ranking, summary_df=summary,
                                ligand_files=ligand_files, ligand_metadata=ligand_metadata,
                                archive_sha256=archive_hash)
    # Use pose_name and compound_name from the producer's table when available.
    docking_rows = {}
    if docking is not None:
        required = {'pose_name', 'compound_name', 'receptor', 'docking_score'}
        if not required.issubset(docking.columns):
            raise ValueError("Docking table lacks pose identity or score columns")
        for row in docking.to_dict('records'):
            name = str(row['pose_name'])
            if name in docking_rows:
                raise ValueError(f"Duplicate docking pose: {name}")
            docking_rows[name] = row
    pose_dir = root / 'poses'
    candidates = sorted(p for p in (pose_dir.rglob('*') if pose_dir.is_dir() else root.glob('*pose*'))
                        if p.is_file() and p.suffix.lower() in ('.pdb', '.pdbqt', '.sdf'))
    # The unsplit PDBQT contains several models already represented by individual PDBs.
    stems = {p.stem for p in candidates}
    indexed = set()
    for pf in candidates:
        stem = pf.stem
        if pf.suffix.lower() == '.pdbqt' and any(s.startswith(stem + '-model') for s in stems):
            continue
        row = docking_rows.get(stem)
        modern = re.fullmatch(r'docking_(.+)_compounds_a_(.+)-model(\d+)', stem)
        legacy = re.fullmatch(r'(.+?)[_-](?:pose|model|m|p)[_-]?(\d+)', stem)
        if row is not None:
            lig, rec = str(row['compound_name']), str(row['receptor'])
            model = re.search(r'-model(\d+)$', stem)
            if model is None:
                raise ValueError(f"Missing pose index: {stem}")
            idx = int(model[1])
            score = _number(row['docking_score'])
        elif modern:
            rec, lig, idx = modern.groups()
            idx, score = int(idx), None
        elif legacy:
            lig, idx = legacy[1], int(legacy[2])
            if len(receptors) != 1:
                raise ValueError(f"Ambiguous receptor for legacy pose: {stem}")
            rec, score = next(iter(receptors)), None
        else:
            session.warnings.append(f"Unindexed pose file: {pf.name}")
            continue
        key = (rec, lig, idx)
        if key in indexed:
            raise ValueError(f"Duplicate structure for pose: {stem}")
        indexed.add(key)
        matches = pd.DataFrame()
        if ranking is not None:
            column = next((c for c in ('compound', 'ligand', 'name') if c in ranking), None)
            if column:
                mask = ranking[column].astype(str).str.casefold() == lig.casefold()
                if 'receptor' in ranking:
                    mask &= ranking['receptor'].astype(str) == rec
                matches = ranking[mask]
        values = matches.iloc[0].to_dict() if len(matches) == 1 else {}
        if score is None and 'score' in values:
            # Legacy tables carry a per-compound score. Never use best_dock as every pose's energy.
            score = _number(values['score'])
        complexes = [root / folder / f'{prefix}{stem}.pdb'
                     for folder in ('fused_complexes', 'Complejos_Fusionados', 'complejos')
                     for prefix in ('Complex_', 'Complejo_', 'complex_', '')]
        session.poses.append(DockingPose(
            lig, idx, score if score is not None else math.nan, rec, pf,
            next((p for p in complexes if p.is_file()), None),
            efficiency=_number(values.get('LE', values.get('efficiency'))),
            confidence=_number(values.get('confidence')), metadata=values,
        ))
    if not session.poses:
        session.warnings.append('No individual docking poses included; isolated-ligand DFT can use input structures.')
    return session
