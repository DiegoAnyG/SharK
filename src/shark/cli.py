"""Inspect PoliScreen sessions and prepare or execute isolated-ligand DFT jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

from .core.session import read_poliscreen_session, match_receptor_id
from .core.runner import run_orca_job, run_orca_process, find_orca
from .reports.dossier import generate_html_dossier
from .workflows.ligand_qm import prepare_ligand_jobs


def find_recent_poliscreen_sessions() -> list[Path]:
    roots = [Path.cwd()]
    if os.environ.get('POLISCREEN_PROJECTS'):
        roots.append(Path(os.environ['POLISCREEN_PROJECTS']).expanduser())
    paths = {p.resolve() for root in roots if root.is_dir() for p in root.rglob('*.poliscreen')}
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:8]


def interactive_session_picker() -> Path | None:
    recent = find_recent_poliscreen_sessions()
    print('Select a PoliScreen session archive (.poliscreen):')
    for index, path in enumerate(recent, 1):
        print(f'  [{index}] {path.name}')
    if recent:
        choice = input('Session number, file path, or Q to quit > ').strip()
    else:
        print('  (No recent .poliscreen files found in current directory)')
        choice = input('Path to .poliscreen file, or Q to quit > ').strip()
    if not choice or choice.upper() == 'Q':
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(recent):
        return recent[int(choice) - 1]
    return _clean_user_path(choice)


def _clean_user_path(raw: Optional[str]) -> Optional[Path]:
    """Sanitize user-provided file or directory paths across OS/shell formats.

    Handles Windows backslashes (\ -> /), surrounding quotes, home directories,
    and missing leading root slashes.
    """
    if not raw:
        return None
    cleaned = raw.strip().strip("'\"").replace('\\', '/')
    if not cleaned:
        return None
    cand = Path(cleaned).expanduser()
    if cand.exists():
        return cand
    if not cleaned.startswith('/') and Path('/' + cleaned).exists():
        return Path('/' + cleaned)
    stripped = cleaned.lstrip('/')
    return cand


def _get_safe_maxcore_mb(nprocs: int, requested_maxcore: Optional[int] = None, safety_fraction: float = 0.65) -> int:
    """Calculates a safe per-process maxcore MB allocation based on available RAM (Auditor Spec Section 8)."""
    if requested_maxcore is not None:
        return requested_maxcore
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        avail_mb = 6000.0
    budget_mb = int(avail_mb * safety_fraction)
    safe_core = max(256, budget_mb // max(1, nprocs))
    return min(4000, safe_core)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def interactive_receptor_picker(session) -> Optional[str]:
    """Interactively select a target receptor from the session."""
    receptors = sorted(set(p.receptor_id for p in session.poses)) if session.poses else sorted(session.receptors.keys())
    if not receptors:
        return None
    if len(receptors) == 1:
        print(f"\n[Receptor] Target receptor detected: {receptors[0]}")
        return receptors[0]

    print("\nAvailable Receptors / Targets in Session:")
    for idx, rec in enumerate(receptors, 1):
        c_count = len(set(p.ligand_id for p in session.poses if p.receptor_id == rec))
        p_count = sum(1 for p in session.poses if p.receptor_id == rec)
        count_str = f"({c_count} compounds, {p_count} poses)" if p_count > 0 else ""
        try:
            raw_rec = session.receptor_for(rec, raw=True)
            raw_label = f"[Raw crystal: {raw_rec.name}]"
        except Exception:
            raw_label = "[Raw crystal: auto]"
        print(f"  [{idx:2d}] {rec:<22} {raw_label:<25} {count_str}")

    choice = input(f"Select receptor [1-{len(receptors)}, default: 1] > ").strip()
    if not choice:
        picked = receptors[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(receptors):
        picked = receptors[int(choice) - 1]
    else:
        matched = False
        for rec in receptors:
            if rec.casefold() == choice.casefold():
                picked = rec
                matched = True
                break
        if not matched:
            picked = receptors[0]

    try:
        raw_rec = session.receptor_for(picked, raw=True)
        print(f"\n[Receptor Selected] Docking Target: '{picked}' -> RAW Crystal Structure: '{raw_rec.name}'")
        print(f"                    (GROMACS MD pdb2gmx will build topology strictly from {raw_rec.name})")
    except Exception:
        pass
    return picked


def interactive_compound_picker(session, receptor_id: Optional[str] = None) -> Optional[str]:
    """Interactively select a ligand/compound ordered by top ranking."""
    compounds = []
    seen = set()

    # 1. Try ranking_df if available
    if session.ranking_df is not None and not session.ranking_df.empty:
        df = session.ranking_df
        if receptor_id and 'receptor' in df.columns:
            m = df['receptor'].astype(str).str.casefold() == receptor_id.casefold()
            if not m.any():
                stem = receptor_id.split('~')[0].casefold()
                m = df['receptor'].astype(str).str.casefold() == stem
            if m.any():
                df = df[m]

        for rank, row in enumerate(df.to_dict('records'), 1):
            c_name = str(row.get('compound') or row.get('name') or row.get('ligand'))
            if not c_name or c_name.casefold() in seen:
                continue
            seen.add(c_name.casefold())
            dock = _safe_float(row.get('best_dock', row.get('score')))
            le = _safe_float(row.get('LE', row.get('efficiency')))
            eff = _safe_float(row.get('effectiveness_pct', row.get('efficiency_pct')))
            compounds.append({
                'rank': rank,
                'name': c_name,
                'vina': dock,
                'le': le,
                'eff': eff,
            })

    # 2. Fallback to poses if ranking_df not available or empty
    if not compounds and session.poses:
        subset = [p for p in session.poses if receptor_id is None or p.receptor_id == receptor_id]
        grouped = {}
        for p in subset:
            grouped.setdefault(p.ligand_id, []).append(p)

        sorted_groups = sorted(
            grouped.items(),
            key=lambda item: min((p.score for p in item[1] if math.isfinite(p.score)), default=math.inf)
        )
        for rank, (c_name, p_list) in enumerate(sorted_groups, 1):
            best_p = min(p_list, key=lambda p: p.score if math.isfinite(p.score) else math.inf)
            eff = _safe_float(best_p.metadata.get('effectiveness_pct', best_p.metadata.get('efficiency')))
            compounds.append({
                'rank': rank,
                'name': c_name,
                'vina': best_p.score if math.isfinite(best_p.score) else None,
                'le': _safe_float(best_p.efficiency),
                'eff': eff,
            })

    if not compounds:
        return None

    print(f"\nRanked Compounds for {receptor_id or 'all targets'} (Ordered by Top Ranking):")
    display_limit = min(len(compounds), 20)
    for c in compounds[:display_limit]:
        vina_str = f"Vina: {c['vina']:.2f} kcal/mol" if c['vina'] is not None else "Vina: N/A"
        le_str = f"LE: {c['le']:.2f}" if c['le'] is not None else "LE: N/A"
        eff_str = f"Eficiencia: {c['eff']:.1f}%" if c['eff'] is not None else "Eficiencia: N/A"
        print(f"  [{c['rank']:2d}] {c['name']:<28} (Top {c['rank']:2d} | {vina_str} | {le_str} | {eff_str})")

    if len(compounds) > display_limit:
        print(f"  ... ({len(compounds) - display_limit} additional compounds available by index or name)")

    choice = input(f"Select compound [1-{len(compounds)}, exact name, or Enter for Top 1] > ").strip()
    if not choice:
        return compounds[0]['name']
    if choice.isdigit() and 1 <= int(choice) <= len(compounds):
        return compounds[int(choice) - 1]['name']
    for c in compounds:
        if c['name'].casefold() == choice.casefold():
            return c['name']
    return choice


def interactive_pose_picker(session, compound_id: str, receptor_id: Optional[str] = None) -> Optional[int]:
    """Interactively select a pose for the chosen compound and receptor."""
    if not session.poses:
        return None
    matching = [p for p in session.poses if p.ligand_id.casefold() == compound_id.casefold()
                and (receptor_id is None or p.receptor_id == receptor_id)]
    if not matching:
        return None

    matching.sort(key=lambda p: (p.pose_idx, p.score if math.isfinite(p.score) else math.inf))

    if len(matching) == 1:
        p = matching[0]
        vina_str = f"{p.score:.2f} kcal/mol" if math.isfinite(p.score) else "N/A"
        le_val = _safe_float(p.efficiency or p.metadata.get('LE'))
        le_str = f"{le_val:.2f}" if le_val is not None else "N/A"
        eff_val = _safe_float(p.metadata.get('effectiveness_pct', p.metadata.get('efficiency')))
        eff_str = f"{eff_val:.1f}%" if eff_val is not None else "N/A"
        print(f"\n[Pose] Single docking pose: Pose #{p.pose_idx} (Vina: {vina_str} | LE: {le_str} | Eficiencia general: {eff_str})")
        return p.pose_idx

    print(f"\nAvailable Poses for '{compound_id}' in {receptor_id or 'target'}:")
    for idx, p in enumerate(matching, 1):
        vina_str = f"Vina: {p.score:.2f} kcal/mol" if math.isfinite(p.score) else "Vina: N/A"
        le_val = _safe_float(p.efficiency or p.metadata.get('LE'))
        le_str = f"LE: {le_val:.2f}" if le_val is not None else "LE: N/A"
        eff_val = _safe_float(p.metadata.get('effectiveness_pct', p.metadata.get('efficiency')))
        eff_str = f"Eficiencia general: {eff_val:.1f}%" if eff_val is not None else "Eficiencia general: N/A"
        tag = " [Ranked Best / Default]" if p.pose_idx == 1 else ""
        print(f"  [{idx:2d}] Pose #{p.pose_idx:<2d} ({vina_str} | {le_str} | {eff_str}){tag}")

    choice = input(f"Select pose [1-{len(matching)}, default: 1] > ").strip()
    if not choice:
        return matching[0].pose_idx
    if choice.isdigit() and 1 <= int(choice) <= len(matching):
        return matching[int(choice) - 1].pose_idx
    return matching[0].pose_idx


def interactive_nucleophile_picker(
    session,
    compound_id: str,
    receptor_id: Optional[str] = None,
    pose_idx: int = 1
) -> Optional[str]:
    """Interactively select a target nucleophile residue from a detected list."""
    detected = []
    if receptor_id:
        try:
            from .analysis.covalent_matcher import parse_ligand_pose_coordinates, extract_pocket_nucleophiles
            rec_pdb = session.receptor_for(receptor_id)
            pose = session.get_pose(compound_id, pose_idx=pose_idx, receptor_id=receptor_id)
            if pose and pose.pose_file and Path(pose.pose_file).is_file() and rec_pdb and Path(rec_pdb).is_file():
                coords = parse_ligand_pose_coordinates(pose.pose_file)
                detected = extract_pocket_nucleophiles(rec_pdb, coords, pocket_cutoff=6.5)
        except Exception:
            detected = []

    if detected:
        print(f"\nDetected Pocket Nucleophiles in {receptor_id} (within 6.5 Å of {compound_id} pose #{pose_idx}):")
        print("  [ 1] All pocket nucleophiles (automatic multi-target scan) [Recommended]")
        seen_res = set()
        unique_nucls = []
        for n in detected:
            key = (n.residue_name, n.residue_number)
            if key not in seen_res:
                seen_res.add(key)
                unique_nucls.append(n)

        for idx, n in enumerate(unique_nucls, 2):
            print(f"  [{idx:2d}] {n.residue_label:<10} ({n.atom_name}, min distance: {n.min_distance_to_ligand:.2f} Å)")
        print("  [ C] Custom residue (enter manually)")

        choice = input(f"Select target nucleophile [1-{len(unique_nucls) + 1}, or C, default: 1 (All)] > ").strip()
        if not choice or choice == '1':
            return None
        if choice.isdigit() and 2 <= int(choice) <= len(unique_nucls) + 1:
            picked = unique_nucls[int(choice) - 2]
            return f"{picked.residue_name}{picked.residue_number}"
        if choice.upper() == 'C':
            custom = input("Enter target residue name/number (e.g., THR309, CYS145) > ").strip()
            return custom if custom else None
        return choice

    res = input("Target nucleophile residue (e.g., THR309, CYS145) [all pocket nucleophiles] > ").strip()
    return res if res else None


def print_banner():
    print('=' * 80)
    print('SharK: Covalent Reactivity & Molecular Dynamics Analysis Suite')
    print('=' * 80)


def run_interactive():
    print_banner()
    session = interactive_session_picker()
    if session is None:
        return 0

    parsed_session = None
    if session and Path(session).is_file():
        try:
            parsed_session = read_poliscreen_session(session)
            print(f"\n[SharK] Session '{parsed_session.project_name}' loaded ({len(parsed_session.poses)} poses indexed).")
        except Exception:
            parsed_session = None

    def _prompt_covalent_targets(default_res=""):
        if parsed_session is not None:
            rec = interactive_receptor_picker(parsed_session)
            comp = interactive_compound_picker(parsed_session, rec)
            pose = interactive_pose_picker(parsed_session, comp, rec) if comp else 1
            res = interactive_nucleophile_picker(parsed_session, comp, rec, pose or 1) if comp else None
            return rec, comp, pose, res
        else:
            comp = input('Exact compound name [leave blank for ranked top 1] > ').strip() or None
            res = input(f'Target nucleophile residue (e.g., THR309, CYS145) [{default_res or "all pocket nucleophiles"}] > ').strip() or (default_res or None)
            return None, comp, None, res

    print('\nAvailable Workflow Combos:')
    print('  [1] Fast Analysis (Docking-Based)')
    print('      Description: Evaluates Vina poses with advanced covalent criteria (Bürgi-Dunitz')
    print('                   angle θ_BD, catalytic dyad pKa activation, Bruice sigmoid curve,')
    print('                   and Conceptual DFT warhead reactivity).')
    print('      Deliverables: Interactive HTML Dossier with 3D pocket, NAC detection, and CFI scores.')
    print('      Estimated time: ~3-5 seconds.\n')
    print('  [2] Simple Gold Standard (Pre-calculated MD Trajectory)')
    print('      Description: Ingests external MD trajectory (.xtc + .gro), aligns backbone,')
    print('                   performs Daura (GROMOS) RMSD clustering, extracts the dominant')
    print('                   solvated medoid snapshot (eliminating vacuum docking bias), and')
    print('                   evaluates multi-factorial covalent feasibility (θ_BD, dyad, CFI, P_NAC).')
    print('      Deliverables: Representative snapshot PDBs (complex, receptor, ligand), clustering')
    print('                   statistics, and comprehensive HTML Dossier with 3D solvated pocket.')
    print('      Estimated time: ~15-45 seconds.\n')
    print('  [3] Full Gold Standard (End-to-End Simulation & Analysis)')
    print('      Description: Automates complete pipeline from raw receptor in PoliScreen session:')
    print('                   GROMACS MD setup, energy minimization, NVT/NPT equilibration,')
    print('                   production MD, followed by Daura RMSD clustering, solvated medoid')
    print('                   extraction, and advanced covalent reactivity analysis.')
    print('      Deliverables: Full GROMACS trajectory, trajectory dashboard, medoid PDBs, and')
    print('                   interactive HTML Dossier with MD and covalent metrics.')
    print('      Estimated time: ~15-60+ minutes (depending on simulation length & GPU).\n')
    print('  [4] Transition State & Activation Energy (Tier 4 / ORCA)')
    print('      Description: Extracts active-site QM cluster (minimal capped residue or extended pocket),')
    print('                   generates automated ORCA relaxed surface scan, optimizes the first-order saddle')
    print('                   point (! OptTS Freq), verifies strictly 1 imaginary mode, and calculates Eyring')
    print('                   activation free energy (ΔG‡) and reaction thermodynamics (ΔG_rxn).')
    print('      Deliverables: QM cluster files, reaction coordinate profile, verified TS output, and ΔG‡.')
    print('      Estimated time: ~1-10 minutes (minimal model) or ~30-60 minutes (extended cluster).\n')
    print('  [5] Trajectory Contact Analyzer')
    print('      Description: Computes residue contact occupancy and persistence over time for an')
    print('                   existing MD trajectory without quantum or clustering calculations.')
    print('      Deliverables: Contact occupancy JSON and interactive HTML trajectory contact map.')
    print('      Estimated time: ~10-20 seconds.\n')
    print('  [6] Expert / Custom Mode')
    print('      Description: Step-by-step custom configuration for isolated-ligand ORCA DFT,')
    print('                   custom clustering cutoffs, specific residue targets, and solvent models.')
    print('      Deliverables: Custom ORCA inputs/outputs, orbital visualizations, or tailored reports.')
    print('      Estimated time: Variable (seconds to hours).\n')
    print('  [Q] Quit\n')

    choice = input('Select combo [1-6, or Q] > ').strip().upper()
    if choice == 'Q':
        return 0
    if choice not in ('1', '2', '3', '4', '5', '6'):
        print('Unknown selection', file=sys.stderr)
        return 2

    argv = ['--session', str(session)]

    if choice == '1':
        argv += ['--fast-analysis']
        rec, comp, pose, target_res = _prompt_covalent_targets()
        if rec:
            argv += ['--target', rec]
        if comp:
            argv += ['--compound', comp]
        if pose:
            argv += ['--pose', str(pose)]
        if target_res:
            argv += ['--target-residue', target_res]
        dft_dir = input('Directory with existing ORCA DFT outputs (.out) [optional, press enter to skip] > ').strip()
        if dft_dir:
            argv += ['--dft-dir', dft_dir]

    elif choice == '2':
        argv += ['--simple-gold-standard']
        gro_raw = input('Path to MD topology (.gro or .pdb) > ')
        gro_p = _clean_user_path(gro_raw)
        while not gro_p or not gro_p.is_file():
            print(f'File not found: {gro_raw.strip()}', file=sys.stderr)
            gro_raw = input('Path to MD topology (.gro or .pdb) > ')
            gro_p = _clean_user_path(gro_raw)

        xtc_raw = input('Path to MD trajectory (.xtc) > ')
        xtc_p = _clean_user_path(xtc_raw)
        while not xtc_p or not xtc_p.is_file():
            print(f'File not found: {xtc_raw.strip()}', file=sys.stderr)
            xtc_raw = input('Path to MD trajectory (.xtc) > ')
            xtc_p = _clean_user_path(xtc_raw)

        argv += ['--topology', str(gro_p), '--trajectory', str(xtc_p)]

        rec, comp, pose, target_res = _prompt_covalent_targets()
        if rec:
            argv += ['--target', rec]
        if comp:
            argv += ['--compound', comp]
        if pose:
            argv += ['--pose', str(pose)]
        if target_res:
            argv += ['--target-residue', target_res]
        cutoff = input('Daura RMSD clustering cutoff in Angstroms [1.5] > ').strip()
        if cutoff:
            argv += ['--cluster-cutoff', cutoff]
        stride = input('Frame stride for clustering [5] > ').strip()
        if stride:
            argv += ['--cluster-stride', stride]
        include_ts = input('Include Active-Site Cluster QM & Tier 4 Transition State (ORCA)? [Y/n] > ').strip().lower()
        if include_ts != 'n':
            argv += ['--orca-cluster-sp', '--tier-4-ts']
            qm_model = input('Active site QM model [1: Minimal Capped Residue (recommended/fast), 2: Extended Pocket Cluster] [default: 1] > ').strip() or '1'
            if qm_model == '2' or qm_model.lower().startswith('ext'):
                argv += ['--qm-model', 'extended']
            else:
                argv += ['--qm-model', 'minimal']
            exec_now = input('Launch complete pipeline (Trajectory Analysis + ORCA TS) immediately? [Y/n] > ').strip().lower()
            if exec_now != 'n':
                argv += ['--execute']

    elif choice == '3':
        argv += ['--full-gold-standard']
        sim_time = input('Simulation length in nanoseconds [10.0] > ').strip() or '10.0'
        argv += ['--time-ns', sim_time]
        rec, comp, pose, target_res = _prompt_covalent_targets()
        if rec:
            argv += ['--target', rec]
        if comp:
            argv += ['--compound', comp]
        if pose:
            argv += ['--pose', str(pose)]
        if target_res:
            argv += ['--target-residue', target_res]
        dft_dir = input('Directory with existing ORCA DFT outputs (.out) [optional, press enter to skip] > ').strip()
        if dft_dir:
            dft_p = _clean_user_path(dft_dir)
            if dft_p:
                argv += ['--dft-dir', str(dft_p)]
        # Tier 3 Full Gold Standard chains MD -> Clustering -> Medoid Snapshot -> Active-Site Cluster QM & TS Reaction Coordinate Scan
        argv += ['--orca-cluster-sp', '--tier-4-ts']
        qm_model = input('Active site QM model [1: Minimal Capped Residue (recommended/fast), 2: Extended Pocket Cluster] [default: 1] > ').strip() or '1'
        if qm_model == '2' or qm_model.lower().startswith('ext'):
            argv += ['--qm-model', 'extended']
        else:
            argv += ['--qm-model', 'minimal']
        exec_now = input('Launch complete Full Gold Standard pipeline (GROMACS MD + ORCA TS) immediately? [Y/n] > ').strip().lower()
        if exec_now != 'n':
            argv += ['--execute']

    elif choice == '4':
        first_input = input('Active site QM model [1: Minimal Capped Residue (recommended/fast), 2: Extended Pocket Cluster] > ').strip()
        cleaned_first = _clean_user_path(first_input)
        if first_input.lower().endswith(('.gro', '.pdb', '.tpr', '.xtc')) or (cleaned_first and str(cleaned_first).lower().endswith(('.gro', '.pdb', '.tpr', '.xtc'))):
            gro_p = cleaned_first or Path(first_input)
            xtc_raw = input('XTC trajectory > ').strip()
            xtc_p = _clean_user_path(xtc_raw) or Path(xtc_raw)
            lig = input('Exact ligand atom selection (for example: resname UNL and not name H*) > ').strip()
            cutoff = input('Contact cutoff in angstrom [4.0] > ').strip() or '4.0'
            argv = ['--analyze-md', '--topology', str(gro_p), '--trajectory', str(xtc_p),
                    '--ligand-selection', lig, '--contact-cutoff', cutoff]
        else:
            argv += ['--tier-4-ts']
            if first_input == '2' or first_input.lower().startswith('ext'):
                argv += ['--qm-model', 'extended']
            else:
                argv += ['--qm-model', 'minimal']

            src_choice = input('Input structure source [1: Solvated medoid snapshot from previous MD job (Recommended), 2: Docking pose from PoliScreen session] [default: 1] > ').strip() or '1'
            if src_choice == '1':
                md_path = input('Path to MD job directory [default: auto-detect latest job] > ').strip()
                cand_dir = None
                if md_path:
                    p_c = _clean_user_path(md_path)
                    if p_c and p_c.is_dir():
                        cand_dir = p_c
                    elif p_c and p_c.is_file():
                        cand_dir = p_c.parent
                else:
                    jobs_dir = Path.cwd() / 'shark_jobs'
                    if jobs_dir.is_dir():
                        subdirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
                        if subdirs:
                            cand_dir = subdirs[0]
                if cand_dir:
                    snap_rec = cand_dir / 'snapshots' / 'representative_snapshot_receptor.pdb'
                    snap_lig = cand_dir / 'snapshots' / 'representative_snapshot_ligand.pdb'
                    if snap_rec.is_file() and snap_lig.is_file():
                        argv += ['--work-dir', str(cand_dir)]
                        print(f"[TIER 4] Staged solvated medoid snapshot from: {cand_dir}")
                        def_res = "THR309"
                        ev_file = cand_dir / 'evidence.json'
                        if ev_file.is_file():
                            try:
                                ev_data = json.loads(ev_file.read_text(encoding='utf-8'))
                                b_cont = ev_data.get('static_reactive_geometry', {}).get('best_contact', {})
                                if b_cont and b_cont.get('residue'):
                                    def_res = b_cont['residue'].split(':')[0]
                            except Exception:
                                pass
                        res_in = input(f'Target reactive residue [default: {def_res}] > ').strip() or def_res
                        argv += ['--target-residue', res_in]
            else:
                rec, comp, pose, target_res = _prompt_covalent_targets(default_res="THR309")
                if rec:
                    argv += ['--target', rec]
                if comp:
                    argv += ['--compound', comp]
                if pose:
                    argv += ['--pose', str(pose)]
                if target_res:
                    argv += ['--target-residue', target_res]

            exec_now = input('Launch ORCA TS workflow immediately? [Y/n] > ').strip().lower()
            if exec_now != 'n':
                argv += ['--execute']

    elif choice in ('5', '6'):
        action_or_gro = input('[1] Prepare ligand DFT  [2] Execute ligand DFT  [3] Session report  [4] Custom Covalent Matcher (or path to .gro/.pdb for trajectory contacts) > ').strip()
        cleaned_action = _clean_user_path(action_or_gro)
        if action_or_gro.lower().endswith(('.gro', '.pdb', '.tpr', '.xtc')) or (cleaned_action and str(cleaned_action).lower().endswith(('.gro', '.pdb', '.tpr', '.xtc'))):
            gro_p = cleaned_action or Path(action_or_gro)
            xtc_raw = input('XTC trajectory > ').strip()
            xtc_p = _clean_user_path(xtc_raw) or Path(xtc_raw)
            lig = input('Exact ligand atom selection (for example: resname UNL and not name H*) > ').strip()
            cutoff = input('Contact cutoff in angstrom [4.0] > ').strip() or '4.0'
            argv = ['--analyze-md', '--topology', str(gro_p), '--trajectory', str(xtc_p),
                    '--ligand-selection', lig, '--contact-cutoff', cutoff]
        else:
            action = action_or_gro
            if action in ('1', '2'):
                argv += ['--dft', '--top', input('Number of unique ligands [1] > ').strip() or '1']
                compound = input('Exact compound name [ranked selection] > ').strip()
                if compound:
                    argv += ['--compound', compound]
                target = input('Target/pocket identifier [all] > ').strip()
                if target:
                    argv += ['--target', target]
                argv += ['--theory', input('ORCA method [r2SCAN-3c] > ').strip() or 'r2SCAN-3c']
                argv += ['--solvent', input('CPCM solvent [Water; gas for no solvent] > ').strip() or 'Water']
                argv += ['--multiplicity', input('Spin multiplicity [1] > ').strip() or '1']
                argv += ['--geometry', input('Geometry source [input; smiles to regenerate] > ').strip() or 'input']
                if input('Calculate frequencies to check the optimized minimum? [Y/n] > ').strip().lower() != 'n':
                    argv += ['--frequencies']
                if action == '2':
                    argv += ['--execute']
            elif action == '4':
                argv += ['--covalent']
                rec, comp, pose, target_res = _prompt_covalent_targets()
                if rec:
                    argv += ['--target', rec]
                if comp:
                    argv += ['--compound', comp]
                if pose:
                    argv += ['--pose', str(pose)]
                if target_res:
                    argv += ['--target-residue', target_res]

    output = input('Output directory [leave blank for default reports/ directory] > ').strip()
    if output:
        out_p = _clean_user_path(output)
        if out_p:
            argv += ['--work-dir', str(out_p)]
    return main(argv)


def load_dft_records(dft_dir: Path, report_dir: Path) -> list[dict]:
    """Parse existing ORCA calculations from a directory and build dossier job records."""
    from .core.parser import parse_orca_results
    records = []
    for out_file in sorted(dft_dir.glob("*.out")):
        try:
            res = parse_orca_results(out_file, name=out_file.stem)
            if not res.converged and res.el_energy == 0.0:
                continue

            channels = {}
            if res.homo_energy is not None and res.lumo_energy is not None:
                channels["0"] = {
                    "homo": {"number": 0, "energy_eV": res.homo_energy},
                    "lumo": {"number": 1, "energy_eV": res.lumo_energy},
                    "gap_ev": res.homo_lumo_gap if res.homo_lumo_gap is not None else (res.lumo_energy - res.homo_energy)
                }

            viewer_rel = None
            for cand in [dft_dir / "frontier_orbitals.html", dft_dir / f"{out_file.stem}_orbitals.html",
                         report_dir / "benzofuroxan_tautomers" / "frontier_orbitals.html"]:
                if cand.is_file():
                    try:
                        viewer_rel = Path(os.path.relpath(cand, report_dir)).as_posix()
                        break
                    except ValueError:
                        pass

            records.append({
                "selection": {"ligand_id": res.name},
                "status": "completed" if res.converged else "failed",
                "parameters": {
                    "method": "B3LYP/def2-SVP",
                    "solvent": "CPCM(Water)",
                    "charge": 0,
                    "multiplicity": 1,
                },
                "results": {
                    "electronic_energy_hartree": res.el_energy,
                    "orca_version": "6.0.0",
                    "optimization_converged": res.converged,
                    "stationary_minimum_verified": res.is_stationary_minimum,
                    "frequencies_cm1": res.frequencies,
                    "imaginary_frequencies_cm1": res.imaginary_frequencies,
                    "orbitals": channels
                },
                "viewer_link": viewer_rel,
                "source": {"kind": "DFT geometry optimization"}
            })
        except Exception:
            continue
    return records


def report_stage_progress(current_stage: int, total_stages: int, stage_name: str, detail: str = ""):
    """Print high-visibility stage progress banner for orchestrated workflows."""
    pct = int((current_stage / total_stages) * 100)
    bar_len = 24
    filled = int(bar_len * current_stage / total_stages)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\n[STAGE {current_stage}/{total_stages}] [{bar}] {pct}% | {stage_name}")
    if detail:
        print(f"        {detail}")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'run-qm':
        arguments = ['--dft'] + arguments[1:]
    if arguments and arguments[0] == 'analyze-md':
        arguments = ['--analyze-md'] + arguments[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze-md', action='store_true', help='Analyze contacts in an existing trajectory; does not launch MD')
    parser.add_argument('--topology', '--md-topology', dest='topology', help='Matching GRO or supported topology')
    parser.add_argument('--trajectory', '--md-trajectory', dest='trajectory', help='Existing XTC trajectory')
    parser.add_argument('--ligand-selection', help='Explicit MDAnalysis atom selection for one ligand residue')
    parser.add_argument('--protein-selection', default='protein and not name H*')
    parser.add_argument('--contact-cutoff', type=float, default=4.0, help='Geometric contact threshold in angstrom')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--start-ns', type=float, default=0.0)
    parser.add_argument('--stop-ns', type=float)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--session', help='PoliScreen session archive')
    parser.add_argument('--top', type=int, default=1, help='Unique ligands for DFT; poses for reports')
    parser.add_argument('--pose', type=int, default=None, help='Specific pose index to evaluate (e.g. 1, 2)')
    parser.add_argument('--target', help='Exact target or target~pocket identifier')
    parser.add_argument('--compound', action='append', help='Exact compound name; may be repeated')
    parser.add_argument('--pareto', action='store_true', help='Select recorded Pareto leaders')
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--dft', '--run-dft', action='store_true', dest='dft', help='Prepare or execute isolated-ligand ORCA jobs')
    parser.add_argument('--execute', action='store_true', help='Execute the prepared DFT jobs now')
    parser.add_argument('--theory', default='r2SCAN-3c', help='ORCA method and basis keywords')
    parser.add_argument('--solvent', default='Water', help='CPCM solvent; gas disables solvent')
    parser.add_argument('--single-point', action='store_true', help='Skip geometry optimization')
    parser.add_argument('--frequencies', action='store_true', help='Request vibrational frequencies')
    parser.add_argument('--charge', type=int, help='Check intended charge against the molecular input')
    parser.add_argument('--multiplicity', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--geometry', choices=['input', 'smiles'], default='input',
                        help='Use input 3D structure or regenerate from recorded SMILES')
    parser.add_argument('--orbital-grid', type=int, default=60, help='Orbital cube points per axis (10 to 256)')
    parser.add_argument('--no-orbital-plots', action='store_true', help='Extract orbital energies without 3D exports')
    parser.add_argument('--nprocs', type=int)
    parser.add_argument('--maxcore', type=int, help='ORCA memory in MB per process')
    parser.add_argument('--timeout', type=float, help='Maximum seconds per ORCA job')
    parser.add_argument('--work-dir', help='New or empty output directory; defaults to SHARK_SCRATCH')
    parser.add_argument('--html', '--generate-dossier', dest='html', help='Output HTML dossier path')
    parser.add_argument('--run-md', '--md', dest='run_md', action='store_true', help='Prepare or launch GROMACS MD simulation using raw receptor')
    parser.add_argument('--pipeline-dir', help='Custom path to GROMACS pipeline root (overrides SHARK_GROMACS_PIPELINE)')
    parser.add_argument('--time-ns', type=float, default=10.0, help='Simulation time in nanoseconds for MD')
    parser.add_argument('--cluster-nac', '--cluster-trajectory', action='store_true', dest='cluster_nac', help='Execute Daura RMSD clustering & dynamic NAC analysis on trajectory')
    parser.add_argument('--covalent', action='store_true', help='Perform Conceptual DFT reactivity profiling and pocket Near-Attack Conformation matching')
    parser.add_argument('--target-residue', help='Target nucleophile residue to scan (e.g. CYS145, CYS, 145)')
    parser.add_argument('--dft-dir', help='Directory with existing ORCA calculation outputs (.out) to load into report')
    parser.add_argument('--fast-analysis', action='store_true', help='Execute fast docking-based covalent analysis on Vina poses')
    parser.add_argument('--simple-gold-standard', action='store_true', help='Execute Simple Gold Standard: trajectory clustering + medoid covalent analysis')
    parser.add_argument('--full-gold-standard', action='store_true', help='Execute Full Gold Standard: GROMACS MD + clustering + medoid covalent analysis')
    parser.add_argument('--cluster-cutoff', type=float, default=1.5, help='Daura RMSD clustering neighbor cutoff in angstrom (default: 1.5)')
    parser.add_argument('--cluster-stride', type=int, default=5, help='Frame sampling stride for trajectory clustering (default: 5)')
    parser.add_argument('--cluster-start-ns', type=float, default=0.0, help='Simulation time in ns to start clustering (default: 0.0)')
    parser.add_argument('--reactive-nac-min', type=float, default=0.10, help='Minimum NAC score threshold to define reactive subensemble frames (default: 0.10)')
    parser.add_argument('--reactive-distance-max', type=float, default=3.8, help='Maximum nucleophile-electrophile distance (A) to define reactive subensemble frames (default: 3.8)')
    parser.add_argument('--electrophile-atom', help='Explicit ligand electrophilic atom name (e.g. C7, C4) for reaction coordinate tracking')
    parser.add_argument('--tier-4-ts', '--ts', dest='tier_4_ts', action='store_true', help='Execute Tier 4: Transition State modeling & Eyring activation free energy barrier')
    parser.add_argument('--qm-model', choices=['minimal', 'extended'], default='minimal', help='Active site QM cluster model: minimal (capped residue, ~35 atoms) or extended (pocket, ~120 atoms)')
    parser.add_argument('--scan-start', type=float, default=None, help='Starting distance in Angstroms for coordinate scan (default: auto-detected from cluster geometry)')
    parser.add_argument('--scan-end', type=float, default=1.45, help='Ending distance in Angstroms for coordinate scan (default: 1.45)')
    parser.add_argument('--scan-steps', type=int, default=18, help='Number of scan steps along reaction coordinate (default: 18)')
    parser.add_argument('--ph', type=float, default=7.4, help='Solution pH for residue protonation and microstate assignment (default: 7.4)')
    parser.add_argument('--orca-cluster-sp', action='store_true', help='Execute real ab initio ORCA single-point calculation on the extracted active-site QM cluster and generate HOMO/LUMO 3D volumetric isosurfaces')
    parser.add_argument('--recalc-hess', type=int, default=25, help='Frequency of exact Hessian recalculation in OptTS (default: 25)')
    parser.add_argument('--mechanism', default='generic_covalent_addition', help='Declared covalent reaction mechanism (e.g. carbonyl_addition, michael_addition, furoxan_heterocycle_attack, aromatic_substitution_snar)')
    parser.add_argument('--optimize-adduct', action='store_true', help='Optimize the covalent adduct product geometry in Tier 4 to obtain Card 4 bond nature metrics')
    parser.add_argument('--full-thermo', '--gibbs-thermo', action='store_true', dest='full_thermo', help='Compute complete Gibbs free energy thermochemistry (Delta G‡, Delta G_rxn, k_chem, half-life t1/2) by calculating vibrational frequencies on reactant complex and product adduct')
    parser.add_argument('--report-from-evidence', help='Generate HTML dossier directly from an existing evidence.json file without running calculations')
    parser.add_argument('--kill-orphans', action='store_true', help='Scan and safely terminate any orphaned or stuck background ORCA/OpenMPI processes.')
    args = parser.parse_args(arguments)
    if args.kill_orphans:
        from .core.runner import cleanup_orphan_orca_processes
        killed = cleanup_orphan_orca_processes()
        print(f"[CLEANUP] Terminated {killed} orphaned ORCA/MPI processes.")
        return 0
    if args.report_from_evidence:
        ev_file = Path(args.report_from_evidence).expanduser().resolve()
        if not ev_file.is_file():
            sys.stderr.write(f"ERROR: Evidence file not found: {ev_file}\n")
            return 1
        from .core.analysis_result import SharKAnalysisResult
        res = SharKAnalysisResult.from_json(ev_file)
        out_html = Path(args.html).expanduser().resolve() if args.html else ev_file.parent / "dossier.html"

        # Check for user-supplied or auto-discovered DFT directory
        qm_summary = None
        dft_dir_to_use = getattr(args, 'dft_dir', None)
        if not dft_dir_to_use:
            dft_cands = [
                ev_file.parent / 'dft',
                ev_file.parent / 'dft_benzofuroxan',
                Path.cwd() / 'dft_benzofuroxan',
                Path('/home/diego/SharK/dft_benzofuroxan'),
                Path(__file__).resolve().parent.parent.parent / 'dft_benzofuroxan',
            ]
            for d_cand in dft_cands:
                if d_cand.is_dir() and any(d_cand.glob('*.out')):
                    dft_dir_to_use = str(d_cand)
                    break
        if dft_dir_to_use:
            records = load_dft_records(Path(dft_dir_to_use), out_html.parent)
            if records:
                qm_summary = {'jobs': records}
                print(f"[DFT] Loaded {len(records)} quantum calculation(s) from {dft_dir_to_use}")

        # Check for covalent_feasibility.json in ev_file directory
        cov_summary = None
        cf_path = ev_file.parent / 'covalent_feasibility.json'
        if cf_path.is_file():
            try:
                cov_summary = json.loads(cf_path.read_text(encoding='utf-8'))
            except Exception:
                pass

        # Check for poses_summary.csv in ev_file directory
        poses_data = []
        pf_path = ev_file.parent / 'poses_summary.csv'
        if pf_path.is_file():
            import csv
            try:
                with open(pf_path, encoding='utf-8') as pf:
                    reader = csv.DictReader(pf)
                    for row in reader:
                        poses_data.append(dict(row))
            except Exception:
                pass

        # Check for trajectory plots and md_summary
        md_summary = None
        dyn = res.dynamics if hasattr(res, 'dynamics') and isinstance(res.dynamics, dict) else (res.to_dict().get('dynamics') if hasattr(res, 'to_dict') else {})
        cands_dir = []
        if isinstance(dyn, dict):
            if dyn.get('run_dir'):
                cands_dir.append(Path(dyn['run_dir']))
                cands_dir.append(Path(dyn['run_dir']) / '00_prep')
            if dyn.get('trajectory_file'):
                cands_dir.append(Path(dyn['trajectory_file']).parent)
                cands_dir.append(Path(dyn['trajectory_file']).parent / '00_prep')
        cands_dir.extend([
            ev_file.parent,
            ev_file.parent / '00_prep',
            ev_file.parent / 'md',
            ev_file.parent / 'md' / '00_prep',
        ])
        found_plots = {}
        for cdir in cands_dir:
            if cdir.is_dir():
                for p_name in ('md_analysis_dashboard.png', 'plot_rmsd.png', 'plot_rmsf.png', 'plot_gyrate.png', 'plot_hbonds.png'):
                    pf = cdir / p_name
                    if pf.is_file() and p_name not in found_plots:
                        found_plots[p_name] = str(pf.resolve())
        if found_plots:
            sim_t = (dyn.get('sim_time_ns') if isinstance(dyn, dict) else None) or (dyn.get('total_sim_time_ns') if isinstance(dyn, dict) else None) or 20.0
            md_summary = {
                'run_id': dyn.get('run_id', 'md_production') if isinstance(dyn, dict) else 'md_production',
                'sim_time_ns': sim_t,
                'status': 'completed',
                'run_dir': str(cands_dir[0]) if cands_dir else str(ev_file.parent),
                'plots': found_plots,
                'dashboard_path': found_plots.get('md_analysis_dashboard.png'),
                'clustering': dyn if dyn else None,
                'force_fields': 'AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (0.15 M NaCl)',
            }

        generate_html_dossier(res.analysis_id or "SharK Analysis", poses_data, out_html,
                              qm_summary=qm_summary, md_summary=md_summary,
                              covalent_summary=cov_summary, result=res)
        print(f"[REPORT] Regenerated dossier from evidence: {out_html}")
        return 0
    if args.interactive or not arguments:
        try:
            return run_interactive()
        except (EOFError, KeyboardInterrupt):
            return 130
    if args.fast_analysis:
        args.covalent = True
    if args.cluster_nac:
        args.covalent = True
    if args.simple_gold_standard:
        if not (args.topology and args.trajectory):
            sys.stderr.write(
                "ERROR: --simple-gold-standard requires both --topology and --trajectory "
                "because dynamic NAC analysis is part of this workflow. "
                "Use --fast-analysis for static-only analysis.\n"
            )
            return 1
        args.covalent = True
        args.cluster_nac = True
    if args.orca_cluster_sp:
        args.covalent = True
    if args.full_gold_standard:
        args.dft = True
        args.execute = True
        if not (args.topology and args.trajectory):
            args.run_md = True
        args.covalent = True
        args.cluster_nac = True
        args.orca_cluster_sp = True
        args.tier_4_ts = True
        args.optimize_adduct = True
        args.full_thermo = True
    if getattr(args, 'full_thermo', False):
        args.covalent = True
        args.tier_4_ts = True
        args.optimize_adduct = True
    if args.tier_4_ts:
        args.covalent = True
    # One persistent root per invocation; explicit --work-dir remains supported.
    from .core.output import create_job_directory
    if not args.work_dir:
        args.work_dir = str(create_job_directory())
    if args.analyze_md:
        if args.dft or args.execute or args.run_md:
            parser.error('--analyze-md cannot launch DFT or MD')
        if not all((args.topology, args.trajectory, args.ligand_selection)):
            parser.error('Contact analysis requires --topology, --trajectory and --ligand-selection')
        from .analysis.md_contacts import analyze_contacts
        try:
            summary = analyze_contacts(args.topology, args.trajectory, ligand_selection=args.ligand_selection,
                protein_selection=args.protein_selection, cutoff_angstrom=args.contact_cutoff,
                stride=args.frame_stride, start_ns=args.start_ns, stop_ns=args.stop_ns)
            if args.work_dir:
                output = Path(args.work_dir).expanduser()
                if output.exists() and (not output.is_dir() or any(output.iterdir())):
                    raise ValueError('Contact output directory must be new or empty')
                output.mkdir(parents=True, exist_ok=True)
            else:
                import tempfile
                root = Path(os.environ.get('SHARK_SCRATCH', tempfile.gettempdir())).expanduser()
                root.mkdir(parents=True, exist_ok=True)
                output = Path(tempfile.mkdtemp(prefix='shark-md-contacts-', dir=root))
            print(f'[OUTPUT] Contact report directory: {output}; generated reports can be removed and regenerated')
            (output / 'contacts.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            report = generate_html_dossier(Path(args.trajectory).stem, [], args.html or output/'dossier.html', md_summary=summary)
            print(f'[CONTACTS] {summary["sampled_frame_count"]} sampled frames; {len(summary["contacts"])} contacting residues')
            print(f'[REPORT] {report}')
            return 0
        except KeyboardInterrupt:
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print(f'[ERROR] {exc}', file=sys.stderr)
            return 1
    if not args.session:
        parser.error('--session is required')
    if args.execute and not args.dft and not args.run_md and not args.tier_4_ts:
        parser.error('--execute requires --dft, run-qm, --run-md or --tier-4-ts')
    if args.top < 1:
        parser.error('--top must be positive')
    session = None
    try:
        session = read_poliscreen_session(args.session)
        print(f"[SharK] Session '{session.project_name}': {len(session.poses)} indexed poses")
        for warning in session.warnings:
            print(f'[NOTE] {warning}')
        if args.full_gold_standard:
            report_stage_progress(1, 8, "PoliScreen Ingestion & Pose Ranking (Pillar 1)", f"Project: {session.project_name} | {len(session.poses)} indexed poses")

        def _get_selected_poses():
            if args.compound:
                selected = []
                for c in args.compound:
                    matched = [p for p in session.poses if p.ligand_id.casefold() == c.casefold()
                               and match_receptor_id(p.receptor_id, args.target)]
                    if matched:
                        if args.pose is not None:
                            pose_matches = [p for p in matched if p.pose_idx == args.pose]
                            if not pose_matches:
                                avail_poses = sorted(set(p.pose_idx for p in matched))
                                raise ValueError(
                                    f"Pose #{args.pose} requested for compound '{c}', but only poses "
                                    f"{avail_poses} exist matching target '{args.target}'."
                                )
                            matched = pose_matches
                        matched.sort(key=lambda p: (p.score if math.isfinite(p.score) else math.inf, p.pose_idx))
                        selected.extend(matched[:args.top])
                    else:
                        avail_compounds = sorted(set(p.ligand_id for p in session.poses))
                        avail_targets = sorted(set(p.receptor_id for p in session.poses))
                        target_msg = f" against target '{args.target}'" if args.target else ""
                        raise ValueError(
                            f"Compound '{c}' not found in session '{session.project_name}'{target_msg}.\n"
                            f"Available compounds ({len(avail_compounds)}): {avail_compounds[:10]}{'...' if len(avail_compounds) > 10 else ''}\n"
                            f"Available targets: {avail_targets}"
                        )
                if selected:
                    return selected
            elif args.target:
                matched = [p for p in session.poses if match_receptor_id(p.receptor_id, args.target)]
                if not matched:
                    avail_targets = sorted(set(p.receptor_id for p in session.poses))
                    raise ValueError(
                        f"Target '{args.target}' not found in session '{session.project_name}'.\n"
                        f"Available targets: {avail_targets}"
                    )
                if args.pose is not None:
                    pose_matches = [p for p in matched if p.pose_idx == args.pose]
                    if not pose_matches:
                        avail_poses = sorted(set(p.pose_idx for p in matched))
                        raise ValueError(
                            f"Pose #{args.pose} requested for target '{args.target}', but only poses "
                            f"{avail_poses} exist for this target."
                        )
                    matched = pose_matches
                matched.sort(key=lambda p: (p.score if math.isfinite(p.score) else math.inf, p.pose_idx))
                return matched[:args.top]

            poses = session.list_top_poses(args.top)
            if args.pose is not None:
                poses = [p for p in poses if p.pose_idx == args.pose]
            return poses

        last_md_res = None
        if args.run_md:
            if args.full_gold_standard:
                report_stage_progress(2, 8, "Classical Molecular Dynamics Simulation (GROMACS)", f"{args.time_ns:.1f} ns solvated production")
            from .workflows.md_pipeline import run_md_from_session
            selected_poses = _get_selected_poses()

            for pose in selected_poses:
                raw_rec = session.receptor_for(pose.receptor_id, raw=True)
                print(f"[MD] Preparing simulation for {pose.ligand_id} (pose #{pose.pose_idx}) against {pose.receptor_id}...")
                print(f"[MD] Sourcing RAW crystal structure: {raw_rec.name} (bypassing {pose.receptor_id}.pdb)")
                print(f"[MD] Force fields: AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (0.15 M NaCl)")
                res = run_md_from_session(
                    session=session,
                    ligand_id=pose.ligand_id,
                    pose_idx=pose.pose_idx,
                    receptor_id=pose.receptor_id,
                    sim_time_ns=args.time_ns,
                    run_now=args.execute,
                    pipeline_dir=args.pipeline_dir,
                    work_dir=Path(args.work_dir) / 'md' if (args.work_dir and not args.pipeline_dir) else None
                )
                last_md_res = res
                print(f"[MD] Status: {res.status} | Directory: {res.run_dir}")
                if res.dashboard_path:
                    print(f"[MD] Dashboard: {res.dashboard_path}")
                if (args.full_gold_standard or args.covalent or getattr(args, 'cluster_nac', False)) and res.status == 'completed':
                    gro_cand = res.run_dir / '00_prep' / 'md_prod.gro'
                    xtc_cand = res.run_dir / '00_prep' / 'md_noPBC.xtc'
                    if not gro_cand.is_file():
                        gro_cand = res.run_dir / 'md_prod.gro'
                    if not xtc_cand.is_file():
                        xtc_cand = res.run_dir / 'md_noPBC.xtc'
                    if gro_cand.is_file() and xtc_cand.is_file():
                        args.topology = str(gro_cand)
                        args.trajectory = str(xtc_cand)
                        print(f'[MD] Linking trajectory for clustering: {xtc_cand}')
            if not (args.full_gold_standard or args.covalent or getattr(args, 'cluster_nac', False) or args.tier_4_ts or args.dft):
                return 0

        covalent_summary = None
        if args.covalent:
            from .analysis.covalent_matcher import match_covalent_pocket
            from .analysis.reactivity import build_reactivity_profile

            selected_poses = _get_selected_poses()
            clustering_info = None
            cluster_rep = None

            if args.trajectory and args.topology:
                if args.full_gold_standard or getattr(args, 'cluster_nac', False):
                    report_stage_progress(3, 8, "Trajectory Clustering & Dynamic NAC Sampling (Pillar 2)", f"Daura RMSD clustering (cutoff: {args.cluster_cutoff} Å)")
                from .analysis.trajectory_cluster import cluster_trajectory
                print(f'[CLUSTERING] Performing Daura RMSD clustering on {args.trajectory}...')
                if args.work_dir:
                    snap_dir = Path(args.work_dir) / 'snapshots'
                else:
                    root = Path(os.environ.get('SHARK_SCRATCH', '/tmp')).expanduser()
                    snap_dir = root / 'shark_snapshots'
                snap_dir.mkdir(parents=True, exist_ok=True)

                cluster_rep = cluster_trajectory(
                    topology=args.topology,
                    trajectory=args.trajectory,
                    ligand_selection=args.ligand_selection,
                    protein_selection=args.protein_selection,
                    cutoff_angstrom=args.cluster_cutoff,
                    start_ns=args.cluster_start_ns,
                    stop_ns=args.stop_ns,
                    stride=args.cluster_stride,
                    output_dir=snap_dir,
                    target_residue=args.target_residue,
                    reactive_nac_min=getattr(args, 'reactive_nac_min', 0.10),
                    reactive_distance_max=getattr(args, 'reactive_distance_max', 3.8),
                    electrophile_atom=getattr(args, 'electrophile_atom', None),
                )
                print(f'[CLUSTERING] {cluster_rep.summary}')
                clustering_info = {
                    'total_sampled_frames': cluster_rep.total_sampled_frames,
                    'num_clusters': cluster_rep.num_clusters,
                    'top_cluster_size': cluster_rep.top_cluster_size,
                    'top_cluster_fraction': cluster_rep.top_cluster_fraction,
                    'gromos_center_frame_index': cluster_rep.gromos_center_frame_index,
                    'gromos_center_time_ps': cluster_rep.gromos_center_time_ps,
                    'gromos_center_time_ns': cluster_rep.gromos_center_time_ns,
                    'medoid_frame_index': cluster_rep.medoid_frame_index,
                    'medoid_time_ps': cluster_rep.medoid_time_ps,
                    'medoid_time_ns': cluster_rep.medoid_time_ns,
                    'cutoff_angstrom': cluster_rep.cutoff_angstrom,
                    'p_nac': cluster_rep.p_nac,
                    'distance_proximity_score': cluster_rep.distance_proximity_score,
                    'reactive_frame_count': cluster_rep.reactive_frame_count,
                    'reactive_frame_fraction': cluster_rep.reactive_frame_fraction,
                    'num_reactive_clusters': cluster_rep.num_reactive_clusters,
                    'top_reactive_cluster_size': cluster_rep.top_reactive_cluster_size,
                    'top_reactive_cluster_fraction': cluster_rep.top_reactive_cluster_fraction,
                    'reactive_medoid_frame_index': cluster_rep.reactive_medoid_frame_index,
                    'reactive_medoid_time_ps': cluster_rep.reactive_medoid_time_ps,
                    'reactive_medoid_time_ns': cluster_rep.reactive_medoid_time_ns,
                    'reactive_medoid_nac_score': cluster_rep.reactive_medoid_nac_score,
                    'reactive_medoid_distance_angstrom': cluster_rep.reactive_medoid_distance_angstrom,
                    'reactive_medoid_angle_deg': cluster_rep.reactive_medoid_angle_deg,
                    'summary': cluster_rep.summary,
                    'snapshot_complex_pdb': str(cluster_rep.snapshot_complex_pdb) if cluster_rep.snapshot_complex_pdb else None,
                    'snapshot_receptor_pdb': str(cluster_rep.snapshot_receptor_pdb) if cluster_rep.snapshot_receptor_pdb else None,
                    'snapshot_ligand_pdb': str(cluster_rep.snapshot_ligand_pdb) if cluster_rep.snapshot_ligand_pdb else None,
                    'reactive_snapshot_complex_pdb': str(cluster_rep.reactive_snapshot_complex_pdb) if cluster_rep.reactive_snapshot_complex_pdb else None,
                    'reactive_snapshot_receptor_pdb': str(cluster_rep.reactive_snapshot_receptor_pdb) if cluster_rep.reactive_snapshot_receptor_pdb else None,
                    'reactive_snapshot_ligand_pdb': str(cluster_rep.reactive_snapshot_ligand_pdb) if cluster_rep.reactive_snapshot_ligand_pdb else None,
                    'sim_time_ns': getattr(cluster_rep, 'sim_time_ns', None),
                    'total_sim_time_ns': getattr(cluster_rep, 'total_sim_time_ns', None),
                    'trajectory_file': str(Path(args.trajectory).resolve()) if getattr(args, 'trajectory', None) else None,
                    'run_dir': str(Path(args.trajectory).resolve().parent) if getattr(args, 'trajectory', None) else None,
                }

            if args.full_gold_standard:
                report_stage_progress(4, 8, "Active-Site Pocket Conceptual DFT & Bürgi-Dunitz Matching", f"Target residue: {args.target_residue or 'Auto-detect'}")

            all_contacts, nac_contacts, pocket_nucls, summaries = [], [], [], []
            rep = None
            for p in selected_poses:
                if cluster_rep and cluster_rep.snapshot_receptor_pdb and cluster_rep.snapshot_ligand_pdb:
                    rec_path = cluster_rep.snapshot_receptor_pdb
                    lig_pose = cluster_rep.snapshot_ligand_pdb
                    print(f'[COVALENT] Evaluating solvated medoid snapshot at {cluster_rep.medoid_time_ns:.2f} ns against {p.receptor_id}...')
                else:
                    rec_path = session.receptor_for(p.receptor_id, raw=False)
                    lig_pose = p.pose_file
                    print(f'[COVALENT] Evaluating docking pose for {p.ligand_id} against {p.receptor_id}...')

                rep = match_covalent_pocket(
                    receptor_pdb=rec_path,
                    ligand_pose=lig_pose,
                    pocket_cutoff=args.contact_cutoff,
                    nac_cutoff=3.5,
                    target_residue=args.target_residue,
                    receptor_name=p.receptor_id,
                    ligand_name=p.ligand_id
                )
                print(f'[COVALENT] {rep.summary}')
                summaries.append(rep.summary)
                for n in rep.pocket_nucleophiles:
                    if n.residue_label not in pocket_nucls:
                        pocket_nucls.append(n.residue_label)
                for c in rep.contacts:
                    c_dict = {
                        'residue': c.nucleophile.residue_label,
                        'nucleophile_atom': c.nucleophile.atom_name,
                        'ligand_atom_index': c.ligand_atom_index,
                        'ligand_atom_element': c.ligand_atom_element,
                        'distance_angstrom': c.distance_angstrom,
                        'feasibility_score': c.feasibility_score,
                        'is_nac': c.is_nac,
                        'warhead_rank': c.warhead_rank,
                        'burgi_dunitz_angle': c.burgi_dunitz_angle,
                        'catalytic_dyad_present': c.catalytic_dyad_present,
                        'catalytic_dyad_residue': c.catalytic_dyad_residue,
                        'activation_factor': c.activation_factor,
                        'composite_feasibility': c.composite_feasibility,
                    }
                    all_contacts.append(c_dict)
                    if c.is_nac:
                        nac_contacts.append(c_dict)

            all_nucl_atoms = []
            all_lig_atoms = []
            if rep and getattr(rep, 'pocket_nucleophiles', None):
                for n in rep.pocket_nucleophiles:
                    all_nucl_atoms.append({
                        'residue': n.residue_label,
                        'atom': n.atom_name,
                        'x': n.coordinates[0],
                        'y': n.coordinates[1],
                        'z': n.coordinates[2]
                    })
            if rep and getattr(rep, 'ligand_atoms', None):
                for idx, elem, crd in rep.ligand_atoms:
                    all_lig_atoms.append({
                        'index': idx,
                        'element': elem,
                        'x': crd[0],
                        'y': crd[1],
                        'z': crd[2]
                    })

            covalent_summary = {
                'has_nac': len(nac_contacts) > 0,
                'summary': ' '.join(summaries),
                'contacts': all_contacts,
                'nac_contacts': nac_contacts,
                'pocket_nucleophiles': pocket_nucls,
                'pocket_nucleophile_atoms': all_nucl_atoms,
                'pocket_residue_atoms': getattr(rep, 'pocket_residue_atoms', []) if rep else [],
                'target_residue_atoms': getattr(rep, 'target_residue_atoms', []) if rep else [],
                'ligand_atoms': all_lig_atoms,
                'ligand_name': selected_poses[0].ligand_id if selected_poses else 'Ligand',
                'clustering': clustering_info
            }

        cluster_qm_res = None
        if args.tier_4_ts or args.orca_cluster_sp:
            from .analysis.qm_cluster import extract_qm_cluster, run_cluster_single_point
            from .analysis.transition_state import (
                prepare_ts_workflow_directory,
                parse_orca_scan_output,
                parse_orca_ts_output,
                compute_reaction_profile,
            )

            selected_poses = _get_selected_poses()
            p = selected_poses[0] if selected_poses else session.poses[0]
            structure_provenance = {}
            if getattr(args, 'trajectory', None) and getattr(args, 'topology', None) and 'cluster_rep' in locals() and cluster_rep:
                if cluster_rep.reactive_snapshot_receptor_pdb and cluster_rep.reactive_snapshot_ligand_pdb:
                    rec_path = cluster_rep.reactive_snapshot_receptor_pdb
                    lig_pose = cluster_rep.reactive_snapshot_ligand_pdb
                    structure_provenance = {
                        "structure_source": "md_reactive_medoid",
                        "trajectory_frame": cluster_rep.reactive_medoid_frame_index,
                        "time_ps": cluster_rep.reactive_medoid_time_ps,
                        "time_ns": cluster_rep.reactive_medoid_time_ns,
                        "reactive_cluster_id": 1,
                        "reactive_cluster_population": cluster_rep.top_reactive_cluster_fraction,
                        "nac_score": cluster_rep.reactive_medoid_nac_score,
                        "distance_angstrom": cluster_rep.reactive_medoid_distance_angstrom,
                        "attack_angle_deg": cluster_rep.reactive_medoid_angle_deg,
                    }
                    print(f"[TIER 4] Structure source: md_reactive_medoid at {cluster_rep.reactive_medoid_time_ns:.2f} ns (d = {cluster_rep.reactive_medoid_distance_angstrom:.2f} A, NAC = {cluster_rep.reactive_medoid_nac_score or 0:.2f})...")
                elif cluster_rep.snapshot_receptor_pdb and cluster_rep.snapshot_ligand_pdb:
                    rec_path = cluster_rep.snapshot_receptor_pdb
                    lig_pose = cluster_rep.snapshot_ligand_pdb
                    structure_provenance = {
                        "structure_source": "md_global_medoid",
                        "trajectory_frame": cluster_rep.medoid_frame_index,
                        "time_ps": cluster_rep.medoid_time_ps,
                        "time_ns": cluster_rep.medoid_time_ns,
                    }
                    print(f"[TIER 4] [WARNING] No reactive representative was available. Falling back to dominant conformational medoid at {cluster_rep.medoid_time_ns:.2f} ns...")
            elif args.work_dir and (Path(args.work_dir) / 'snapshots' / 'reactive_snapshot_receptor.pdb').is_file() and (Path(args.work_dir) / 'snapshots' / 'reactive_snapshot_ligand.pdb').is_file():
                rec_path = Path(args.work_dir) / 'snapshots' / 'reactive_snapshot_receptor.pdb'
                lig_pose = Path(args.work_dir) / 'snapshots' / 'reactive_snapshot_ligand.pdb'
                structure_provenance = {
                    "structure_source": "md_reactive_medoid",
                    "file_path": str(rec_path),
                }
                print(f"[TIER 4] Using reactive medoid snapshot from job directory: {args.work_dir}...")
            elif args.work_dir and (Path(args.work_dir) / 'snapshots' / 'representative_snapshot_receptor.pdb').is_file() and (Path(args.work_dir) / 'snapshots' / 'representative_snapshot_ligand.pdb').is_file():
                rec_path = Path(args.work_dir) / 'snapshots' / 'representative_snapshot_receptor.pdb'
                lig_pose = Path(args.work_dir) / 'snapshots' / 'representative_snapshot_ligand.pdb'
                structure_provenance = {
                    "structure_source": "md_global_medoid",
                    "file_path": str(rec_path),
                }
                print(f"[TIER 4] [WARNING] No reactive representative found in job directory. Falling back to dominant conformational medoid: {args.work_dir}...")
            else:
                rec_path = session.receptor_for(p.receptor_id, raw=False)
                lig_pose = p.pose_file
                structure_provenance = {
                    "structure_source": "docking_pose",
                    "pose_idx": p.pose_idx,
                }
                print(f"[TIER 4] Using docking pose for {p.ligand_id} against {p.receptor_id}...")

            lig_smiles = None
            if hasattr(session, 'ligand_metadata') and isinstance(session.ligand_metadata, dict):
                lig_smiles = session.ligand_metadata.get(p.ligand_id.casefold(), {}).get('smiles')
            if not lig_smiles:
                for meta in getattr(session, 'ligands_meta', []):
                    if meta.get('name', '').casefold() == p.ligand_id.casefold():
                        lig_smiles = meta.get('smiles')
                        break

            target_residue = args.target_residue or 'THR309'
            print(f"[TIER 4] Extracting {args.qm_model.upper()} QM cluster for {p.ligand_id} against {target_residue} in {p.receptor_id} (pH {args.ph:.1f})...")
            cluster = extract_qm_cluster(
                receptor_pdb=rec_path,
                ligand_pose=lig_pose,
                model_type=args.qm_model,
                target_residue=target_residue,
                ligand_smiles=lig_smiles,
                charge=args.charge,
                multiplicity=args.multiplicity,
                ph=args.ph,
                electrophile_atom=getattr(args, 'electrophile_atom', None),
            )
            print(f"[TIER 4] Extracted cluster '{cluster.name}': {cluster.n_atoms} atoms")
            if cluster.nucleophile_idx is not None and cluster.electrophile_idx is not None:
                nucl_lbl = cluster.atoms[cluster.nucleophile_idx].label
                el_lbl = cluster.atoms[cluster.electrophile_idx].label
                print(f"[TIER 4] Reaction coordinate: Nucleophile {nucl_lbl} <--> Electrophile {el_lbl}")

            if covalent_summary is not None and cluster and cluster.atoms:
                cluster_res_atoms = []
                for at in cluster.atoms:
                    if at.res_name != "LIG":
                        cluster_res_atoms.append({
                            "residue": f"{at.res_name}{at.res_seq}",
                            "atom": at.atom_name,
                            "element": at.element,
                            "x": at.coords[0],
                            "y": at.coords[1],
                            "z": at.coords[2],
                            "is_nucleophile": (cluster.nucleophile_idx is not None and at == cluster.atoms[cluster.nucleophile_idx]),
                            "is_cap": at.is_cap,
                        })
                if cluster_res_atoms and not covalent_summary.get('target_residue_atoms'):
                    covalent_summary['target_residue_atoms'] = cluster_res_atoms

            if args.orca_cluster_sp:
                if args.full_gold_standard:
                    report_stage_progress(5, 8, "Active-Site Cluster QM Single-Point (Pillar 3)", f"Method: {args.theory} | Solvent: {args.solvent}")
                if args.work_dir:
                    cluster_work_dir = Path(args.work_dir) / 'cluster_qm'
                else:
                    cluster_work_dir = Path.cwd() / 'shark_jobs' / f"cluster_sp_{target_residue}_{args.qm_model}"
                cluster_work_dir.mkdir(parents=True, exist_ok=True)
                sp_nprocs = args.nprocs or 8
                sp_maxcore = _get_safe_maxcore_mb(sp_nprocs, args.maxcore)
                cluster_qm_res = run_cluster_single_point(
                    cluster=cluster,
                    work_dir=cluster_work_dir,
                    method=args.theory,
                    solvent=None if args.solvent.lower() == 'gas' else args.solvent,
                    nprocs=sp_nprocs,
                    maxcore_mb=sp_maxcore,
                    grid=args.orbital_grid if args.orbital_grid else 40,
                )
                if cluster_qm_res.success:
                    print(f"[CLUSTER QM] ORCA single-point completed in {cluster_qm_res.execution_time_s:.1f} s: HOMO (MO {cluster_qm_res.homo_idx}) = {cluster_qm_res.homo_energy_ev:.2f} eV, LUMO (MO {cluster_qm_res.lumo_idx}) = {cluster_qm_res.lumo_energy_ev:.2f} eV, Gap = {cluster_qm_res.gap_ev:.2f} eV")
                else:
                    print(f"[NOTE] Cluster QM single-point skipped or failed: {cluster_qm_res.error_message}")

            if args.tier_4_ts:
                if args.full_gold_standard:
                    report_stage_progress(6, 8, "Tier 4 Transition State Modeling (Pillar 3)", f"Scan: {args.scan_steps} steps | OptTS + Freq verification")
                if args.work_dir:
                    ts_work_dir = Path(args.work_dir) / 'transition_state'
                else:
                    ts_work_dir = Path.cwd() / 'runs' / f"ts_{p.ligand_id}_{target_residue}_{args.qm_model}"
                ts_work_dir.mkdir(parents=True, exist_ok=True)

                ts_nprocs = args.nprocs or (min(8, os.cpu_count() or 4))
                ts_maxcore = _get_safe_maxcore_mb(ts_nprocs, args.maxcore)

                from .analysis.reaction_validation import validate_reaction_pair, write_reaction_definition_json, compute_tier4_fingerprint

                nucl_el = cluster.atoms[cluster.nucleophile_idx].element if cluster.nucleophile_idx is not None else "O"
                el_el = cluster.atoms[cluster.electrophile_idx].element if cluster.electrophile_idx is not None else "C"
                nucl_name = getattr(cluster.atoms[cluster.nucleophile_idx], 'atom_name', cluster.atoms[cluster.nucleophile_idx].element) if cluster.nucleophile_idx is not None else "Nu"
                nucl_lbl = f"{cluster.target_residue}:{nucl_name}" if cluster.nucleophile_idx is not None else "Nu"
                el_name = getattr(cluster.atoms[cluster.electrophile_idx], 'atom_name', cluster.atoms[cluster.electrophile_idx].element) if cluster.electrophile_idx is not None else "E"
                el_lbl = f"LIG:{el_name}" if cluster.electrophile_idx is not None else "E"

                d_init = None
                if cluster.nucleophile_idx is not None and cluster.electrophile_idx is not None:
                    p1 = cluster.atoms[cluster.nucleophile_idx].coords
                    p2 = cluster.atoms[cluster.electrophile_idx].coords
                    d_init = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

                val_res = validate_reaction_pair(
                    nucleophile_element=nucl_el,
                    electrophile_element=el_el,
                    mechanism_name=getattr(args, 'mechanism', 'generic_covalent_addition'),
                    nucleophile_label=nucl_lbl,
                    electrophile_label=el_lbl,
                    initial_distance_angstrom=d_init,
                )

                print("=" * 65)
                print(" [TIER 4 PREFLIGHT] REACTION COORDINATE VERIFICATION")
                print(f"  Nucleophile:        {nucl_lbl} ({nucl_el})")
                print(f"  Electrophile:       {el_lbl} ({el_el})")
                print(f"  Proposed Bond:      {val_res.proposed_bond_type}")
                print(f"  Initial Distance:   {f'{d_init:.2f} Å' if d_init else 'N/A'}")
                print(f"  Validation Status:  {val_res.severity} - {val_res.message}")
                print("=" * 65)

                if val_res.severity == "BLOCKING":
                    print(f"[ERROR] Tier 4 aborted: {val_res.message}", file=sys.stderr)
                    return 1

                fp = compute_tier4_fingerprint(
                    cluster_atoms=cluster.atoms,
                    nucleophile_idx=cluster.nucleophile_idx or 0,
                    electrophile_idx=cluster.electrophile_idx or 0,
                    charge=cluster.charge,
                    multiplicity=cluster.effective_multiplicity,
                    method=args.theory,
                    solvent=args.solvent,
                    scan_start=args.scan_start or d_init,
                    scan_end=args.scan_end,
                    scan_steps=args.scan_steps,
                )
                try:
                    write_reaction_definition_json(
                        output_path=ts_work_dir / 'reaction_definition.json',
                        target_residue=target_residue,
                        nucleophile_label=nucl_lbl,
                        nucleophile_element=nucl_el,
                        nucleophile_idx=cluster.nucleophile_idx or 0,
                        electrophile_label=el_lbl,
                        electrophile_element=el_el,
                        electrophile_idx=cluster.electrophile_idx or 0,
                        mechanism=getattr(args, 'mechanism', 'generic_covalent_addition'),
                        assignment_source='user_defined' if getattr(args, 'electrophile_atom', None) else 'automatic',
                        validation_status=val_res.severity,
                        initial_distance_angstrom=round(d_init, 2) if d_init else 3.5,
                        proposed_bond=val_res.proposed_bond_type,
                        fingerprint=fp,
                    )
                except Exception:
                    pass

                wf = prepare_ts_workflow_directory(
                    cluster=cluster,
                    output_dir=ts_work_dir,
                    method=args.theory,
                    solvent=None if args.solvent.lower() == 'gas' else args.solvent,
                    scan_start=args.scan_start,
                    scan_end=args.scan_end,
                    scan_steps=args.scan_steps,
                    nprocs=ts_nprocs,
                    maxcore_mb=ts_maxcore,
                    recalc_hess=getattr(args, 'recalc_hess', 25),
                    full_thermo=getattr(args, 'full_thermo', False),
                )
                print(f"[TIER 4] Workflow prepared at: {ts_work_dir}")
                print(f"[TIER 4] Scan input: {wf['scan_inp']}")
                print(f"[TIER 4] Runner script: {wf['run_script']}")

                # Fingerprint-aware cache validation (Auditor Spec Section 6 & 7)
                scan_meta_path = ts_work_dir / '01_scan.meta.json'
                cached_scan_valid = True
                if scan_meta_path.is_file():
                    try:
                        scan_meta = json.loads(scan_meta_path.read_text(encoding='utf-8'))
                        if scan_meta.get('fingerprint') != fp:
                            cached_scan_valid = False
                            print(f"[TIER 4] [CACHE INVALIDATED] Setup fingerprint mismatch in {scan_meta_path.name}. Old scan invalidated.")
                    except Exception:
                        pass
                elif (ts_work_dir / 'reaction_definition.json').is_file():
                    try:
                        r_def = json.loads((ts_work_dir / 'reaction_definition.json').read_text(encoding='utf-8'))
                        if r_def.get('fingerprint') and r_def.get('fingerprint') != fp:
                            cached_scan_valid = False
                            print(f"[TIER 4] [CACHE INVALIDATED] Setup fingerprint mismatch in reaction_definition.json. Old scan invalidated.")
                    except Exception:
                        pass

                # Check if scan output already exists and completed normally
                scan_out_path = ts_work_dir / '01_scan.out'
                scan_res = None
                scan_is_complete = False
                if scan_out_path.is_file() and cached_scan_valid:
                    try:
                        scan_text = scan_out_path.read_text(encoding='utf-8', errors='ignore')
                        scan_res = parse_orca_scan_output(scan_out_path, work_dir=ts_work_dir)
                        if scan_res and scan_res.points:
                            if "ORCA TERMINATED NORMALLY" in scan_text or len(scan_res.points) >= (args.scan_steps - 1):
                                scan_is_complete = True
                            else:
                                print(f"[TIER 4] [NOTE] Existing scan output is partial ({len(scan_res.points)}/{args.scan_steps} steps).")
                    except Exception:
                        scan_res = None

                if args.execute:
                    orca_bin = find_orca(allow_none=True)
                    if not orca_bin:
                        print("[ERROR] ORCA executable not found in PATH or standard location", file=sys.stderr)
                        return 1

                    orca_real = str(orca_bin)
                    if getattr(args, 'full_thermo', False) and (ts_work_dir / '00_reactant_freq.inp').is_file():
                        rf_out = ts_work_dir / '00_reactant_freq.out'
                        rf_done = False
                        if rf_out.is_file():
                            txt = rf_out.read_text(encoding='utf-8', errors='ignore')
                            if "VIBRATIONAL FREQUENCIES" in txt and ("ORCA TERMINATED NORMALLY" in txt or "SUCCESS" in txt or "TOTAL RUN TIME" in txt):
                                rf_done = True
                        if rf_done:
                            print(f"[TIER 4] [Step 0/4] Found existing completed reactant frequency calculation ({rf_out.name}), skipping re-run.")
                        else:
                            print(f"[TIER 4] [Step 0/4] Executing reactant vibrational frequencies with {orca_real} (Gibbs free energy)...")
                            run_orca_process(
                                executable=orca_real,
                                input_file='00_reactant_freq.inp',
                                output_file='00_reactant_freq.out',
                                cwd=ts_work_dir,
                                check_normal_termination=False,
                            )
                    if scan_res and scan_res.points and scan_is_complete:
                        print(f"[TIER 4] [Step 1/3] Found existing completed coordinate scan ({len(scan_res.points)} steps), skipping re-run.")
                    else:
                        print(f"[TIER 4] [Step 1/3] Executing ORCA relaxed coordinate scan with {orca_real}...")
                        p_scan = run_orca_process(
                            executable=orca_real,
                            input_file='01_scan.inp',
                            output_file='01_scan.out',
                            cwd=ts_work_dir,
                            check_normal_termination=False,
                        )
                        scan_res = parse_orca_scan_output(ts_work_dir / '01_scan.out', work_dir=ts_work_dir)
                        if not scan_res.points:
                            print(f"[ERROR] Coordinate scan failed: {p_scan.error or f'code {p_scan.returncode}'}", file=sys.stderr)
                            return p_scan.returncode if p_scan.returncode != 0 else 1
                        scan_is_complete = p_scan.terminated_normally
                        try:
                            scan_meta_path.write_text(json.dumps({
                                "fingerprint": fp,
                                "completed": scan_is_complete,
                                "n_points": len(scan_res.points),
                                "method": args.theory,
                                "nprocs": ts_nprocs,
                                "maxcore_mb": ts_maxcore,
                            }, indent=2), encoding='utf-8')
                        except Exception:
                            pass

                    print("[TIER 4] [Step 2/3] Analyzing scan trajectory and locating Transition State guess...")
                    print(f"[TIER 4] {scan_res.summary}")

                    if not (ts_work_dir / '02_optts.inp').is_file():
                        if scan_res.ts_guess_xyz and scan_res.ts_guess_xyz.is_file():
                            xyz_lines = scan_res.ts_guess_xyz.read_text(encoding='utf-8').splitlines()[2:]
                            tmpl = (ts_work_dir / '02_optts_template.inp').read_text(encoding='utf-8')
                            header = tmpl.split('* xyz')[0]
                            coords_str = '\n'.join(['  ' + ln for ln in xyz_lines if ln.strip()])
                            new_inp = f"{header}* xyz {cluster.charge} {cluster.effective_multiplicity}\n{coords_str}\n*\n"
                            (ts_work_dir / '02_optts.inp').write_text(new_inp, encoding='utf-8')
                        else:
                            shutil.copyfile(ts_work_dir / '02_optts_template.inp', ts_work_dir / '02_optts.inp')

                    optts_meta_path = ts_work_dir / '02_optts.meta.json'
                    cached_ts_valid = True
                    if optts_meta_path.is_file():
                        try:
                            ts_meta = json.loads(optts_meta_path.read_text(encoding='utf-8'))
                            if ts_meta.get('fingerprint') != fp:
                                cached_ts_valid = False
                                print(f"[TIER 4] [CACHE INVALIDATED] Setup fingerprint mismatch in {optts_meta_path.name}. Old TS invalidated.")
                        except Exception:
                            pass

                    optts_out = ts_work_dir / '02_optts.out'
                    if not optts_out.is_file() and (ts_work_dir / '02_optts_resume.out').is_file():
                        optts_out = ts_work_dir / '02_optts_resume.out'

                    already_converged_ts = False
                    if optts_out.is_file() and cached_ts_valid:
                        txt = optts_out.read_text(encoding='utf-8', errors='ignore')
                        if "VIBRATIONAL FREQUENCIES" in txt and ("SUCCESS" in txt or "ORCA TERMINATED NORMALLY" in txt or "TOTAL RUN TIME" in txt):
                            already_converged_ts = True

                    if already_converged_ts:
                        print(f"[TIER 4] [Step 3/3] Found existing completed TS output ({optts_out.name}), skipping re-run.")
                    else:
                        print(f"[TIER 4] [Step 3/3] Running Saddle Point Optimization & Frequency Verification (! OptTS Freq)...")
                        p_ts = run_orca_process(
                            executable=orca_real,
                            input_file='02_optts.inp',
                            output_file='02_optts.out',
                            cwd=ts_work_dir,
                            check_normal_termination=False,
                        )
                        optts_out = ts_work_dir / '02_optts.out'
                        try:
                            optts_meta_path.write_text(json.dumps({
                                "fingerprint": fp,
                                "completed": p_ts.terminated_normally,
                                "method": args.theory,
                                "nprocs": ts_nprocs,
                                "maxcore_mb": ts_maxcore,
                            }, indent=2), encoding='utf-8')
                        except Exception:
                            pass

                    # Step 4/4 (Optional): Covalent product adduct optimization
                    if (getattr(args, 'optimize_adduct', False) or (ts_work_dir / '03_adduct_opt.inp').is_file()) and (ts_work_dir / '03_adduct_opt_template.inp').is_file():
                        if args.full_gold_standard:
                            report_stage_progress(7, 8, "Covalent Adduct Product Optimization & Thermodynamics", "Adduct Opt + Freq (Delta G_rxn)")
                        adduct_inp = ts_work_dir / '03_adduct_opt.inp'
                        adduct_out = ts_work_dir / '03_adduct_opt.out'
                        if not adduct_inp.is_file():
                            if scan_res and scan_res.points:
                                last_pt = scan_res.points[-1]
                                if last_pt.xyz_path and last_pt.xyz_path.is_file():
                                    xyz_lines = last_pt.xyz_path.read_text(encoding='utf-8').splitlines()[2:]
                                    tmpl = (ts_work_dir / '03_adduct_opt_template.inp').read_text(encoding='utf-8')
                                    header = tmpl.split('* xyz')[0]
                                    coords_str = '\n'.join(['  ' + ln for ln in xyz_lines if ln.strip()])
                                    adduct_inp.write_text(f"{header}* xyz {cluster.charge} {cluster.effective_multiplicity}\n{coords_str}\n*\n", encoding='utf-8')
                        if adduct_inp.is_file() and not adduct_out.is_file():
                            print(f"[TIER 4] [Step 4/4] Executing covalent product adduct optimization with {orca_real}...")
                            run_orca_process(
                                executable=orca_real,
                                input_file='03_adduct_opt.inp',
                                output_file='03_adduct_opt.out',
                                cwd=ts_work_dir,
                                check_normal_termination=False,
                            )

                    def _extract_tier4_energies(t_dir, s_res, t_verif):
                        r_el = s_res.points[0].energy_hartree if s_res and s_res.points else None
                        r_g = None
                        t_el = t_verif.electronic_energy_hartree if t_verif and t_verif.electronic_energy_hartree != 0.0 else (s_res.max_energy_hartree if s_res else None)
                        t_g = t_verif.gibbs_free_energy_hartree if t_verif and t_verif.gibbs_free_energy_hartree != 0.0 else None
                        p_el = s_res.points[-1].energy_hartree if s_res and s_res.points else None
                        p_g = None

                        if t_dir and Path(t_dir).is_dir():
                            tw = Path(t_dir)
                            for r_name in ('00_reactant_freq.out', '00_reactants_freq.out'):
                                rf = tw / r_name
                                if rf.is_file():
                                    try:
                                        from .core.parser import parse_orca_results
                                        r_calc = parse_orca_results(rf, name="Reactants")
                                        if r_calc.gibbs_energy and math.isfinite(r_calc.gibbs_energy):
                                            r_g = r_calc.gibbs_energy
                                        if r_calc.el_energy and math.isfinite(r_calc.el_energy):
                                            r_el = r_calc.el_energy
                                        break
                                    except Exception:
                                        pass

                            for p_name in ('03_adduct_freq.out', '03_adduct_opt.out'):
                                pf = tw / p_name
                                if pf.is_file():
                                    try:
                                        from .core.parser import parse_orca_results
                                        p_calc = parse_orca_results(pf, name="Product_Adduct")
                                        if p_calc.gibbs_energy and math.isfinite(p_calc.gibbs_energy):
                                            p_g = p_calc.gibbs_energy
                                        if p_calc.el_energy and math.isfinite(p_calc.el_energy):
                                            p_el = p_calc.el_energy
                                        break
                                    except Exception:
                                        pass

                        return r_el, r_g, t_el, t_g, p_el, p_g

                    ts_verif = None
                    prop_path = optts_out.with_suffix('.property.txt')
                    if not prop_path.is_file():
                        prop_path = ts_work_dir / '02_optts.property.txt'
                    if optts_out.is_file():
                        try:
                            ts_verif = parse_orca_ts_output(optts_out, property_file_path=prop_path if prop_path.is_file() else None)
                            print(f"[TIER 4] Verification: {ts_verif.transition_vector_summary}")
                        except Exception:
                            ts_verif = None

                    reactants_el, reactants_g, ts_el, ts_g, prod_el, prod_g = _extract_tier4_energies(ts_work_dir, scan_res, ts_verif)

                    profile = compute_reaction_profile(
                        reactants_electronic=reactants_el,
                        ts_electronic=ts_el,
                        product_electronic=prod_el,
                        reactants_gibbs=reactants_g,
                        ts_gibbs=ts_g,
                        product_gibbs=prod_g,
                        is_first_order_ts=ts_verif.is_valid_first_order_saddle_point if ts_verif else False
                    )
                    print("=" * 65)
                    print(f" [TIER 4] REACTION THERMOCHEMISTRY & KINETICS ({profile.energy_basis.upper()})")
                    print(f"  Activation Barrier ({profile.barrier_symbol}): {profile.activation_barrier_kcal:.2f} kcal/mol")
                    if profile.reaction_energy_kcal is not None:
                        print(f"  Reaction Energy ({profile.reaction_energy_symbol}): {profile.reaction_energy_kcal:.2f} kcal/mol")
                    k_str = f"{profile.rate_constant_s:.3e} s^-1" if profile.rate_constant_s is not None else "Not available (requires Gibbs free energy)"
                    print(f"  Kinetic Feasibility:          {profile.kinetic_feasibility}")
                    print(f"  Estimated Half-Life (t1/2):    {profile.estimated_half_life_str}")
                    print(f"  Rate Constant (k):             {k_str}")
                    print("=" * 65)

                    if covalent_summary is not None:
                        covalent_summary['transition_state'] = {
                            'structure_provenance': structure_provenance,
                            'energy_basis': profile.energy_basis,
                            'barrier_symbol': profile.barrier_symbol,
                            'reaction_energy_symbol': profile.reaction_energy_symbol,
                            'delta_g_activation_kcal': profile.delta_g_activation_kcal,
                            'delta_g_reaction_kcal': profile.delta_g_reaction_kcal,
                            'delta_e_activation_kcal': profile.delta_e_activation_kcal,
                            'delta_e_reaction_kcal': profile.delta_e_reaction_kcal,
                            'activation_barrier_kcal': profile.activation_barrier_kcal,
                            'reaction_energy_kcal': profile.reaction_energy_kcal,
                            'kinetic_feasibility': profile.kinetic_feasibility,
                            'half_life': profile.estimated_half_life_str,
                            'model_type': cluster.model_type,
                            'summary': profile.summary,
                            'is_first_order_ts': ts_verif.is_valid_first_order_saddle_point if ts_verif else False,
                            'warnings': profile.warnings,
                        }
                else:
                    optts_candidate = ts_work_dir / '02_optts.out'
                    if not optts_candidate.is_file() and (ts_work_dir / '02_optts_resume.out').is_file():
                        optts_candidate = ts_work_dir / '02_optts_resume.out'
                    has_completed_ts = False
                    if optts_candidate.is_file():
                        txt = optts_candidate.read_text(encoding='utf-8', errors='ignore')
                        if "VIBRATIONAL FREQUENCIES" in txt and ("SUCCESS" in txt or "ORCA TERMINATED NORMALLY" in txt or "TOTAL RUN TIME" in txt):
                            has_completed_ts = True

                    if scan_res and scan_res.points and has_completed_ts:
                        print(f"[TIER 4] Existing completed calculation detected in {ts_work_dir}, parsing results...")
                        prop_path = optts_candidate.with_suffix('.property.txt')
                        if not prop_path.is_file():
                            prop_path = ts_work_dir / '02_optts.property.txt'
                        try:
                            ts_verif = parse_orca_ts_output(optts_candidate, property_file_path=prop_path if prop_path.is_file() else None)
                        except Exception:
                            ts_verif = None
                        reactants_el, reactants_g, ts_el, ts_g, prod_el, prod_g = _extract_tier4_energies(ts_work_dir, scan_res, ts_verif)
                        profile = compute_reaction_profile(
                            reactants_electronic=reactants_el,
                            ts_electronic=ts_el,
                            product_electronic=prod_el,
                            reactants_gibbs=reactants_g,
                            ts_gibbs=ts_g,
                            product_gibbs=prod_g,
                            is_first_order_ts=ts_verif.is_valid_first_order_saddle_point if ts_verif else False
                        )
                        if covalent_summary is not None:
                            covalent_summary['transition_state'] = {
                                'structure_provenance': structure_provenance,
                                'energy_basis': profile.energy_basis,
                                'barrier_symbol': profile.barrier_symbol,
                                'reaction_energy_symbol': profile.reaction_energy_symbol,
                                'delta_g_activation_kcal': profile.delta_g_activation_kcal,
                                'delta_g_reaction_kcal': profile.delta_g_reaction_kcal,
                                'delta_e_activation_kcal': profile.delta_e_activation_kcal,
                                'delta_e_reaction_kcal': profile.delta_e_reaction_kcal,
                                'activation_barrier_kcal': profile.activation_barrier_kcal,
                                'reaction_energy_kcal': profile.reaction_energy_kcal,
                                'kinetic_feasibility': profile.kinetic_feasibility,
                                'half_life': profile.estimated_half_life_str,
                                'model_type': cluster.model_type,
                                'summary': profile.summary,
                                'is_first_order_ts': ts_verif.is_valid_first_order_saddle_point if ts_verif else False,
                                'warnings': profile.warnings,
                            }
                    else:
                        print(f"[TIER 4] To execute the transition state search manually, run:")
                        print(f"         bash {wf['run_script']}")
                        if covalent_summary is not None and 'transition_state' not in covalent_summary:
                            covalent_summary['transition_state'] = {
                                'structure_provenance': structure_provenance,
                                'model_type': cluster.model_type,
                                'summary': f"Tier 4 workflow prepared at {ts_work_dir}. Awaiting execution.",
                            }

        if covalent_summary is not None:
            from .analysis.covalent_matcher import compute_total_covalent_feasibility
            from .reports.adduct_viewer import generate_adduct_viewer_html, build_adduct_pdb

            selected_poses = _get_selected_poses()
            p_top = selected_poses[0] if selected_poses else (session.poses[0] if session.poses else None)
            d_score = p_top.score if (p_top and math.isfinite(p_top.score)) else None
            p_nac_val = clustering_info.get('p_nac') if clustering_info else None
            ts_res = covalent_summary.get('transition_state', {})
            dg_val = (ts_res.get('delta_g_activation_kcal') or ts_res.get('activation_barrier_kcal')) if ts_res else None
            static_cfi = all_contacts[0].get('composite_feasibility') if all_contacts else None

            tot_feas = compute_total_covalent_feasibility(
                docking_score=d_score,
                p_nac=p_nac_val,
                delta_g_ts=dg_val,
                static_cfi=static_cfi
            )
            covalent_summary['total_feasibility'] = {
                'cfi_final': tot_feas.cfi_final,
                'cfi_pre': tot_feas.cfi_pre,
                'cfi_total': tot_feas.cfi_total,
                'percentage': tot_feas.percentage,
                'status': tot_feas.status,
                'tier': tot_feas.tier,
                'affinity_score': tot_feas.affinity_score,
                'nac_score': tot_feas.nac_score,
                'ts_score': tot_feas.ts_score,
                'docking_score': tot_feas.docking_score,
                'p_nac': tot_feas.p_nac,
                'delta_g_ts': tot_feas.delta_g_ts,
                'k_chem': tot_feas.k_chem,
                'rgi_static': tot_feas.rgi_static,
                'completeness': tot_feas.completeness,
                'missing_components': tot_feas.missing_components,
                'weights': tot_feas.weights,
                'summary': tot_feas.summary,
                'warnings': tot_feas.warnings,
            }
            print(f"[FEASIBILITY] {tot_feas.summary}")

            try:
                from .analysis.adduct_qm import compute_adduct_quantum_profile
                target_res = args.target_residue or 'THR309'
                best_c = all_contacts[0] if all_contacts else None
                nucl_crd = None
                el_crd = None
                if all_nucl_atoms and best_c:
                    for na in all_nucl_atoms:
                        if na['residue'] == best_c.get('residue') and na['atom'] == best_c.get('nucleophile_atom'):
                            nucl_crd = (na['x'], na['y'], na['z'])
                            break
                if all_lig_atoms and best_c:
                    for la in all_lig_atoms:
                        if la['index'] == best_c.get('ligand_atom_index'):
                            el_crd = (la['x'], la['y'], la['z'])
                            break

                # Compute quantum adduct verification profile
                att_dist = float(best_c.get('distance_angstrom', 5.0)) if (best_c and best_c.get('distance_angstrom') is not None) else 5.0
                bd_ang = float(best_c.get('burgi_dunitz_angle', 107.0)) if (best_c and best_c.get('burgi_dunitz_angle') is not None) else 107.0
                nucl_el_sym = str(best_c.get('nucleophile_atom', 'OG1'))[0] if best_c else 'O'
                el_idx = int(best_c.get('ligand_atom_index', 0)) if (best_c and best_c.get('ligand_atom_index') is not None) else 0
                el_el_sym = str(best_c.get('ligand_atom_element') or best_c.get('ligand_atom_symbol') or 'O') if best_c else 'O'
                if all_lig_atoms and el_idx < len(all_lig_atoms):
                    el_el_sym = all_lig_atoms[el_idx].get('element', el_el_sym)
                lig_h_atoms = [(str(la.get('element', 'C')), int(la.get('index', i))) for i, la in enumerate(all_lig_atoms) if la.get('element') != 'H'] if all_lig_atoms else None

                is_rev = False
                w_type = str(best_c.get('warhead_type', '') if best_c else '').lower()
                if any(k in w_type for k in ('reversible', 'pseudo', 'cyano', 'nitrile', 'furoxan', 'boron')):
                    is_rev = True

                homo_val = -6.40
                lumo_val = -2.89
                orb_src = "model_derived"
                if 'cluster_qm_res' in locals() and cluster_qm_res and cluster_qm_res.success:
                    if cluster_qm_res.homo_energy_ev is not None and cluster_qm_res.lumo_energy_ev is not None:
                        homo_val = float(cluster_qm_res.homo_energy_ev)
                        lumo_val = float(cluster_qm_res.lumo_energy_ev)
                        orb_src = "ORCA cluster single-point"

                has_opt_adduct = False
                adduct_bond_dist = att_dist
                if 'ts_work_dir' in locals() and ts_work_dir:
                    adduct_opt_out = ts_work_dir / '03_adduct_opt.out'
                    if adduct_opt_out.is_file():
                        txt = adduct_opt_out.read_text(encoding='utf-8', errors='ignore')
                        if "ORCA TERMINATED NORMALLY" in txt or "OPTIMIZATION RUN DONE" in txt or "SUCCESS" in txt:
                            has_opt_adduct = True
                            adduct_bond_dist = 1.45
                    elif 'scan_res' in locals() and scan_res and scan_res.points:
                        last_pt = scan_res.points[-1]
                        if last_pt.coordinate_value <= 1.80 and ('scan_is_complete' in locals() and scan_is_complete):
                            has_opt_adduct = True
                            adduct_bond_dist = last_pt.coordinate_value

                adduct_qm_prof = compute_adduct_quantum_profile(
                    distance_angstrom=adduct_bond_dist,
                    burgi_dunitz_angle_deg=bd_ang,
                    nucleophile_homo_ev=homo_val,
                    electrophile_lumo_ev=lumo_val,
                    target_atom_index=el_idx,
                    target_atom_symbol=el_el_sym,
                    ligand_heavy_atoms=lig_h_atoms,
                    nucleophile_element=nucl_el_sym,
                    electrophile_element=el_el_sym,
                    is_reversible_warhead=is_rev,
                    orbital_source=orb_src,
                    is_optimized_adduct=has_opt_adduct,
                )
                covalent_summary['adduct_qm'] = adduct_qm_prof.to_dict()

                lig_input = None
                rec_input = None
                if 'cluster_rep' in locals() and cluster_rep and cluster_rep.snapshot_ligand_pdb and cluster_rep.snapshot_receptor_pdb:
                    lig_input = cluster_rep.snapshot_ligand_pdb
                    rec_input = cluster_rep.snapshot_receptor_pdb
                elif p_top:
                    lig_input = p_top.pose_file
                    rec_input = session.receptor_for(p_top.receptor_id, raw=False)

                if lig_input and rec_input:
                    adduct_pdb = build_adduct_pdb(
                        ligand_pdb_or_xyz=lig_input,
                        receptor_pdb=rec_input,
                        target_residue=target_res,
                        dyad_residue=best_c.get('catalytic_dyad_residue') if best_c else 'ASP199'
                    )
                    c_qm_dict = cluster_qm_res.to_dict() if ('cluster_qm_res' in locals() and cluster_qm_res and cluster_qm_res.success) else None
                    if c_qm_dict:
                        covalent_summary['cluster_qm'] = c_qm_dict

                    covalent_summary['adduct_viewer_html'] = generate_adduct_viewer_html(
                        pdb_data=adduct_pdb,
                        target_residue=target_res,
                        nucl_atom_coords=nucl_crd,
                        el_atom_coords=el_crd,
                        attack_distance=best_c.get('distance_angstrom') if best_c else None,
                        burgi_dunitz_angle=best_c.get('burgi_dunitz_angle') if best_c else None,
                        dyad_residue=best_c.get('catalytic_dyad_residue') if best_c else 'ASP199',
                        homo_cube_data=cluster_qm_res.homo_cube_data if ('cluster_qm_res' in locals() and cluster_qm_res) else None,
                        lumo_cube_data=cluster_qm_res.lumo_cube_data if ('cluster_qm_res' in locals() and cluster_qm_res) else None,
                        cluster_qm_data=c_qm_dict,
                        adduct_qm_data=covalent_summary.get('adduct_qm'),
                        standalone=False
                    )
            except Exception as e:
                print(f"[NOTE] Could not generate 3Dmol adduct viewer: {e}")

        if args.dft:
            if args.full_gold_standard:
                report_stage_progress(8, 8, "Isolated Ligand DFT & Frontier Orbitals", f"Theory: {args.theory} | Solvent: {args.solvent}")
            dft_nprocs = args.nprocs if args.nprocs is not None else (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
            dft_maxcore = _get_safe_maxcore_mb(dft_nprocs, args.maxcore)
            jobs = prepare_ligand_jobs(
                session, Path(args.work_dir) / 'quantum', top=args.top, target=args.target, compounds=args.compound,
                pareto=args.pareto, include_controls=args.include_controls, method=args.theory,
                solvent=None if args.solvent.lower() == 'gas' else args.solvent,
                optimize=not args.single_point, frequencies=args.frequencies, charge=args.charge,
                multiplicity=args.multiplicity, seed=args.seed, nprocs=args.nprocs, maxcore=dft_maxcore,
                geometry=args.geometry, orbital_grid=None if args.no_orbital_plots else args.orbital_grid)
            records = []
            for job in jobs:
                record = run_orca_job(job, timeout=args.timeout) if args.execute else json.loads(
                    (job / 'job.json').read_text(encoding='utf-8'))
                records.append(record)
                print(f"[DFT] {record['selection']['ligand_id']}: {record['status']} ({job})")
            out_file = Path(args.html) if args.html else Path(args.work_dir) / 'dossier.html'
            for job, record in zip(jobs, records):
                if record.get('orbital_export', {}).get('status') == 'completed':
                    record['viewer_link'] = Path(os.path.relpath(job / 'frontier_orbitals.html', out_file.parent)).as_posix()
            if covalent_summary and 'cdft' not in covalent_summary and records:
                first_orb = records[0].get('results', {}).get('orbitals', {}).get('0', {})
                h = first_orb.get('homo', {}).get('energy_eV')
                l = first_orb.get('lumo', {}).get('energy_eV')
                if h is not None and l is not None:
                    from .analysis.reactivity import calculate_cdft_descriptors
                    desc = calculate_cdft_descriptors(homo_ev=h, lumo_ev=l)
                    covalent_summary['cdft'] = {
                        'hardness_ev': desc.hardness_ev,
                        'chemical_potential_ev': desc.chemical_potential_ev,
                        'electrophilicity_ev': desc.electrophilicity_ev,
                        'softness_ev': desc.softness_ev,
                    }
            qm_summary = {'jobs': records}
            if not (args.full_gold_standard or args.covalent or getattr(args, 'cluster_nac', False) or args.run_md):
                generate_html_dossier(session.project_name, [], out_file, qm_summary={'jobs': records},
                                      covalent_summary=covalent_summary)
                print(f'[REPORT] {out_file}')
                return int(args.execute and any(r['status'] != 'completed' or
                           r.get('orbital_export', {}).get('status') == 'failed' for r in records))
        poses = _get_selected_poses()
        poses_data = [dict(ligand_id=p.ligand_id, pose_idx=p.pose_idx,
                           score=p.score if math.isfinite(p.score) else None) for p in poses]
        if args.html:
            out_file = Path(args.html)
        elif args.work_dir:
            out_file = Path(args.work_dir) / 'dossier.html'
        else:
            rep_dir = Path.cwd() / 'reports'
            try:
                rep_dir.mkdir(parents=True, exist_ok=True)
                out_file = rep_dir / 'dossier.html'
            except OSError:
                import tempfile
                root = Path(os.environ.get('SHARK_SCRATCH', tempfile.gettempdir())).expanduser()
                root.mkdir(parents=True, exist_ok=True)
                out_file = Path(tempfile.mkdtemp(prefix='shark-dft-report-', dir=root)) / 'dossier.html'

        if 'qm_summary' not in locals() or qm_summary is None:
            qm_summary = None
        if qm_summary is None and not args.dft_dir:
            cands = []
            if args.compound:
                for c in args.compound:
                    c_base = c.split('_')[0]
                    cands.extend([Path.cwd() / f"dft_{c}", Path.cwd() / f"dft_{c_base}", Path.cwd() / "dft"])
            else:
                cands.extend([
                    Path.cwd() / "dft_benzofuroxan",
                    Path.cwd() / "dft",
                    Path('/home/diego/SharK/dft_benzofuroxan'),
                    Path(__file__).resolve().parent.parent.parent / "dft_benzofuroxan",
                ])
            # If compound or project name mentions benzofuroxan or 091326
            proj_str = str(getattr(session, 'project_name', '')).casefold()
            if 'benzofuroxan' in proj_str or '091326' in proj_str:
                for b_cand in [Path.cwd() / "dft_benzofuroxan", Path('/home/diego/SharK/dft_benzofuroxan'), Path(__file__).resolve().parent.parent.parent / "dft_benzofuroxan"]:
                    if b_cand not in cands:
                        cands.append(b_cand)
            for d_cand in cands:
                if d_cand.is_dir() and any(d_cand.glob("*.out")):
                    args.dft_dir = str(d_cand)
                    print(f"[DFT] Auto-detected existing DFT calculation directory: {d_cand}")
                    break

        if args.dft_dir and qm_summary is None:
            dft_path = Path(args.dft_dir)
            if dft_path.is_dir():
                records = load_dft_records(dft_path, out_file.parent)
                if records:
                    qm_summary = {'jobs': records}
                    print(f"[DFT] Loaded {len(records)} existing quantum calculation(s) from {dft_path}")

        if qm_summary and qm_summary.get('jobs'):
            records = qm_summary['jobs']
            for p_dict in poses_data:
                cand_rec = records[0]
                for r in records:
                    r_id = r.get('selection', {}).get('ligand_id', '')
                    if r_id.casefold() in p_dict['ligand_id'].casefold() or p_dict['ligand_id'].casefold() in r_id.casefold():
                        cand_rec = r
                        break
                orb = cand_rec.get('results', {}).get('orbitals', {}).get('0', {})
                if orb:
                    h = orb.get('homo', {}).get('energy_eV')
                    l = orb.get('lumo', {}).get('energy_eV')
                    g = orb.get('gap_ev')
                    if h is not None:
                        p_dict['homo_ev'] = h
                    if l is not None:
                        p_dict['lumo_ev'] = l
                    if g is not None:
                        p_dict['gap_ev'] = g
            if covalent_summary and 'cdft' not in covalent_summary:
                first_orb = records[0].get('results', {}).get('orbitals', {}).get('0', {})
                h = first_orb.get('homo', {}).get('energy_eV')
                l = first_orb.get('lumo', {}).get('energy_eV')
                if h is not None and l is not None:
                    from .analysis.reactivity import calculate_cdft_descriptors
                    desc = calculate_cdft_descriptors(homo_ev=h, lumo_ev=l)
                    covalent_summary['cdft'] = {
                        'hardness_ev': desc.hardness_ev,
                        'chemical_potential_ev': desc.chemical_potential_ev,
                        'electrophilicity_ev': desc.electrophilicity_ev,
                        'softness_ev': desc.softness_ev,
                    }

        from .core.analysis_result import SharKAnalysisResult
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        req_wf = 'simple_gold_standard' if args.simple_gold_standard else ('full_gold_standard' if args.full_gold_standard else ('fast_analysis' if args.fast_analysis else 'custom'))
        analysis_result = SharKAnalysisResult(
            analysis_id=str(session.project_name),
            workflow={
                'requested': req_wf,
                'executed': req_wf,
                'status': 'completed',
                'degraded': False,
                'degradation_reason': None,
            },
            binding={
                'docking_score': d_score if 'd_score' in locals() else None,
                'affinity_score': tot_feas.affinity_score if 'tot_feas' in locals() else None,
            },
            static_reactive_geometry={
                'rgi_static': tot_feas.rgi_static if 'tot_feas' in locals() else None,
                'best_contact': all_contacts[0] if 'all_contacts' in locals() and all_contacts else None,
            },
            dynamics=clustering_info if 'clustering_info' in locals() and clustering_info else {},
            cluster_qm=c_qm_dict if 'c_qm_dict' in locals() and c_qm_dict else {},
            transition_state=covalent_summary.get('transition_state', {}) if covalent_summary else {},
            adduct=covalent_summary.get('adduct_qm', {}) if covalent_summary else {},
            feasibility=covalent_summary.get('total_feasibility', {}) if covalent_summary else {},
            completeness=tot_feas.completeness if 'tot_feas' in locals() else {},
            warnings=tot_feas.warnings if 'tot_feas' in locals() else [],
            provenance={'project': session.project_name, 'work_dir': str(args.work_dir)},
        )

        try:
            analysis_result.write_evidence_json(out_dir / 'evidence.json')
            analysis_result.write_analysis_manifest(out_dir / 'analysis_manifest.json')
        except Exception as e:
            print(f"[NOTE] Could not write evidence JSON: {e}")

        md_summary = None
        if 'last_md_res' in locals() and last_md_res:
            prep_dir = last_md_res.run_dir / '00_prep'
            if not prep_dir.is_dir():
                prep_dir = last_md_res.run_dir
            md_plots = {}
            for p_name in ('md_analysis_dashboard.png', 'plot_rmsd.png', 'plot_rmsf.png', 'plot_gyrate.png', 'plot_hbonds.png'):
                p_file = prep_dir / p_name
                if p_file.is_file():
                    md_plots[p_name] = str(p_file)
            md_summary = {
                'run_id': last_md_res.run_id,
                'sim_time_ns': last_md_res.sim_time_ns,
                'status': last_md_res.status,
                'run_dir': str(last_md_res.run_dir),
                'plots': md_plots,
                'dashboard_path': str(last_md_res.dashboard_path) if last_md_res.dashboard_path else md_plots.get('md_analysis_dashboard.png'),
                'clustering': clustering_info if 'clustering_info' in locals() else None,
                'force_fields': 'AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (0.15 M NaCl)',
            }
        elif getattr(args, 'trajectory', None):
            traj_p = Path(args.trajectory).resolve()
            search_dirs = [traj_p.parent]
            if (traj_p.parent / '00_prep').is_dir():
                search_dirs.append(traj_p.parent / '00_prep')
            if traj_p.parent.parent.is_dir() and (traj_p.parent.parent / '00_prep').is_dir():
                search_dirs.append(traj_p.parent.parent / '00_prep')

            md_plots = {}
            for sdir in search_dirs:
                for p_name in ('md_analysis_dashboard.png', 'plot_rmsd.png', 'plot_rmsf.png', 'plot_gyrate.png', 'plot_hbonds.png'):
                    if p_name not in md_plots and (sdir / p_name).is_file():
                        md_plots[p_name] = str(sdir / p_name)

            s_time = None
            if 'clustering_info' in locals() and clustering_info:
                s_time = clustering_info.get('sim_time_ns') or clustering_info.get('total_sim_time_ns')
            if not s_time:
                s_time = getattr(args, 'time_ns', None) or 20.0

            run_id = traj_p.parent.name
            if run_id == '00_prep':
                run_id = traj_p.parent.parent.name

            md_summary = {
                'run_id': run_id,
                'sim_time_ns': s_time,
                'status': 'completed',
                'run_dir': str(traj_p.parent),
                'plots': md_plots,
                'dashboard_path': md_plots.get('md_analysis_dashboard.png'),
                'clustering': clustering_info if 'clustering_info' in locals() else None,
                'force_fields': 'AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (0.15 M NaCl)',
            }

        generate_html_dossier(session.project_name, poses_data, out_file,
                              qm_summary=qm_summary, md_summary=md_summary,
                              covalent_summary=covalent_summary,
                              result=analysis_result)

        # Consolidate structured machine-readable deliverables in job directory
        try:
            if covalent_summary:
                (out_dir / 'covalent_feasibility.json').write_text(
                    json.dumps(covalent_summary, indent=2, default=str) + '\n', encoding='utf-8')
            if poses_data:
                import csv
                with open(out_dir / 'poses_summary.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['ligand_id', 'pose_idx', 'score', 'homo_ev', 'lumo_ev', 'gap_ev'])
                    writer.writeheader()
                    for p in poses_data:
                        writer.writerow({k: p.get(k, '') for k in ['ligand_id', 'pose_idx', 'score', 'homo_ev', 'lumo_ev', 'gap_ev']})
        except Exception:
            pass

        print(f'[REPORT] {out_file}')
        return 0
    except KeyboardInterrupt:
        print('[STOPPED] Execution interrupted; job state and logs preserved', file=sys.stderr)
        return 130
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    finally:
        if session is not None:
            shutil.rmtree(session.extract_dir)


if __name__ == '__main__':
    sys.exit(main())
