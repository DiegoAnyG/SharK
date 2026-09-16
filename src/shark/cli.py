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

from .core.session import read_poliscreen_session
from .core.runner import run_orca_job
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
    print('Select a PoliScreen session:')
    for index, path in enumerate(recent, 1):
        print(f'  [{index}] {path.name}')
    choice = input('Session number, file path, or Q to quit > ').strip()
    if choice.upper() == 'Q':
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(recent):
        return recent[int(choice) - 1]
    if choice.upper() == 'M':
        choice = input('Session file path > ').strip()
    return Path(choice).expanduser()


def print_banner():
    print('SharK: PoliScreen sessions and isolated-ligand ORCA DFT')


def run_interactive():
    print_banner()
    session = interactive_session_picker()
    if session is None:
        return 0
    action = input('[1] Prepare ligand DFT  [2] Execute ligand DFT  [3] Session report  [4] Analyze existing MD contacts  [Q] Quit > ').strip().upper()
    if action == 'Q':
        return 0
    if action not in ('1', '2', '3', '4'):
        print('Unknown action', file=sys.stderr)
        return 2
    argv = ['--session', str(session)]
    if action == '4':
        argv = ['--analyze-md', '--topology', input('Matching topology (GRO or supported TPR) > ').strip(),
                '--trajectory', input('XTC trajectory > ').strip(),
                '--ligand-selection', input('Exact ligand atom selection (for example: resname UNL and not name H*) > ').strip(),
                '--contact-cutoff', input('Contact cutoff in angstrom [4.0] > ').strip() or '4.0']
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
    output = input('Output directory [temporary scratch] > ').strip()
    if output:
        argv += ['--work-dir', output]
    return main(argv)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'run-qm':
        arguments = ['--dft'] + arguments[1:]
    if arguments and arguments[0] == 'analyze-md':
        arguments = ['--analyze-md'] + arguments[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze-md', action='store_true', help='Analyze contacts in an existing trajectory; does not launch MD')
    parser.add_argument('--topology', help='Matching GRO or supported topology')
    parser.add_argument('--trajectory', help='Existing XTC trajectory')
    parser.add_argument('--ligand-selection', help='Explicit MDAnalysis atom selection for one ligand residue')
    parser.add_argument('--protein-selection', default='protein and not name H*')
    parser.add_argument('--contact-cutoff', type=float, default=4.0, help='Geometric contact threshold in angstrom')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--start-ns', type=float, default=0.0)
    parser.add_argument('--stop-ns', type=float)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--session', help='PoliScreen session archive')
    parser.add_argument('--top', type=int, default=1, help='Unique ligands for DFT; poses for reports')
    parser.add_argument('--target', help='Exact target or target~pocket identifier')
    parser.add_argument('--compound', action='append', help='Exact compound name; may be repeated')
    parser.add_argument('--pareto', action='store_true', help='Select recorded Pareto leaders')
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--dft', action='store_true', help='Prepare isolated-ligand ORCA jobs')
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
    parser.add_argument('--html', help='Output HTML dossier path')
    parser.add_argument('--run-md', '--md', dest='run_md', action='store_true', help='Prepare or launch GROMACS MD simulation using raw receptor')
    parser.add_argument('--pipeline-dir', help='Custom path to GROMACS pipeline root (overrides SHARK_GROMACS_PIPELINE)')
    parser.add_argument('--time-ns', type=float, default=10.0, help='Simulation time in nanoseconds for MD')
    parser.add_argument('--covalent', action='store_true', help='Perform Conceptual DFT reactivity profiling and pocket Near-Attack Conformation matching')
    parser.add_argument('--target-residue', help='Target nucleophile residue to scan (e.g. CYS145, CYS, 145)')
    args = parser.parse_args(arguments)
    if args.interactive or not arguments:
        try:
            return run_interactive()
        except (EOFError, KeyboardInterrupt):
            return 130
    if args.analyze_md:
        if args.dft or args.execute or args.md:
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
    if args.execute and not args.dft and not args.run_md:
        parser.error('--execute requires --dft, run-qm or --run-md')
    if args.top < 1:
        parser.error('--top must be positive')
    session = None
    try:
        session = read_poliscreen_session(args.session)
        print(f"[SharK] Session '{session.project_name}': {len(session.poses)} indexed poses")
        for warning in session.warnings:
            print(f'[NOTE] {warning}')

        def _get_selected_poses():
            if args.compound:
                selected = []
                for c in args.compound:
                    matched = [p for p in session.poses if p.ligand_id.casefold() == c.casefold()
                               and (args.target is None or p.receptor_id == args.target)]
                    if matched:
                        matched.sort(key=lambda p: (p.score if math.isfinite(p.score) else math.inf, p.pose_idx))
                        selected.extend(matched[:args.top])
                if selected:
                    return selected
            return session.list_top_poses(args.top)

        if args.run_md:
            from .workflows.md_pipeline import run_md_from_session
            selected_poses = _get_selected_poses()

            for pose in selected_poses:
                print(f"[MD] Preparing simulation for {pose.ligand_id} (pose #{pose.pose_idx}) against {pose.receptor_id}...")
                res = run_md_from_session(
                    session=session,
                    ligand_id=pose.ligand_id,
                    pose_idx=pose.pose_idx,
                    sim_time_ns=args.time_ns,
                    run_now=args.execute,
                    pipeline_dir=args.pipeline_dir,
                    work_dir=args.work_dir
                )
                print(f"[MD] Status: {res.status} | Directory: {res.run_dir}")
                if res.dashboard_path:
                    print(f"[MD] Dashboard: {res.dashboard_path}")
            return 0

        covalent_summary = None
        if args.covalent:
            from .analysis.covalent_matcher import match_covalent_pocket
            from .analysis.reactivity import build_reactivity_profile

            selected_poses = _get_selected_poses()

            all_contacts, nac_contacts, pocket_nucls, summaries = [], [], [], []
            for p in selected_poses:
                rec_path = session.receptor_for(p.receptor_id, raw=False)
                rep = match_covalent_pocket(
                    receptor_pdb=rec_path,
                    ligand_pose=p.pose_file,
                    pocket_cutoff=args.contact_cutoff,
                    nac_cutoff=3.5,
                    target_residue=args.target_residue,
                    receptor_name=p.receptor_id,
                    ligand_name=p.ligand_id
                )
                print(f"[COVALENT] {rep.summary}")
                summaries.append(rep.summary)
                for n in rep.pocket_nucleophiles:
                    if n.residue_label not in pocket_nucls:
                        pocket_nucls.append(n.residue_label)
                for c in rep.contacts:
                    all_contacts.append({
                        'residue': c.nucleophile.residue_label,
                        'nucleophile_atom': c.nucleophile.atom_name,
                        'ligand_atom_index': c.ligand_atom_index,
                        'ligand_atom_element': c.ligand_atom_element,
                        'distance_angstrom': c.distance_angstrom,
                        'feasibility_score': c.feasibility_score,
                        'is_nac': c.is_nac,
                        'warhead_rank': c.warhead_rank
                    })
                    if c.is_nac:
                        nac_contacts.append(c)

            covalent_summary = {
                'has_nac': len(nac_contacts) > 0,
                'summary': " ".join(summaries),
                'contacts': all_contacts,
                'nac_contacts': [
                    {
                        'residue': c.nucleophile.residue_label,
                        'nucleophile_atom': c.nucleophile.atom_name,
                        'ligand_atom_index': c.ligand_atom_index,
                        'ligand_atom_element': c.ligand_atom_element,
                        'distance_angstrom': c.distance_angstrom,
                        'feasibility_score': c.feasibility_score,
                        'is_nac': c.is_nac,
                        'warhead_rank': c.warhead_rank
                    }
                    for c in nac_contacts
                ],
                'pocket_nucleophiles': pocket_nucls
            }

        if args.dft:
            jobs = prepare_ligand_jobs(
                session, args.work_dir, top=args.top, target=args.target, compounds=args.compound,
                pareto=args.pareto, include_controls=args.include_controls, method=args.theory,
                solvent=None if args.solvent.lower() == 'gas' else args.solvent,
                optimize=not args.single_point, frequencies=args.frequencies, charge=args.charge,
                multiplicity=args.multiplicity, seed=args.seed, nprocs=args.nprocs, maxcore=args.maxcore,
                geometry=args.geometry, orbital_grid=None if args.no_orbital_plots else args.orbital_grid)
            records = []
            for job in jobs:
                record = run_orca_job(job, timeout=args.timeout) if args.execute else json.loads(
                    (job / 'job.json').read_text(encoding='utf-8'))
                records.append(record)
                print(f"[DFT] {record['selection']['ligand_id']}: {record['status']} ({job})")
            out_file = Path(args.html) if args.html else jobs[0].parent / 'dossier.html'
            for job, record in zip(jobs, records):
                if record.get('orbital_export', {}).get('status') == 'completed':
                    record['viewer_link'] = Path(os.path.relpath(job / 'frontier_orbitals.html', out_file.parent)).as_posix()
            generate_html_dossier(session.project_name, [], out_file, qm_summary={'jobs': records},
                                  covalent_summary=covalent_summary)
            print(f'[REPORT] {out_file}')
            return int(args.execute and any(r['status'] != 'completed' or
                       r.get('orbital_export', {}).get('status') == 'failed' for r in records))
        poses = session.list_top_poses(args.top)
        poses_data = [dict(ligand_id=p.ligand_id, pose_idx=p.pose_idx,
                           score=p.score if math.isfinite(p.score) else None) for p in poses]
        if args.html:
            out_file = Path(args.html)
        elif args.work_dir:
            out_file = Path(args.work_dir) / 'dossier.html'
        else:
            import tempfile
            root = Path(os.environ.get('SHARK_SCRATCH', tempfile.gettempdir())).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            out_file = Path(tempfile.mkdtemp(prefix='shark-dft-report-', dir=root)) / 'dossier.html'
        generate_html_dossier(session.project_name, poses_data, out_file, covalent_summary=covalent_summary)
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
