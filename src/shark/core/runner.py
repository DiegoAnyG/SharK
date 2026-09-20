"""Execute recorded ORCA jobs with isolated processes and explicit failure states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from .frontier import file_hash, read_orbital_metadata, discover_frontier_fields


def find_orca(allow_none: bool = False) -> Optional[Path]:
    configured = os.environ.get('SHARK_ORCA')
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            candidate /= 'orca.exe' if os.name == 'nt' else 'orca'
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        if not allow_none:
            raise FileNotFoundError('SHARK_ORCA does not resolve to an executable')

    found = shutil.which('orca')
    if found:
        candidate = Path(found)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    # Search local user bioinformatics directories without hardcoding personal usernames
    bio_base = Path.home() / "bioinformatics"
    if bio_base.is_dir():
        for cand in bio_base.glob("orca*"):
            exe = cand / ('orca.exe' if os.name == 'nt' else 'orca')
            if exe.is_file() and os.access(exe, os.X_OK):
                return exe.resolve()

    local_bin = Path.home() / ".local" / "bin" / ('orca.exe' if os.name == 'nt' else 'orca')
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return local_bin.resolve()

    if allow_none:
        return None
    raise FileNotFoundError('ORCA not found; set SHARK_ORCA to its executable or installation directory')


def read_qm_results(output: Path, *, optimize: bool, frequencies: bool) -> dict:
    text = output.read_text(encoding='utf-8', errors='replace')
    if 'ORCA TERMINATED NORMALLY' not in text:
        raise ValueError('ORCA did not terminate normally')
    if optimize and 'THE OPTIMIZATION HAS CONVERGED' not in text:
        raise ValueError('Geometry optimization did not converge')
    values = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-+\d.Ee]+)', text)
    if not values or not math.isfinite(float(values[-1])):
        raise ValueError('No finite final electronic energy in ORCA output')
    metadata = read_orbital_metadata(output)
    channels = {}
    for operator, rows in metadata['channels'].items():
        occupied = [r for r in rows if r['occupation'] > 0]
        virtual = [r for r in rows if r['occupation'] == 0]
        if not occupied or not virtual:
            raise ValueError('Incomplete frontier orbital table')
        homo = max(occupied, key=lambda r: r['energy_eV'])
        lumo = min(virtual, key=lambda r: r['energy_eV'])
        channels[str(operator)] = dict(homo=homo, lumo=lumo, gap_ev=lumo['energy_eV'] - homo['energy_eV'])
    result = dict(electronic_energy_hartree=float(values[-1]), orbitals=channels,
                  orca_version=metadata['orca_version'], optimization_converged=optimize,
                  stationary_minimum_verified=False)
    if frequencies:
        if 'VIBRATIONAL FREQUENCIES' not in text:
            raise ValueError('Requested frequencies are absent')
        block = text.rsplit('VIBRATIONAL FREQUENCIES', 1)[1].split('NORMAL MODES', 1)[0]
        modes = [float(v) for v in re.findall(r'^\s*\d+:\s*([-+\d.]+)\s+cm', block, re.M)]
        if not modes:
            raise ValueError('Requested frequencies could not be parsed')
        result['frequencies_cm1'] = modes
        result['imaginary_frequencies_cm1'] = [f for f in modes if f < -1e-3]
        result['stationary_minimum_verified'] = optimize and not result['imaginary_frequencies_cm1']
    return result


def _stop_process_tree(proc):
    """Terminates an ORCA process and its entire process tree (including OpenMPI workers)."""
    if proc is None:
        return
    pid = proc.pid
    if os.name == 'posix':
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            pgid = pid

        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            subprocess.run(['pkill', '-9', '-P', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
    else:
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


def cleanup_orphan_orca_processes() -> int:
    """Scans and safely terminates orphaned ORCA/MPI binaries owned by current user."""
    targets = ['orca', 'orca_prop_mpi', 'orca_scf_mpi', 'orca_gtoint_mpi', 'orca_md_mpi', 'orca_cpscf_mpi', 'mpirun']
    killed = 0
    if os.name == 'posix':
        import getpass
        try:
            current_user = getpass.getuser()
            for t in targets:
                res = subprocess.run(['pkill', '-u', current_user, '-9', '-f', t],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    killed += 1
        except Exception:
            pass
    return killed


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 59:
        h = m // 60
        m = m % 60
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _extract_orca_progress_status(out_path: Path, elapsed_s: float, input_file: str | Path = "") -> str:
    try:
        if not out_path.is_file():
            return f"[ORCA] Starting {Path(input_file).name}... ({_format_time(elapsed_s)})"
        size = out_path.stat().st_size
        if size == 0:
            return f"[ORCA] Starting {Path(input_file).name}... ({_format_time(elapsed_s)})"

        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            if size > 32768:
                f.seek(size - 32768)
            tail = f.read()

        elapsed_str = _format_time(elapsed_s)

        # 1. Relaxed coordinate scan
        scan_matches = list(re.finditer(r"RELAXED SURFACE SCAN STEP\s+(\d+).*?Bond\s*\([^)]*\)\s*:\s*([-\d.]+)", tail, re.DOTALL))
        if scan_matches:
            last_m = scan_matches[-1]
            step_num = int(last_m.group(1))
            coord_val = float(last_m.group(2))
            m_steps = re.search(r"steps\s*=\s*(\d+)", tail)
            total_steps = int(m_steps.group(1)) if m_steps else 18
            pct = min(99, int((step_num / total_steps) * 100))
            if step_num > 1:
                eta_s = (elapsed_s / (step_num - 1)) * (total_steps - step_num)
                eta_str = f" | ETA: {_format_time(eta_s)}"
            else:
                eta_str = ""
            bar_len = 16
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            return f"[ORCA SCAN] [{bar}] {pct:2d}% (Step {step_num}/{total_steps}, d={coord_val:.2f} Å) | Elapsed: {elapsed_str}{eta_str}"

        # 2. Geometry Optimization / OptTS
        opt_matches = list(re.finditer(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", tail))
        if opt_matches:
            cycle = int(opt_matches[-1].group(1))
            e_match = list(re.finditer(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", tail))
            e_str = f" | E: {float(e_match[-1].group(1)):.6f} Eh" if e_match else ""
            return f"[ORCA OPT] Cycle {cycle}{e_str} | Elapsed: {elapsed_str}"

        # 3. Vibrational Frequencies / Hessian
        if "VIBRATIONAL FREQUENCIES" in tail or "CALCULATING HESSIAN" in tail or "NUMERICAL HESSIAN" in tail:
            return f"[ORCA FREQ] Calculating Hessian & Frequencies | Elapsed: {elapsed_str}"

        # 4. SCF Iterations
        scf_m = list(re.finditer(r"ITERATION\s+(\d+):", tail))
        if scf_m:
            return f"[ORCA SCF] Iteration {scf_m[-1].group(1)} | Elapsed: {elapsed_str}"

        return f"[ORCA] Running {Path(input_file).name} | Elapsed: {elapsed_str}"
    except Exception:
        return f"[ORCA] Running | Elapsed: {_format_time(elapsed_s)}"


@dataclass
class ProcessResult:
    returncode: int
    terminated_normally: bool
    timed_out: bool = False
    interrupted: bool = False
    error: Optional[str] = None
    execution_time_s: float = 0.0


def run_orca_process(
    executable: str | Path | None = None,
    input_file: str | Path = "",
    output_file: str | Path = "",
    cwd: str | Path = ".",
    *,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
    extra_args: Optional[list[str]] = None,
    stdin_content: Optional[str] = None,
    check_normal_termination: bool = True,
    output_mode: str = "w",
    show_progress: bool = True,
) -> ProcessResult:
    """Execute an ORCA or helper command in an isolated process group with resource and timeout safety.

    Parameters
    ----------
    executable : str, Path, or None
        Path to the orca or orca_plot executable. If None, resolved via find_orca().
    input_file : str or Path
        Input file (e.g. 'calculation.inp' or 'cluster.inp').
    output_file : str or Path
        Path to output file for stdout/stderr redirection.
    cwd : str or Path
        Working directory for the calculation.
    timeout : float or None
        Maximum execution time in seconds.
    env : dict or None
        Environment variables. If None, copies os.environ and prepends executable directory to PATH.
    extra_args : list[str] or None
        Additional command-line arguments to append after input_file.
    stdin_content : str or None
        Optional input string passed via stdin (e.g. for orca_plot interactive menus).
    check_normal_termination : bool
        If True, inspects output file for 'ORCA TERMINATED NORMALLY'.
    output_mode : str
        File open mode for output file ('w' or 'x').

    Returns
    -------
    ProcessResult
        Execution status, return code, normal termination flag, and elapsed time.
    """
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Timeout must be a finite positive number")

    work_dir = Path(cwd).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if executable is None:
        exe = find_orca()
    else:
        exe = Path(executable).expanduser().resolve()
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise FileNotFoundError(f"Executable not found or not executable: {exe}")

    run_env = os.environ.copy() if env is None else env.copy()
    run_env["PATH"] = str(exe.parent) + os.pathsep + run_env.get("PATH", "")

    out_p = (work_dir / output_file).resolve() if not Path(output_file).is_absolute() else Path(output_file).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(exe)]
    if input_file:
        inp_p = Path(input_file)
        cmd.append(str(inp_p.name) if inp_p.parent == work_dir else str(inp_p))
    if extra_args:
        cmd.extend(extra_args)

    options = dict(start_new_session=True) if os.name == "posix" else dict(
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    proc = None
    old_term = None
    start_t = time.monotonic()
    timed_out = False
    interrupted = False
    error = None

    try:
        if os.name == "posix" and threading.current_thread() is threading.main_thread():
            def sigterm_handler(signum, frame):
                raise KeyboardInterrupt
            old_term = signal.signal(signal.SIGTERM, sigterm_handler)

        with open(out_p, output_mode, encoding="utf-8") as out_f:
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdin=subprocess.PIPE if stdin_content is not None else None,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                env=run_env,
                text=True if stdin_content is not None else False,
                **options,
            )
            try:
                if stdin_content is not None:
                    proc.communicate(input=stdin_content, timeout=timeout)
                else:
                    last_status_len = 0
                    while True:
                        ret_poll = proc.poll()
                        if ret_poll is not None:
                            break
                        now = time.monotonic()
                        elapsed_s = now - start_t
                        if timeout is not None and elapsed_s > timeout:
                            raise subprocess.TimeoutExpired(cmd, timeout)

                        if show_progress and out_p.is_file():
                            status_str = _extract_orca_progress_status(out_p, elapsed_s, input_file)
                            if status_str:
                                pad = " " * max(0, last_status_len - len(status_str))
                                sys.stderr.write(f"\r{status_str}{pad}")
                                sys.stderr.flush()
                                last_status_len = len(status_str)
                        time.sleep(1.0)

                    if show_progress and last_status_len > 0:
                        sys.stderr.write("\r" + " " * last_status_len + "\r")
                        sys.stderr.flush()
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_tree(proc)
                error = f"ORCA execution timed out after {timeout} seconds"
            except KeyboardInterrupt:
                interrupted = True
                _stop_process_tree(proc)
                error = "ORCA execution was interrupted by user"
                raise
    except KeyboardInterrupt:
        interrupted = True
        if proc is not None:
            _stop_process_tree(proc)
        raise
    except Exception as exc:
        if proc is not None:
            _stop_process_tree(proc)
        error = str(exc)
        return ProcessResult(
            returncode=-1,
            terminated_normally=False,
            timed_out=timed_out,
            interrupted=interrupted,
            error=error,
            execution_time_s=time.monotonic() - start_t,
        )
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)

    elapsed = time.monotonic() - start_t
    ret = proc.returncode if proc is not None else -1

    terminated_normally = False
    if ret == 0 and not timed_out and not interrupted:
        if check_normal_termination:
            if out_p.is_file():
                text = out_p.read_text(encoding="utf-8", errors="replace")
                if "ORCA TERMINATED NORMALLY" in text:
                    terminated_normally = True
                else:
                    error = "ORCA process exited with code 0 but missing 'ORCA TERMINATED NORMALLY'"
            else:
                error = "Output file not found after process completion"
        else:
            terminated_normally = True
    elif not error:
        error = f"ORCA exited with return code {ret}"

    return ProcessResult(
        returncode=ret,
        terminated_normally=terminated_normally,
        timed_out=timed_out,
        interrupted=interrupted,
        error=error,
        execution_time_s=elapsed,
    )


def _export_frontiers(job, exe, results, grid, timeout):
    """Export the actual final orbitals with ORCA 6's documented plot menu."""
    plot = exe.with_name('orca_plot.exe' if os.name == 'nt' else 'orca_plot')
    if not plot.is_file():
        raise ValueError('orca_plot is missing from the ORCA installation')
    commands = ['4', str(grid), '5', '7']
    for operator, frontiers in results['orbitals'].items():
        for role in ('homo', 'lumo'):
            commands += ['2', str(frontiers[role]['index']), '3', operator, '11']
    commands += ['12']
    menu = '\n'.join(commands) + '\n'
    (job / 'orca_plot.stdin').write_text(menu, encoding='utf-8')

    plot_res = run_orca_process(
        executable=plot,
        input_file='calculation.gbw',
        output_file=job / 'orca_plot.log',
        cwd=job,
        timeout=timeout if timeout is not None else 300,
        extra_args=['-i'],
        stdin_content=menu,
        check_normal_termination=False,
    )
    if plot_res.returncode != 0:
        raise ValueError('orca_plot failed; inspect orca_plot.log')
    from ..reports.orbital_viewer import export_orbital_report
    fields = discover_frontier_fields(job)
    export_orbital_report(fields, job, formats=('html',))
    return dict(status='completed', grid_points_per_axis=grid, viewer='frontier_orbitals.html',
                source_gbw_sha256=file_hash(job / 'calculation.gbw'),
                executable_sha256=file_hash(plot), spatial_convergence='not assessed')


def run_orca_job(job_dir: str | Path, *, timeout: float | None = None) -> dict:
    """Run once; preserve inputs, logs, hashes and status even when ORCA fails."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Timeout must be a finite positive number')
    job = Path(job_dir).resolve()
    manifest = job / 'job.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or record.get('status') != 'prepared':
        raise ValueError('Only a prepared schema-1 job can be executed; prepare a new job to rerun')
    for name in ('calculation.inp', 'geometry.xyz'):
        if file_hash(job / name) != record['inputs'].get(name):
            raise ValueError('Job inputs changed after preparation; prepare a new job')
    # Recheck allocation at execution time; the job may have moved to a smaller host.
    from ..workflows.ligand_qm import _resources
    _resources(record['parameters']['nprocs'], record['parameters']['maxcore_mb'])
    def save():
        staging = manifest.with_suffix('.json.tmp')
        staging.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staging.replace(manifest)
    proc = None
    old_term = None
    record.update(status='running', started_utc=datetime.now(timezone.utc).isoformat())
    record['results'] = None
    save()
    try:
        exe = find_orca()
        record['orca_executable_sha256'] = file_hash(exe)
        record['orca_executable_name'] = exe.name
        save()
        if os.name == 'posix' and threading.current_thread() is threading.main_thread():
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            old_term = signal.signal(signal.SIGTERM, interrupted)
        options = dict(start_new_session=True) if os.name == 'posix' else dict(
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        env = os.environ.copy()
        env['PATH'] = str(exe.parent) + os.pathsep + env.get('PATH', '')
        with (job / 'calculation.out').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([str(exe), 'calculation.inp'], cwd=job, stdout=log,
                                    stderr=subprocess.STDOUT, env=env, **options)
            last_status_len = 0
            start_t = time.monotonic()
            while True:
                ret_poll = proc.poll()
                if ret_poll is not None:
                    record['returncode'] = ret_poll
                    break
                now = time.monotonic()
                elapsed_s = now - start_t
                if timeout is not None and elapsed_s > timeout:
                    raise subprocess.TimeoutExpired([str(exe), 'calculation.inp'], timeout)
                out_p = job / 'calculation.out'
                if out_p.is_file():
                    status_str = _extract_orca_progress_status(out_p, elapsed_s, 'calculation.inp')
                    if status_str:
                        pad = " " * max(0, last_status_len - len(status_str))
                        sys.stderr.write(f"\r{status_str}{pad}")
                        sys.stderr.flush()
                        last_status_len = len(status_str)
                time.sleep(0.5)
            if last_status_len > 0:
                sys.stderr.write("\r" + " " * last_status_len + "\r")
                sys.stderr.flush()
        if record['returncode'] != 0:
            raise ValueError(f'ORCA exited with code {record["returncode"]}; inspect calculation.out')
        record['results'] = read_qm_results(job / 'calculation.out',
                                            optimize=record['parameters']['optimize'],
                                            frequencies=record['parameters']['frequencies'])
        record['status'] = 'completed'
        grid = record['parameters'].get('orbital_grid')
        if grid is not None:
            try:
                record['orbital_export'] = _export_frontiers(job, exe, record['results'], grid, timeout)
            except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                record['orbital_export'] = dict(status='failed', error=(str(exc) if isinstance(exc, ValueError)
                                                                      else type(exc).__name__))
    except BaseException as exc:
        if proc is not None:
            _stop_process_tree(proc)
        record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['error'] = ('ORCA execution timed out' if isinstance(exc, subprocess.TimeoutExpired)
                           else str(exc) if isinstance(exc, ValueError)
                           else f'{type(exc).__name__}: ORCA execution could not complete')
        record['results'] = None
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        record['outputs'] = {p.name: file_hash(p) for p in sorted(job.iterdir())
                             if p.is_file() and p.name not in ('job.json', 'job.json.tmp')
                             and p.name not in record['inputs']}
        save()
    return record
