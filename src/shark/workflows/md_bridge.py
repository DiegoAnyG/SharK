"""GROMACS Molecular Dynamics workflow bridge for SharK.

Automates the configuration, parameterization, and launching of nanosecond
GPU-accelerated molecular dynamics simulations from selected docking poses
using the standardized GROMACS md_pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Optional

from ..core.runner import run_external_process


def find_gromacs_pipeline(custom_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Locates the GROMACS pipeline root directory dynamically."""
    if custom_dir and Path(custom_dir).is_dir():
        return Path(custom_dir).resolve()

    env_dir = os.environ.get("SHARK_GROMACS_PIPELINE") or os.environ.get("GROMACS_PIPELINE_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir).resolve()

    # Search standard candidate locations
    candidates = [
        Path.home() / "GROMACS" / "md_pipeline",
        Path("/opt/gromacs/md_pipeline"),
        Path.cwd().parent / "GROMACS" / "md_pipeline"
    ]
    for c in candidates:
        if c.is_dir() and (c / "scripts" / "02_prepare_receptor.py").is_file():
            return c.resolve()

    return None


def resolve_gromacs_tool_paths() -> list[Path]:
    """Returns explicitly configured extra binary directories for MD tools."""
    extra_paths: list[Path] = []
    configured = os.environ.get("SHARK_GROMACS_BIN_DIRS", "")
    for value in configured.split(os.pathsep):
        if not value.strip():
            continue
        c = Path(value).expanduser().resolve()
        if c.is_dir() and c not in extra_paths:
            extra_paths.append(c)
    return extra_paths


@dataclass
class MDRunResult:
    run_id: str
    run_dir: Path
    sim_time_ns: float
    status: str  # "completed", "failed", "timed_out", "prepared"
    execution_time_s: float = 0.0
    error_message: Optional[str] = None
    backbone_rmsd_final: Optional[float] = None
    mean_rmsf: Optional[float] = None
    mean_coord_dist: Optional[float] = None
    dashboard_path: Optional[Path] = None


def setup_and_launch_md(
    receptor_pdb: str | Path,
    ligand_pose_file: str | Path,
    ligand_name: str = "LIG",
    ligand_ref_file: Optional[str | Path] = None,
    sim_time_ns: float = 10.0,
    enable_metal_restraint: bool = True,
    run_now: bool = False,
    pipeline_dir: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    timeout: Optional[float] = None,
) -> MDRunResult:
    """Sets up a GROMACS MD run directory and optionally executes the pipeline."""
    pipeline_root = find_gromacs_pipeline(pipeline_dir)
    if pipeline_root is None:
        raise FileNotFoundError(
            "GROMACS pipeline not found. Set SHARK_GROMACS_PIPELINE or GROMACS_PIPELINE_DIR environment variable."
        )

    rec_p = Path(receptor_pdb).resolve()
    lig_p = Path(ligand_pose_file).resolve()

    run_id = f"run_{ligand_name}_{int(sim_time_ns)}ns"
    base_run_dir = Path(work_dir).resolve() if work_dir else (pipeline_root / "runs")
    run_dir = base_run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = run_dir / "00_prep"
    prep_dir.mkdir(exist_ok=True)

    # Stage inputs in run prep directory
    shutil.copy2(rec_p, prep_dir / "receptor_raw.pdb")
    shutil.copy2(lig_p, prep_dir / f"{ligand_name}_pose.pdb")

    # Also stage inputs in pipeline_root/inputs for compatibility with 00_extract_pose.sh
    pipe_inputs = pipeline_root / "inputs"
    pipe_inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rec_p, pipe_inputs / f"{run_id}_receptor.pdb")
    shutil.copy2(lig_p, pipe_inputs / f"{run_id}_pose.pdb")

    # Stage reference ligand structure with true bond-order chemistry (SDF/MOL/MOL2)
    ref_rel_path = ""
    if ligand_ref_file:
        ref_p = Path(ligand_ref_file).resolve()
        if ref_p.is_file() and ref_p.suffix.lower() in (".sdf", ".mol", ".mol2"):
            ref_ext = ref_p.suffix.lower()
            shutil.copy2(ref_p, prep_dir / f"{ligand_name}_ref{ref_ext}")
            shutil.copy2(ref_p, pipe_inputs / f"{run_id}_ref{ref_ext}")
            ref_rel_path = f"inputs/{run_id}_ref{ref_ext}"
    elif lig_p.is_file() and lig_p.suffix.lower() in (".sdf", ".mol", ".mol2"):
        ref_ext = lig_p.suffix.lower()
        shutil.copy2(lig_p, prep_dir / f"{ligand_name}_ref{ref_ext}")
        shutil.copy2(lig_p, pipe_inputs / f"{run_id}_ref{ref_ext}")
        ref_rel_path = f"inputs/{run_id}_ref{ref_ext}"
    elif lig_p.is_file():
        # Attempt to export genuine SDF with RDKit connectivity if only PDB pose is available
        try:
            from rdkit import Chem
            mol = Chem.MolFromPDBFile(str(lig_p), removeHs=False)
            if mol is not None:
                sdf_path = pipe_inputs / f"{run_id}_ref.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()
                shutil.copy2(sdf_path, prep_dir / f"{ligand_name}_ref.sdf")
                ref_rel_path = f"inputs/{run_id}_ref.sdf"
        except Exception:
            pass

    tool_dirs = [str(p) for p in resolve_gromacs_tool_paths()]
    path_export = f'export PATH="{":".join(tool_dirs)}:$PATH"' if tool_dirs else ""

    cuda_devices = os.environ.get("SHARK_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    if not re.fullmatch(r"[A-Za-z0-9_,.:-]+", cuda_devices):
        raise ValueError("CUDA device selection contains unsupported characters")

    has_fe = False
    if rec_p.is_file():
        with open(rec_p, "r", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    rname = line[17:20].strip()
                    aname = line[12:16].strip()
                    elem = line[76:78].strip() if len(line) >= 78 else ""
                    if rname in ("HEM", "HEME") or aname in ("FE", "FE2") or elem == "FE":
                        has_fe = True
                        break
    effective_metal_restraint = enable_metal_restraint and has_fe

    # Generate a run-local config.env for the configured GROMACS pipeline.
    config_content = f"""# Automatically generated by SharK MD Bridge for GROMACS Pipeline
PROJECT_ROOT="{pipeline_root}"
SCRIPTS_DIR="{pipeline_root}/scripts"
MDP_TEMPLATES_DIR="{pipeline_root}/mdp_templates"

# Environment & PATH resolution for GROMACS / ACPYPE / AmberTools
{path_export}

# Run Identification
RUN_ID="{run_id}"
WORKDIR="{run_dir}"

# Ligand and Receptor Parameters
LIGAND_NAME="{ligand_name}"
LIGAND_RESNAME="{ligand_name[:3].upper()}"
LIGAND_PH="7.4"
LIGAND_CHARGE_METHOD="bcc"
LIGAND_NET_CHARGE="0"
POSRES_FORCE_CONST="1000"
RECEPTOR_NAME="{rec_p.stem}"
CHAIN="ALL"
COFACTORS="HEM,HEME,MUR,CD,CL,FE,GDP,GTP,ADP,ATP,NAD,NADH,NADP,NADPH"
RESTRAINT_FE_N={"true" if effective_metal_restraint else "false"}

# Input Paths (Staged inside {pipeline_root}/inputs and {run_dir}/00_prep)
POLISCREEN_RECEPTOR_PDB="inputs/{run_id}_receptor.pdb"
POLISCREEN_LIGAND_POSE="inputs/{run_id}_pose.pdb"
POLISCREEN_LIGAND_SDF="{ref_rel_path}"

# Force Fields and Solvent Box (Amber / GAFF2 Standard)
PROTEIN_FF="amber99sb-ildn"
WATER_MODEL="spce"
BOX_TYPE="dodecahedron"
BOX_DISTANCE_NM=1.0
SALT_CONCENTRATION_M=0.15

# Thermodynamic and Temporal Parameters
TEMP_K=300.0
PRESSURE_BAR=1.0
DT_PS=0.002
EQ_TIME_PS=100
SIM_TIME_NS={sim_time_ns}

# GPU Acceleration Flags
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{cuda_devices}}}"
GPU_FLAGS="-nb gpu -pme gpu -bonded gpu -update gpu"
"""
    cfg_file = run_dir / "config.env"
    cfg_file.write_text(config_content, encoding="utf-8")

    status = "prepared"
    execution_time_s = 0.0
    error_message = None
    if run_now:
        # Launch pipeline script
        run_script = pipeline_root / "scripts" / "run_pipeline.sh"
        if run_script.is_file():
            print(f"[SharK-MD] Launching simulation {run_id} ({sim_time_ns} ns)...")
            exec_env = os.environ.copy()
            if tool_dirs:
                exec_env["PATH"] = ":".join(tool_dirs) + ":" + exec_env.get("PATH", "")

            log_file = run_dir / "pipeline_exec.log"
            bash = shutil.which("bash")
            if not bash:
                raise FileNotFoundError("Bash is required by the configured GROMACS pipeline but was not found in PATH")

            def md_progress(path: Path, elapsed_s: float) -> str:
                try:
                    with path.open("rb") as stream:
                        stream.seek(0, os.SEEK_END)
                        size = stream.tell()
                        stream.seek(max(0, size - 65536))
                        text = stream.read().decode("utf-8", errors="replace")
                    phases = re.findall(r"^###\s+(.+?)\s+###$", text, re.MULTILINE)
                    steps = re.findall(r"(?:Step\s*=?\s*|step\s+)(\d+)", text, re.IGNORECASE)
                    phase = phases[-1] if phases else "Preparation"
                    step = f" | Step {steps[-1]}" if steps else ""
                    return f"[SharK-MD] {phase}{step} | Elapsed: {int(elapsed_s)} s"
                except OSError:
                    return f"[SharK-MD] Running | Elapsed: {int(elapsed_s)} s"

            proc_res = run_external_process(
                command=[bash, run_script, cfg_file],
                output_file=log_file,
                cwd=run_dir,
                timeout=timeout,
                env=exec_env,
                show_progress=True,
                progress_formatter=md_progress,
                process_name="GROMACS pipeline",
            )
            execution_time_s = proc_res.execution_time_s
            error_message = proc_res.error

            if log_file.is_file():
                log_text = log_file.read_text(encoding="utf-8", errors="replace")
                for phase_label in re.findall(r"^###\s+(.+?)\s+###$", log_text, re.MULTILINE):
                    display_label = "Production MD" if "production" in phase_label.lower() else phase_label
                    print(f"[SharK-MD] -> {display_label}")

            if proc_res.terminated_normally:
                status = "completed"
                print(f"[SharK-MD] Simulation {run_id} completed successfully.")
            else:
                status = "timed_out" if proc_res.timed_out else "failed"
                print(f"[SharK-MD] Simulation {run_id} {status}. Log saved to {log_file}")
                if proc_res.error:
                    print(f"[SharK-MD] {proc_res.error}")
                err_lines = []
                if log_file.is_file():
                    err_lines = [line.strip() for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-15:] if line.strip()]
                if err_lines:
                    print("[SharK-MD] Error tail:\n" + "\n".join(err_lines))
        else:
            status = "failed"
            error_message = "Configured GROMACS pipeline is missing scripts/run_pipeline.sh"
            print(f"[SharK-MD] {error_message}")

    # Check for analysis outputs
    dash_path = prep_dir / "md_analysis_dashboard.png"
    return MDRunResult(
        run_id=run_id,
        run_dir=run_dir,
        sim_time_ns=sim_time_ns,
        status=status,
        execution_time_s=execution_time_s,
        error_message=error_message,
        dashboard_path=dash_path if dash_path.is_file() else None
    )


def run_md_from_session(
    session: object,
    ligand_id: str,
    pose_idx: int = 1,
    sim_time_ns: float = 10.0,
    enable_metal_restraint: bool = True,
    run_now: bool = False,
    pipeline_dir: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    receptor_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> MDRunResult:
    """Extracts raw receptor and pose from PoliScreen session and prepares/launches MD."""
    pose = session.get_pose(ligand_id, pose_idx=pose_idx, receptor_id=receptor_id)
    if pose is None:
        raise ValueError(f"Pose not found in session: ligand={ligand_id}, pose_idx={pose_idx}, receptor={receptor_id}")

    # Retrieve the raw receptor PDB required for pdb2gmx
    raw_receptor_pdb = session.receptor_for(pose.receptor_id, raw=True)
    prep_receptor_pdb = session.receptor_for(pose.receptor_id, raw=False)
    print(f"[SharK-MD] Target Receptor ID: {pose.receptor_id}")
    print(f"[SharK-MD] Staging RAW crystal structure: {raw_receptor_pdb.name} (docking-prepared '{prep_receptor_pdb.name}' bypassed)")
    print(f"[SharK-MD] Classical Force Field: AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (water, 0.15 M NaCl)")

    # Locate reference ligand topology (MOL2 / SDF) if present in session
    ref_lig_path = None
    if hasattr(session, "ligand_files") and session.ligand_files:
        ref_lig_path = session.ligand_files.get(ligand_id.casefold()) or session.ligand_files.get(ligand_id)

    if not ref_lig_path and hasattr(session, "extract_dir") and session.extract_dir:
        for folder in ("input_ligands", "ligandos_entrada", ""):
            cand_dir = session.extract_dir / folder
            if cand_dir.is_dir():
                for f in cand_dir.iterdir():
                    if f.is_file() and f.stem.casefold() == ligand_id.casefold() and f.suffix.lower() in (".mol2", ".sdf", ".mol"):
                        ref_lig_path = f
                        break
            if ref_lig_path:
                break

    if ref_lig_path:
        print(f"[SharK-MD] Matched reference ligand topology: {ref_lig_path.name}")
    else:
        print(f"[SharK-MD] No reference SDF/MOL2 found for {ligand_id}; using OpenBabel pose conversion.")

    return setup_and_launch_md(
        receptor_pdb=raw_receptor_pdb,
        ligand_pose_file=pose.pose_file,
        ligand_name=ligand_id,
        ligand_ref_file=ref_lig_path,
        sim_time_ns=sim_time_ns,
        enable_metal_restraint=enable_metal_restraint,
        run_now=run_now,
        pipeline_dir=pipeline_dir,
        work_dir=work_dir,
        timeout=timeout,
    )
