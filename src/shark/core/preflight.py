"""Read-only workflow preflight using the same validators as execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .runner import find_orca
from ..workflows.ligand_qm import prepare_ligand_jobs, _resources
from ..workflows.md_bridge import find_gromacs_pipeline, resolve_md_preset


@dataclass(frozen=True)
class PreflightCheck:
    code: str
    category: str
    status: str
    message: str
    stage: str = "preflight"
    action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    checks: list[PreflightCheck] = field(default_factory=list)
    ligand_charges: dict[str, int] = field(default_factory=dict)
    md_capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.status == "blocking"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "failed" if self.blocking else "passed",
            "checks": [check.to_dict() for check in self.checks],
            "ligand_charges": dict(self.ligand_charges),
            "md_capabilities": dict(self.md_capabilities),
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination


def run_preflight(
    *,
    session: Any,
    execute: bool,
    run_md: bool,
    require_orca: bool,
    pipeline_dir: str | Path | None,
    topology: str | Path | None,
    trajectory: str | Path | None,
    nprocs: int,
    maxcore_mb: int | None,
    ligand_request: dict[str, Any] | None = None,
    md_preset: str = "amber99sb-ildn_spce",
) -> PreflightReport:
    report = PreflightReport()

    if getattr(session, "archive_sha256", ""):
        report.checks.append(PreflightCheck("SESSION_HASHED", "identity", "passed", "Input session hash recorded"))
    else:
        report.checks.append(PreflightCheck(
            "SESSION_HASH_UNAVAILABLE", "identity", "warning",
            "Directory session has no archive hash; file-level provenance remains required",
        ))

    if require_orca:
        try:
            _resources(nprocs, maxcore_mb)
            report.checks.append(PreflightCheck("RESOURCES_VALID", "resources", "passed", "CPU and memory request fits this host"))
        except ValueError as exc:
            report.checks.append(PreflightCheck("RESOURCES_INVALID", "resources", "blocking", str(exc), action="Reduce --nprocs or --maxcore"))
    else:
        report.checks.append(PreflightCheck(
            "ORCA_RESOURCES_NOT_APPLICABLE", "resources", "passed",
            "No ORCA execution was requested; ORCA CPU and memory checks are not applicable",
        ))

    if execute and require_orca:
        if find_orca(allow_none=True):
            report.checks.append(PreflightCheck("ORCA_AVAILABLE", "tool", "passed", "ORCA executable resolved"))
        else:
            report.checks.append(PreflightCheck("ORCA_UNAVAILABLE", "tool", "blocking", "ORCA executable was not found", action="Configure SHARK_ORCA"))

    try:
        preset_info = resolve_md_preset(md_preset)
        report.md_capabilities = {**preset_info, "preset": preset_info["name"]}
        if run_md:
            report.checks.append(PreflightCheck(
                "MD_PRESET_VALID", "chemistry", "passed",
                f"MD preset '{preset_info['name']}' validated ({preset_info['protein_ff']} / {preset_info['water_model']} / {preset_info['ions']})",
            ))
    except ValueError as exc:
        report.checks.append(PreflightCheck("MD_PRESET_INVALID", "chemistry", "blocking", str(exc), action="Choose a supported --md-preset"))

    if run_md:

        try:
            root = find_gromacs_pipeline(pipeline_dir)
            if root is None:
                raise FileNotFoundError("Configured GROMACS pipeline root was not found")
            runner = root / "scripts" / "run_pipeline.sh"
            if execute and not runner.is_file():
                raise FileNotFoundError("Configured GROMACS pipeline is missing scripts/run_pipeline.sh")
            if execute and not shutil.which("bash"):
                raise FileNotFoundError("Bash required by the GROMACS pipeline was not found")
            report.checks.append(PreflightCheck("GROMACS_PIPELINE_VALID", "tool", "passed", "GROMACS pipeline layout resolved"))
        except (FileNotFoundError, ValueError) as exc:
            report.checks.append(PreflightCheck("GROMACS_PIPELINE_INVALID", "tool", "blocking", str(exc), action="Configure --pipeline-dir or SHARK_GROMACS_PIPELINE"))

    for label, value in (("topology", topology), ("trajectory", trajectory)):
        if value is None:
            continue
        path = Path(value).expanduser()
        if path.is_file() and path.stat().st_size > 0:
            report.checks.append(PreflightCheck(f"{label.upper()}_VALID", "input", "passed", f"{label.title()} input is readable"))
        else:
            report.checks.append(PreflightCheck(f"{label.upper()}_INVALID", "input", "blocking", f"{label.title()} input is missing or empty"))

    if ligand_request:
        scratch = Path(os.environ.get("SHARK_SCRATCH", tempfile.gettempdir())).expanduser()
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="shark-preflight-", dir=scratch) as temporary:
                jobs = prepare_ligand_jobs(session, Path(temporary) / "quantum", **ligand_request)
                for job in jobs:
                    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
                    ligand = record["selection"]["ligand_id"]
                    report.ligand_charges[ligand.casefold()] = int(record["parameters"]["charge"])
            report.checks.append(PreflightCheck("CHEMICAL_IDENTITY_VALID", "chemistry", "passed", "Ligand identity, charge and spin are internally consistent"))
        except (ValueError, OSError) as exc:
            status = "blocking" if execute else "warning"
            report.checks.append(PreflightCheck("CHEMICAL_IDENTITY_INVALID", "chemistry", status, str(exc), action="Provide a valid SDF, MOL or MOL2 structure with explicit chemistry"))

    return report
