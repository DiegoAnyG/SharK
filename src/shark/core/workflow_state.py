"""Atomic workflow-stage state for status and conservative resumption."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .frontier import file_hash


MANIFEST_NAME = "execution_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_fingerprint(session_sha256: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"session_sha256": session_sha256, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_workflow_manifest(
    work_dir: str | Path,
    *,
    project: str,
    session_sha256: str,
    stages: dict[str, dict[str, Any]],
) -> Path:
    """Create or extend a workflow manifest without erasing recorded stage results."""
    path = Path(work_dir).expanduser().resolve() / MANIFEST_NAME
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported execution manifest schema")
        recorded = data.get("session", {}).get("sha256")
        if recorded and session_sha256 and recorded != session_sha256:
            raise ValueError("Work directory belongs to a different input session")
    else:
        data = {
            "schema_version": 1,
            "created_utc": _now(),
            "project": project,
            "session": {"sha256": session_sha256},
            "stages": {},
        }
    for name, parameters in stages.items():
        fingerprint = stage_fingerprint(session_sha256, parameters)
        current = data["stages"].get(name)
        if current and current.get("fingerprint") != fingerprint:
            current = {
                "status": "invalidated",
                "validation": "not_evaluated",
                "fingerprint": fingerprint,
                "reason": "Stage parameters changed",
            }
        elif current is None:
            current = {
                "status": "planned",
                "validation": "not_evaluated",
                "fingerprint": fingerprint,
            }
        data["stages"][name] = current
    data["updated_utc"] = _now()
    _atomic_json(path, data)
    return path


def update_workflow_stage(
    work_dir: str | Path,
    stage: str,
    *,
    status: str,
    validation: str | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    path = Path(work_dir).expanduser().resolve() / MANIFEST_NAME
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    record = data.setdefault("stages", {}).setdefault(stage, {})
    record["status"] = status
    if status == "running":
        record["pid"] = os.getpid()
    else:
        record.pop("pid", None)
    if validation is not None:
        record["validation"] = validation
    if reason:
        record["reason"] = reason
    else:
        record.pop("reason", None)
    if details:
        record["details"] = details
    record["updated_utc"] = _now()
    data["updated_utc"] = record["updated_utc"]
    _atomic_json(path, data)


def inspect_workflow(work_dir: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load root stage state and verify every isolated-ligand job input fingerprint."""
    root = Path(work_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest:
        for stage in manifest.get("stages", {}).values():
            pid = stage.get("pid")
            if stage.get("status") == "running" and isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                except (OSError, ValueError):
                    stage["observed_status"] = "interrupted"
                    stage["observed_reason"] = "Recorded worker process is no longer running"
    jobs = []
    for job_file in sorted(root.rglob("job.json")):
        try:
            record = json.loads(job_file.read_text(encoding="utf-8"))
            changed = [
                name for name, expected in record.get("inputs", {}).items()
                if not (job_file.parent / name).is_file() or file_hash(job_file.parent / name) != expected
            ]
            status = record.get("status", "unknown")
            if status == "completed" and record.get("orbital_export", {}).get("status") == "failed":
                status = "completed_export_failed"
            jobs.append({
                "path": job_file.parent.relative_to(root).as_posix(),
                "status": "invalidated" if changed else status,
                "ligand": record.get("selection", {}).get("ligand_id", job_file.parent.name),
                "changed_inputs": changed,
            })
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            jobs.append({"path": job_file.parent.relative_to(root).as_posix(), "status": "invalid_manifest"})
    return manifest, jobs
