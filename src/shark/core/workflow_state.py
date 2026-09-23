"""Atomic workflow-stage state for status and conservative resumption."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .frontier import file_hash


MANIFEST_NAME = "execution_manifest.json"
LOCK_NAME = ".shark-workflow.lock"


class WorkflowBusyError(RuntimeError):
    """Raised when another live SharK process owns a workflow directory."""


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_workflow_lock(work_dir: str | Path) -> str:
    """Acquire an exclusive per-workflow writer lock, reclaiming stale locks."""
    lock_path = Path(work_dir).expanduser().resolve() / LOCK_NAME
    token = uuid4().hex
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "token": token,
        "created_utc": _now(),
    }
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(existing.get("pid", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                owner_pid = 0
            if _process_is_alive(owner_pid):
                raise WorkflowBusyError(
                    f"Workflow directory is already in use by SharK process {owner_pid}"
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            os.write(descriptor, (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
        return token
    raise WorkflowBusyError("Workflow lock changed while a stale lock was being reclaimed")


def release_workflow_lock(work_dir: str | Path, token: str) -> None:
    """Release a workflow lock only when it is still owned by the caller."""
    lock_path = Path(work_dir).expanduser().resolve() / LOCK_NAME
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if existing.get("token") == token:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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
                "parameters": parameters,
                "reason": "Stage parameters changed",
            }
        elif current is None:
            current = {
                "status": "planned",
                "validation": "not_evaluated",
                "fingerprint": fingerprint,
                "parameters": parameters,
            }
        else:
            current["parameters"] = parameters
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


def record_workflow_failure(work_dir: str | Path, failure: dict[str, Any]) -> None:
    """Append a typed failure once while preserving an atomically readable manifest."""
    path = Path(work_dir).expanduser().resolve() / MANIFEST_NAME
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = data.setdefault("failures", [])
    identity = (failure.get("stage"), failure.get("code"), failure.get("message"))
    if not any((item.get("stage"), item.get("code"), item.get("message")) == identity for item in failures):
        item = dict(failure)
        item.setdefault("recorded_utc", _now())
        failures.append(item)
    data["updated_utc"] = _now()
    _atomic_json(path, data)


def workflow_recovery_plan(
    manifest: dict[str, Any] | None,
    jobs: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Classify recorded work without promoting process completion to scientific validation."""
    plan = {"reusable": [], "review_required": [], "repeat_required": [], "pending": []}
    for name, stage in (manifest or {}).get("stages", {}).items():
        status = stage.get("observed_status", stage.get("status", "unknown"))
        validation = stage.get("validation", "not_evaluated")
        if status == "completed" and validation == "validated":
            plan["reusable"].append(name)
        elif status == "completed":
            plan["review_required"].append(name)
        elif status in ("failed", "invalid", "invalidated", "interrupted", "timed_out", "skipped"):
            plan["repeat_required"].append(name)
        else:
            plan["pending"].append(name)
    for job in jobs:
        label = f"ligand_dft:{job.get('ligand', job.get('path', 'job'))}"
        if job.get("status") == "completed":
            plan["reusable"].append(label)
        elif job.get("status") in ("failed", "interrupted", "invalidated", "invalid_manifest", "completed_export_failed"):
            plan["repeat_required"].append(label)
        else:
            plan["pending"].append(label)
    for key in plan:
        plan[key] = sorted(set(plan[key]))
    return plan


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
