"""Dependency-aware continuation for prepared Tier 4 ORCA workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from ..analysis.transition_state import (
    parse_orca_irc_output,
    parse_orca_scan_output,
    parse_orca_ts_output,
    validate_tier4_homogeneity,
)
from ..core.frontier import file_hash
from ..core.parser import parse_orca_results
from ..core.runner import find_orca, run_orca_process


STAGES = {
    "reactant": ("00_reactant_freq.inp", "00_reactant_freq.out", "00_reactant_freq.meta.json"),
    "scan": ("01_scan.inp", "01_scan.out", "01_scan.meta.json"),
    "transition_state": ("02_optts.inp", "02_optts.out", "02_optts.meta.json"),
    "irc": ("02b_irc.inp", "02b_irc.out", "02b_irc.meta.json"),
    "product": ("03_adduct_opt.inp", "03_adduct_opt.out", "03_adduct_opt.meta.json"),
}


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normal_optimization(path: Path) -> bool:
    if not path.is_file():
        return False
    result = parse_orca_results(path)
    return result.converged and result.optimization_converged is True


def _stationary_minimum(path: Path) -> bool:
    if not _normal_optimization(path):
        return False
    result = parse_orca_results(path)
    return bool(result.frequencies) and not [frequency for frequency in result.frequencies if frequency < -15.0]


def _scan_steps(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\bB\s+\d+\s+\d+\s*=\s*[^,]+,\s*[^,]+,\s*(\d+)", text)
    if not match:
        raise ValueError("Could not recover the expected scan length from 01_scan.inp")
    return int(match.group(1))


def _scan_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    result = parse_orca_scan_output(path, work_dir=path.parent)
    return (
        "ORCA TERMINATED NORMALLY" in text
        and "ORCA finished by error termination" not in text
        and "RELAXED SURFACE SCAN RESULTS" in text
        and len(result.points) >= _scan_steps(path.parent / "01_scan.inp")
    )


def _ts_valid(path: Path) -> bool:
    return path.is_file() and parse_orca_ts_output(path).is_valid_first_order_saddle_point


def _irc_valid(path: Path) -> bool:
    return path.is_file() and parse_orca_irc_output(path).connects_reactant_and_product


def _validator(stage: str, *, full_thermo: bool) -> Callable[[Path], bool]:
    if stage == "scan":
        return _scan_valid
    if stage == "transition_state":
        return _ts_valid
    if stage == "irc":
        return _irc_valid
    if stage == "reactant" or (stage == "product" and full_thermo):
        return _stationary_minimum
    return _normal_optimization


def _supporting_hashes(root: Path, stage: str, output_path: Path) -> dict[str, str]:
    paths: list[Path] = []
    if stage == "scan" and output_path.is_file():
        result = parse_orca_scan_output(output_path, work_dir=root)
        paths = [point.xyz_path for point in result.points if point.xyz_path and point.xyz_path.is_file()]
    else:
        property_file = output_path.with_suffix(".property.txt")
        if property_file.is_file():
            paths = [property_file]
    return {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(set(paths))}


def write_tier4_stage_record(
    directory: str | Path,
    stage: str,
    fingerprint: str,
    *,
    valid: bool,
    validation: str,
    dependencies: dict[str, str] | None = None,
) -> Path:
    """Atomically bind a Tier 4 result to its exact input, output and dependencies."""
    root = Path(directory).resolve()
    input_name, output_name, meta_name = STAGES[stage]
    input_path, output_path = root / input_name, root / output_name
    data = {
        "schema_version": 1,
        "stage": stage,
        "fingerprint": fingerprint,
        "input_sha256": file_hash(input_path),
        "output_sha256": file_hash(output_path),
        "supporting_outputs": _supporting_hashes(root, stage, output_path),
        "dependencies": dependencies or {},
        "valid": bool(valid),
        "validation": validation,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = root / meta_name
    _atomic_json(path, data)
    return path


def tier4_stage_reusable(
    directory: str | Path,
    stage: str,
    fingerprint: str,
    *,
    full_thermo: bool = False,
    dependencies: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Require recorded provenance and current scientific validation before reuse."""
    root = Path(directory).resolve()
    input_name, output_name, meta_name = STAGES[stage]
    input_path, output_path, meta_path = root / input_name, root / output_name, root / meta_name
    if not all(path.is_file() for path in (input_path, output_path, meta_path)):
        return False, "input, output or provenance record is missing"
    try:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        hashes_match = (
            record.get("schema_version") == 1
            and record.get("fingerprint") == fingerprint
            and record.get("input_sha256") == file_hash(input_path)
            and record.get("output_sha256") == file_hash(output_path)
            and record.get("supporting_outputs", {}) == _supporting_hashes(root, stage, output_path)
            and record.get("dependencies", {}) == (dependencies or {})
        )
        if not hashes_match:
            return False, "fingerprint, dependency or file hash changed"
        if not record.get("valid"):
            return False, record.get("validation", "recorded result is invalid")
        if not _validator(stage, full_thermo=full_thermo)(output_path):
            return False, "current output failed scientific validation"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"could not validate provenance: {exc}"
    return True, record.get("validation", "validated")


def _archive(directory: Path, prefixes: tuple[str, ...], label: str) -> Path | None:
    artifacts = [
        path for path in directory.iterdir()
        if path.is_file()
        and path.name.startswith(prefixes)
        and not path.name.endswith("_template.inp")
        and path.name not in {"00_reactant_freq.inp", "01_scan.inp"}
    ]
    if not artifacts:
        return None
    attempts = directory / "attempts"
    attempts.mkdir(exist_ok=True)
    number = 1
    while (attempts / f"{label}-{number}").exists():
        number += 1
    destination = attempts / f"{label}-{number}"
    destination.mkdir()
    for artifact in artifacts:
        artifact.replace(destination / artifact.name)
    return destination


def _input_from_xyz(template: Path, xyz: Path, output: Path) -> None:
    template_text = template.read_text(encoding="utf-8")
    marker = re.search(r"\*\s*xyz\s+(-?\d+)\s+(\d+)", template_text, re.IGNORECASE)
    if not marker:
        raise ValueError(f"Charge and multiplicity are missing from {template.name}")
    lines = xyz.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Scan geometry is incomplete: {xyz.name}")
    header = template_text[:marker.start()]
    coordinates = "\n".join(f"  {line}" for line in lines[2:] if line.strip())
    output.write_text(
        f"{header}* xyz {marker.group(1)} {marker.group(2)}\n{coordinates}\n*\n",
        encoding="utf-8",
    )


def _recorded_invalid_current(
    directory: Path,
    stage: str,
    fingerprint: str,
    dependencies: dict[str, str] | None = None,
) -> bool:
    input_name, output_name, meta_name = STAGES[stage]
    input_path, output_path, meta_path = (
        directory / input_name, directory / output_name, directory / meta_name
    )
    if not all(path.is_file() for path in (input_path, output_path, meta_path)):
        return False
    try:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        return (
            record.get("schema_version") == 1
            and record.get("fingerprint") == fingerprint
            and record.get("input_sha256") == file_hash(input_path)
            and record.get("output_sha256") == file_hash(output_path)
            and record.get("supporting_outputs", {}) == _supporting_hashes(directory, stage, output_path)
            and record.get("dependencies", {}) == (dependencies or {})
            and record.get("valid") is False
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def resume_tier4_workflow(
    directory: str | Path,
    *,
    timeout: float,
    full_thermo: bool = False,
    optimize_adduct: bool = False,
    run_irc: bool = False,
) -> dict[str, str]:
    """Continue only the first invalid Tier 4 stage and its dependent descendants."""
    root = Path(directory).expanduser().resolve()
    reaction_path = root / "reaction_definition.json"
    if not reaction_path.is_file():
        raise ValueError("Tier 4 reaction_definition.json is missing")
    reaction = json.loads(reaction_path.read_text(encoding="utf-8"))
    fingerprint = reaction.get("fingerprint")
    if not fingerprint or reaction.get("validation_status") == "BLOCKING":
        raise ValueError("Tier 4 reaction definition is missing a usable validated fingerprint")

    homogeneity = validate_tier4_homogeneity(root)
    if homogeneity["stage_count"] > 1 and not homogeneity["homogeneous"]:
        raise ValueError(f"Tier 4 stage homogeneity check failed: {'; '.join(homogeneity['mismatches'])}")

    orca = None
    states: dict[str, str] = {}

    def execute(stage: str, dependencies: dict[str, str] | None = None) -> None:
        nonlocal orca
        if orca is None:
            orca = find_orca()
        input_name, output_name, _ = STAGES[stage]
        proc = run_orca_process(
            executable=orca,
            input_file=input_name,
            output_file=output_name,
            cwd=root,
            timeout=timeout,
            check_normal_termination=True,
        )
        if not proc.terminated_normally:
            raise RuntimeError(proc.error or f"ORCA {stage} calculation did not terminate normally")
        valid = _validator(stage, full_thermo=full_thermo)(root / output_name)
        validation = {
            "scan": "complete coordinate scan",
            "transition_state": "verified first-order saddle point",
            "irc": "verified forward and backward IRC path",
            "reactant": "verified stationary reactant minimum",
            "product": "verified product minimum" if full_thermo else "converged product optimization",
        }[stage]
        write_tier4_stage_record(
            root, stage, fingerprint, valid=valid,
            validation=validation if valid else f"Invalid {stage} result",
            dependencies=dependencies,
        )
        if not valid:
            raise ValueError(f"ORCA completed, but the {stage} result failed scientific validation")
        states[stage] = "completed"

    if full_thermo:
        reusable, _ = tier4_stage_reusable(root, "reactant", fingerprint, full_thermo=True)
        if reusable:
            states["reactant"] = "reused"
        else:
            if _recorded_invalid_current(root, "reactant", fingerprint):
                raise ValueError("Reactant calculation completed but is not a stationary minimum; change the setup before retrying")
            _archive(root, ("00_reactant_freq",), "reactant-retry")
            execute("reactant")

    reusable_scan, _ = tier4_stage_reusable(root, "scan", fingerprint)
    if reusable_scan:
        states["scan"] = "reused"
    else:
        _archive(root, ("01_scan", "02_optts", "02b_irc", "03_adduct_opt"), "scan-dependency-reset")
        execute("scan")

    scan = parse_orca_scan_output(root / "01_scan.out", work_dir=root)
    scan_hash = file_hash(root / "01_scan.out")
    ts_dependencies = {"scan_output_sha256": scan_hash}
    reusable_ts, _ = tier4_stage_reusable(
        root, "transition_state", fingerprint, dependencies=ts_dependencies
    )
    if reusable_ts:
        states["transition_state"] = "reused"
    else:
        if _recorded_invalid_current(root, "transition_state", fingerprint, ts_dependencies):
            raise ValueError("TS optimization completed but is not a first-order saddle point; change the setup before retrying")
        _archive(root, ("02_optts", "02b_irc"), "ts-dependency-reset")
        if not scan.ts_guess_xyz or not scan.ts_guess_xyz.is_file():
            raise ValueError("Validated scan has no TS-guess geometry; downstream input cannot be regenerated safely")
        _input_from_xyz(root / "02_optts_template.inp", scan.ts_guess_xyz, root / "02_optts.inp")
        execute("transition_state", ts_dependencies)

    if run_irc:
        ts_hash = file_hash(root / "02_optts.out")
        irc_dependencies = {"ts_output_sha256": ts_hash}
        reusable_irc, _ = tier4_stage_reusable(
            root, "irc", fingerprint, dependencies=irc_dependencies
        )
        if reusable_irc:
            states["irc"] = "reused"
        else:
            if _recorded_invalid_current(root, "irc", fingerprint, irc_dependencies):
                raise ValueError("IRC calculation completed but did not connect reactant and product; change the setup before retrying")
            _archive(root, ("02b_irc",), "irc-retry")
            ts_xyz = root / "02_optts.xyz"
            if not ts_xyz.is_file():
                ts_xyz = scan.ts_guess_xyz
            if not ts_xyz or not ts_xyz.is_file():
                raise ValueError("Validated TS has no XYZ geometry; IRC input cannot be generated safely")
            irc_template = root / "02b_irc_template.inp"
            if not irc_template.is_file():
                raise ValueError("IRC template 02b_irc_template.inp is missing")
            _input_from_xyz(irc_template, ts_xyz, root / "02b_irc.inp")
            execute("irc", irc_dependencies)

    if optimize_adduct:
        product_dependencies = {"scan_output_sha256": scan_hash}
        reusable_product, _ = tier4_stage_reusable(
            root, "product", fingerprint,
            full_thermo=full_thermo,
            dependencies=product_dependencies,
        )
        if reusable_product:
            states["product"] = "reused"
        else:
            if _recorded_invalid_current(root, "product", fingerprint, product_dependencies):
                raise ValueError("Product calculation completed but failed validation; change the setup before retrying")
            _archive(root, ("03_adduct_opt",), "product-retry")
            product_xyz = scan.points[-1].xyz_path if scan.points else None
            if not product_xyz or not product_xyz.is_file():
                raise ValueError("Validated scan has no product geometry; product input cannot be regenerated safely")
            _input_from_xyz(root / "03_adduct_opt_template.inp", product_xyz, root / "03_adduct_opt.inp")
            execute("product", product_dependencies)
    return states
