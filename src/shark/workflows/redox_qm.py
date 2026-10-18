"""Delta-SCF redox chemistry workflow for vertical/adiabatic electron affinity and reduction potentials.

Computes one-electron reduction and oxidation energetics (vertical EA, vertical IP,
adiabatic EA, inner/outer-sphere reorganization energy, and standard reduction
potentials vs SHE, SCE, and Ag/AgCl) using implicit solvation (CPCM or SMD).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..core.frontier import file_hash
from ..core.parser import parse_orca_results
from ..core.runner import _read_xyz_geometry, find_orca, run_orca_process


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
SHE_ABSOLUTE_POTENTIAL_V = 4.281
SCE_VS_SHE_V = 0.241
AG_AGCL_VS_SHE_V = 0.197


def calculate_redox_thermodynamics(
    e_ox_hartree: float,
    e_red_hartree: float,
    *,
    g_ox_hartree: float | None = None,
    g_red_hartree: float | None = None,
    e_red_vertical_hartree: float | None = None,
    e_ox_vertical_hartree: float | None = None,
    n_electrons: int = 1,
    s2_red: float | None = None,
    ideal_s2_red: float = 0.75,
) -> dict[str, Any]:
    """Compute Delta-SCF redox observables and standard reduction potentials in Volts."""
    if n_electrons < 1:
        raise ValueError("Number of transferred electrons must be at least 1")
    if not (math.isfinite(e_ox_hartree) and math.isfinite(e_red_hartree)):
        raise ValueError("Electronic energies for oxidized and reduced states must be finite")

    adiabatic_ea_ev = (e_ox_hartree - e_red_hartree) * HARTREE_TO_EV
    vertical_ea_ev = (
        (e_ox_hartree - e_red_vertical_hartree) * HARTREE_TO_EV
        if e_red_vertical_hartree is not None and math.isfinite(e_red_vertical_hartree)
        else adiabatic_ea_ev
    )
    vertical_ip_ev = (
        (e_ox_vertical_hartree - e_ox_hartree) * HARTREE_TO_EV
        if e_ox_vertical_hartree is not None and math.isfinite(e_ox_vertical_hartree)
        else None
    )
    reorganization_ev = (
        (e_red_vertical_hartree - e_red_hartree) * HARTREE_TO_EV
        if e_red_vertical_hartree is not None and math.isfinite(e_red_vertical_hartree)
        else 0.0
    )

    use_gibbs = (
        g_ox_hartree is not None
        and g_red_hartree is not None
        and math.isfinite(g_ox_hartree)
        and math.isfinite(g_red_hartree)
        and g_ox_hartree != 0.0
        and g_red_hartree != 0.0
    )
    if use_gibbs:
        delta_g_hartree = float(g_red_hartree) - float(g_ox_hartree)
        basis = "gibbs_thermochemistry"
    else:
        delta_g_hartree = e_red_hartree - e_ox_hartree
        basis = "electronic_scf_approximation"

    delta_g_ev = delta_g_hartree * HARTREE_TO_EV
    delta_g_kcal = delta_g_hartree * HARTREE_TO_KCAL
    e_abs_v = -delta_g_ev / float(n_electrons)
    e_vs_she_v = e_abs_v - SHE_ABSOLUTE_POTENTIAL_V
    e_vs_sce_v = e_vs_she_v - SCE_VS_SHE_V
    e_vs_ag_agcl_v = e_vs_she_v - AG_AGCL_VS_SHE_V

    spin_valid = True
    s2_deviation = None
    if s2_red is not None and math.isfinite(s2_red):
        s2_deviation = abs(float(s2_red) - float(ideal_s2_red))
        spin_valid = s2_deviation <= 0.50

    return {
        "n_electrons": n_electrons,
        "free_energy_basis": basis,
        "delta_g_red_ev": round(delta_g_ev, 6),
        "delta_g_red_kcal_mol": round(delta_g_kcal, 4),
        "vertical_electron_affinity_ev": round(vertical_ea_ev, 6),
        "adiabatic_electron_affinity_ev": round(adiabatic_ea_ev, 6),
        "vertical_ionization_potential_ev": round(vertical_ip_ev, 6) if vertical_ip_ev is not None else None,
        "reorganization_energy_ev": round(reorganization_ev, 6),
        "e_abs_v": round(e_abs_v, 4),
        "e_red_vs_she_v": round(e_vs_she_v, 4),
        "e_red_vs_sce_v": round(e_vs_sce_v, 4),
        "e_red_vs_ag_agcl_v": round(e_vs_ag_agcl_v, 4),
        "spin_contamination_s2": round(float(s2_red), 6) if s2_red is not None else None,
        "ideal_s2": round(float(ideal_s2_red), 6),
        "s2_deviation": round(s2_deviation, 6) if s2_deviation is not None else None,
        "spin_valid": spin_valid,
    }


def prepare_and_run_redox_workflow(
    job_dir: str | Path,
    *,
    adiabatic: bool = False,
    frequencies: bool = False,
    solvation_model: str = "CPCM",
    execute: bool = False,
    timeout: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare and optionally execute Delta-SCF redox calculations for a ligand job."""
    job = Path(job_dir).expanduser().resolve()
    parent_path = job / "job.json"
    if not parent_path.is_file():
        raise FileNotFoundError(f"Ligand job manifest not found: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parameters = parent.get("parameters", {})

    ref_charge = int(parameters.get("charge", 0))
    ref_mult = int(parameters.get("multiplicity", 1))
    if ref_mult != 1:
        raise ValueError("Delta-SCF redox workflow requires a closed-shell reference state (multiplicity=1)")

    geom_path = job / ("calculation.xyz" if (job / "calculation.xyz").is_file() else "geometry.xyz")
    if not geom_path.is_file():
        raise ValueError("Reference geometry XYZ file is missing for Delta-SCF redox workflow")
    symbols, geom_rows = _read_xyz_geometry(geom_path)

    solv_model = solvation_model.strip().upper()
    if solv_model not in ("CPCM", "SMD"):
        raise ValueError(f"Unsupported solvation model '{solvation_model}'; choose 'CPCM' or 'SMD'")
    solvent = parameters.get("solvent") or "Water"
    method = parameters.get("method", "r2SCAN-3c")
    nprocs = int(parameters.get("nprocs", 1))
    maxcore_mb = int(parameters.get("maxcore_mb", 1024))

    states: dict[str, dict[str, Any]] = {
        "ox_sp": {"charge": ref_charge, "multiplicity": 1, "job_type": "SP"},
        "red_vert": {"charge": ref_charge - 1, "multiplicity": 2, "job_type": "SP"},
        "ox_cat_vert": {"charge": ref_charge + 1, "multiplicity": 2, "job_type": "SP"},
    }
    if adiabatic:
        states["red_opt"] = {
            "charge": ref_charge - 1,
            "multiplicity": 2,
            "job_type": "Opt Freq" if frequencies else "Opt",
        }

    setup = {
        "method": method,
        "solvent": solvent,
        "solvation_model": solv_model,
        "adiabatic": bool(adiabatic),
        "frequencies": bool(frequencies),
        "nprocs": nprocs,
        "maxcore_mb": maxcore_mb,
        "geometry_source": geom_path.name,
        "geometry_sha256": file_hash(geom_path),
        "states": states,
    }
    fingerprint = hashlib.sha256(json.dumps(setup, sort_keys=True).encode("utf-8")).hexdigest()

    root = job / "redox"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "redox.json"

    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("fingerprint") != fingerprint and not resume:
            record = {
                "schema_version": 1,
                "status": "prepared",
                "fingerprint": fingerprint,
                "setup": setup,
                "states": {},
            }
    else:
        record = {
            "schema_version": 1,
            "status": "prepared",
            "fingerprint": fingerprint,
            "setup": setup,
            "states": {},
        }

    for label, spec in states.items():
        inp_name = f"{label}.inp"
        out_name = f"{label}.out"
        kw = f"{method} {spec['job_type']} TightSCF CPCM({solvent})"
        smd_block = f'%cpcm\n smd true\n smdsolvent "{solvent}"\nend\n' if solv_model == "SMD" else ""
        deck = (
            f"! {kw}\n"
            f"%pal nprocs {nprocs} end\n"
            f"%maxcore {maxcore_mb}\n"
            f"{smd_block}"
            f"* xyz {spec['charge']} {spec['multiplicity']}\n"
            + "\n".join(geom_rows)
            + "\n*\n"
        )
        inp_path = root / inp_name
        inp_hash = hashlib.sha256(deck.encode("utf-8")).hexdigest()
        prev = record["states"].get(label, {})
        if not (prev.get("status") == "completed" and prev.get("input_sha256") == inp_hash):
            inp_path.write_text(deck, encoding="utf-8")
            record["states"][label] = {
                "status": "prepared",
                "charge": spec["charge"],
                "multiplicity": spec["multiplicity"],
                "job_type": spec["job_type"],
                "input": inp_name,
                "input_sha256": inp_hash,
                "output": out_name,
            }

    def _save() -> None:
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
        parent["redox"] = record
        p_tmp = parent_path.with_suffix(".json.tmp")
        p_tmp.write_text(json.dumps(parent, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        p_tmp.replace(parent_path)

    _save()
    if not execute:
        return record

    executable = find_orca()
    record["status"] = "running"
    _save()
    try:
        for label in states:
            st = record["states"][label]
            out_path = root / st["output"]
            if st.get("status") == "completed" and out_path.is_file() and file_hash(out_path) == st.get("output_sha256"):
                continue
            st["status"] = "running"
            _save()
            proc = run_orca_process(
                executable=executable,
                input_file=st["input"],
                output_file=out_path,
                cwd=root,
                timeout=timeout,
            )
            if proc.returncode != 0 or not proc.terminated_normally:
                raise ValueError(proc.error or f"Redox state '{label}' calculation failed")
            st["status"] = "completed"
            st["output_sha256"] = file_hash(out_path)
            _save()

        parsed = {label: parse_orca_results(root / label) for label in states}
        for label, res in parsed.items():
            if not res.converged or res.el_energy == 0.0:
                raise ValueError(f"Redox state '{label}' did not converge to a valid SCF energy")

        red_final = parsed["red_opt"] if "red_opt" in parsed else parsed["red_vert"]
        thermo = calculate_redox_thermodynamics(
            e_ox_hartree=parsed["ox_sp"].el_energy,
            e_red_hartree=red_final.el_energy,
            g_ox_hartree=parsed["ox_sp"].gibbs_energy or None,
            g_red_hartree=red_final.gibbs_energy or None,
            e_red_vertical_hartree=parsed["red_vert"].el_energy,
            e_ox_vertical_hartree=parsed["ox_cat_vert"].el_energy,
            n_electrons=1,
            s2_red=red_final.spin_contamination_s2,
            ideal_s2_red=red_final.ideal_s2 if red_final.ideal_s2 is not None else 0.75,
        )
        if not thermo["spin_valid"]:
            raise ValueError(
                f"Reduced doublet state exhibits severe spin contamination (<S^2>={thermo['spin_contamination_s2']})"
            )
        record["thermodynamics"] = thermo
        record["status"] = "completed"
        record["validated"] = True
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["validated"] = False
        record["error"] = str(exc)
        _save()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return record

    _save()
    return record
