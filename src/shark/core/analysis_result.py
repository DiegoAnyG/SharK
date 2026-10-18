"""Authoritative versioned structured scientific result schema for SharK.

Defines SharKAnalysisResult and EvidenceValue with explicit scientific typing,
evidence provenance, completeness tracking, and persistence helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_STATUSES = {
    "computed",
    "validated",
    "model_derived",
    "not_evaluated",
    "not_applicable",
    "not_available",
    "skipped",
    "failed",
    "invalidated",
    "inconclusive",
    "incomplete",
    "legacy",
}

VALID_EVIDENCE_TYPES = {
    "ab_initio",
    "qm_parsed",
    "md_derived",
    "static_geometry",
    "model_derived",
    "docking_derived",
    "experimental",
    "legacy_unknown",
}

FORMAL_OUTCOME_STATES = {
    "PREPARED",
    "COMPLETED_VALIDATED",
    "INVALIDATED",
    "INCONCLUSIVE",
    "NON_REACTIVE_ENSEMBLE",
}


@dataclass
class EvidenceValue:
    """Represents a single typed scientific output with explicit provenance."""
    name: str
    value: Any | None
    units: Optional[str] = None
    status: str = "computed"
    evidence_type: str = "model_derived"
    source_stage: Optional[str] = None
    source_file: Optional[str] = None
    method: Optional[str] = None
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            # Normalize or record warning rather than crashing on legacy formats
            self.warnings.append(f"Unrecognized status '{self.status}', default to 'incomplete'")
            self.status = "incomplete"
        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            self.warnings.append(f"Unrecognized evidence_type '{self.evidence_type}', default to 'legacy_unknown'")
            self.evidence_type = "legacy_unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "status": self.status,
            "evidence_type": self.evidence_type,
            "source_stage": self.source_stage,
            "source_file": self.source_file,
            "method": self.method,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvidenceValue:
        return cls(
            name=d.get("name", "unnamed"),
            value=d.get("value"),
            units=d.get("units"),
            status=d.get("status", "computed"),
            evidence_type=d.get("evidence_type", "model_derived"),
            source_stage=d.get("source_stage"),
            source_file=d.get("source_file"),
            method=d.get("method"),
            reason=d.get("reason"),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class SharKAnalysisResult:
    """Authoritative, JSON-serializable structured scientific result for SharK."""
    schema_version: str = "2.0"
    analysis_id: str = ""
    workflow: Dict[str, Any] = field(default_factory=lambda: {
        "requested": "fast_analysis",
        "executed": "fast_analysis",
        "status": "completed",
        "degraded": False,
        "degradation_reason": None,
    })
    binding: Dict[str, Any] = field(default_factory=dict)
    static_reactive_geometry: Dict[str, Any] = field(default_factory=dict)
    dynamics: Dict[str, Any] = field(default_factory=dict)
    cluster_qm: Dict[str, Any] = field(default_factory=dict)
    transition_state: Dict[str, Any] = field(default_factory=dict)
    adduct: Dict[str, Any] = field(default_factory=dict)
    feasibility: Dict[str, Any] = field(default_factory=dict)
    experimental: Dict[str, Any] = field(default_factory=dict)
    dft_descriptors: Dict[str, Any] = field(default_factory=dict)
    evidence_layers: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def compute_formal_outcome_state(self) -> str:
        """Classify the overall workflow into a formal JOSS/scientific outcome state.

        States:
        - PREPARED: Inputs generated, external calculations not yet executed.
        - INVALIDATED: Completed or cached outputs failed scientific/fingerprint validation.
        - NON_REACTIVE_ENSEMBLE: MD completed and validated with P_NAC == 0.0 (covalent attack ruled out by ensemble).
        - COMPLETED_VALIDATED: All requested stages completed and passed scientific validators.
        - INCONCLUSIVE: Partial execution, missing TS verification, or degraded stages.
        """
        wf_status = str(self.workflow.get("status", "completed")).lower()
        ts_stat = str(self.transition_state.get("status", "")).lower() if isinstance(self.transition_state, dict) else ""
        if wf_status in ("invalid", "invalidated") or ts_stat == "invalid":
            return "INVALIDATED"
        p_nac_raw = self.dynamics.get("p_nac") if isinstance(self.dynamics, dict) else None
        p_nac_val = p_nac_raw.get("value") if isinstance(p_nac_raw, dict) else (
            p_nac_raw.value if isinstance(p_nac_raw, EvidenceValue) else p_nac_raw
        )
        if p_nac_val is not None and float(p_nac_val) == 0.0 and ts_stat in ("not_applicable", "skipped", "not_evaluated", ""):
            if not self.workflow.get("degraded", False):
                return "NON_REACTIVE_ENSEMBLE"
        if wf_status == "prepared":
            return "PREPARED"
        if wf_status == "completed" and not self.workflow.get("degraded", False):
            return "COMPLETED_VALIDATED"
        return "INCONCLUSIVE"

    def build_evidence_layers(self) -> Dict[str, Any]:
        """Return strictly separated epistemological layers without conflating proxies and observables."""
        outcome = self.compute_formal_outcome_state()
        return {
            "experimental_measurement": self.experimental or {"status": "not_provided"},
            "docking_score": self.binding or {"status": "not_evaluated"},
            "dft_descriptor": self.dft_descriptors or {"status": "not_evaluated"},
            "md_score": self.dynamics or {"status": "not_evaluated"},
            "reaction_evidence": {
                "static_reactive_geometry": self.static_reactive_geometry or {"status": "not_evaluated"},
                "cluster_qm": self.cluster_qm or {"status": "not_evaluated"},
                "transition_state": self.transition_state or {"status": "not_evaluated"},
                "adduct": self.adduct or {"status": "not_evaluated"},
            },
            "validated_outcome": {
                "formal_state": outcome,
                "feasibility": self.feasibility or {"status": "not_evaluated"},
                "degraded": bool(self.workflow.get("degraded", False)),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a nested dictionary."""
        def _serialize_item(val: Any) -> Any:
            if isinstance(val, EvidenceValue):
                return val.to_dict()
            if isinstance(val, dict):
                return {k: _serialize_item(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_serialize_item(v) for v in val]
            return val

        layers = self.evidence_layers if self.evidence_layers else self.build_evidence_layers()
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "workflow": _serialize_item(self.workflow),
            "binding": _serialize_item(self.binding),
            "static_reactive_geometry": _serialize_item(self.static_reactive_geometry),
            "dynamics": _serialize_item(self.dynamics),
            "cluster_qm": _serialize_item(self.cluster_qm),
            "transition_state": _serialize_item(self.transition_state),
            "adduct": _serialize_item(self.adduct),
            "feasibility": _serialize_item(self.feasibility),
            "experimental": _serialize_item(self.experimental),
            "dft_descriptors": _serialize_item(self.dft_descriptors),
            "evidence_layers": _serialize_item(layers),
            "completeness": _serialize_item(self.completeness),
            "warnings": list(self.warnings),
            "provenance": _serialize_item(self.provenance),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def write_evidence_json(self, filepath: str | Path) -> Path:
        """Writes evidence.json to disk."""
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        temporary = p.with_suffix(p.suffix + ".tmp")
        temporary.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(p)
        return p

    def write_analysis_manifest(self, filepath: str | Path) -> Path:
        """Writes analysis_manifest.json containing stage execution states."""
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        def _extract_val_and_reason(item: Any) -> Tuple[Any, Optional[str]]:
            if isinstance(item, dict):
                return item.get("value"), item.get("reason")
            if isinstance(item, EvidenceValue):
                return item.value, item.reason
            return item, None

        p_nac_raw = self.dynamics.get("p_nac") if isinstance(self.dynamics, dict) else None
        p_nac_val, p_nac_reason = _extract_val_and_reason(p_nac_raw)

        s_bind_raw = None
        if isinstance(self.binding, dict):
            for k in ("s_bind", "docking_score", "affinity_score"):
                if self.binding.get(k) is not None:
                    s_bind_raw = self.binding.get(k)
                    break
        s_bind_val, _ = _extract_val_and_reason(s_bind_raw)

        cqm_status = self.cluster_qm.get("status", "not_evaluated") if isinstance(self.cluster_qm, dict) else "not_evaluated"
        ts_status = self.transition_state.get("status", "not_evaluated") if isinstance(self.transition_state, dict) else "not_evaluated"
        adduct_status = self.adduct.get("status", "not_evaluated") if isinstance(self.adduct, dict) else "not_evaluated"

        manifest_data = {
            "schema_version": "1.0",
            "analysis_id": self.analysis_id,
            "workflow": self.workflow,
            "stages": {
                "binding": {
                    "status": "completed" if s_bind_val is not None else "not_evaluated"
                },
                "static_geometry": {
                    "status": "completed" if self.static_reactive_geometry else "not_evaluated"
                },
                "dynamics": {
                    "status": "completed" if p_nac_val is not None else "not_evaluated",
                    "reason": p_nac_reason
                },
                "cluster_qm": {
                    "status": cqm_status
                },
                "transition_state": {
                    "status": ts_status
                },
                "adduct": {
                    "status": adduct_status
                }
            },
            "completeness": self.completeness,
            "provenance": self.provenance,
        }
        temporary = p.with_suffix(p.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(p)
        return p

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SharKAnalysisResult:
        """Constructs a SharKAnalysisResult instance from a dictionary."""
        version = d.get("schema_version", "1.0")
        warnings = list(d.get("warnings", []))
        if version != "2.0":
            warnings.append(f"Loaded legacy schema version '{version}', mapped to version 2.0")

        return cls(
            schema_version="2.0",
            analysis_id=d.get("analysis_id", ""),
            workflow=d.get("workflow", {}),
            experimental=d.get("experimental", {}),
            binding=d.get("binding", {}),
            dft_descriptors=d.get("dft_descriptors", {}),
            static_reactive_geometry=d.get("static_reactive_geometry", {}),
            dynamics=d.get("dynamics", {}),
            cluster_qm=d.get("cluster_qm", {}),
            transition_state=d.get("transition_state", {}),
            adduct=d.get("adduct", {}),
            feasibility=d.get("feasibility", {}),
            completeness=d.get("completeness", {}),
            evidence_layers=d.get("evidence_layers", {}),
            warnings=warnings,
            provenance=d.get("provenance", {}),
        )

    @classmethod
    def from_json(cls, s: Union[str, Path]) -> SharKAnalysisResult:
        """Parses a JSON string or file path into a SharKAnalysisResult."""
        if isinstance(s, Path) or (isinstance(s, str) and "\n" not in s and "{" not in s and Path(s).is_file()):
            content = Path(s).read_text(encoding="utf-8")
        else:
            content = str(s)
        return cls.from_dict(json.loads(content))
