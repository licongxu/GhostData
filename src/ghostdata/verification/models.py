"""Contracts separating planned experiments, raw evidence, and verdicts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _frozen_json_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = deepcopy(dict(value))
    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return MappingProxyType(copied)


def _required_strings(**values: object) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class VerificationSpec:
    verification_id: str
    claim_id: str
    experiment_type: str
    hypothesis: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_invariants: tuple[str, ...] = ()
    origin: str = "fixed_library"

    def __post_init__(self) -> None:
        _required_strings(
            verification_id=self.verification_id,
            claim_id=self.claim_id,
            experiment_type=self.experiment_type,
            hypothesis=self.hypothesis,
            origin=self.origin,
        )
        object.__setattr__(
            self, "parameters", _frozen_json_mapping("parameters", self.parameters)
        )
        object.__setattr__(self, "expected_invariants", tuple(self.expected_invariants))
        if any(
            not isinstance(invariant, str) or not invariant.strip()
            for invariant in self.expected_invariants
        ):
            raise ValueError("expected_invariants must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "claim_id": self.claim_id,
            "experiment_type": self.experiment_type,
            "hypothesis": self.hypothesis,
            "parameters": deepcopy(dict(self.parameters)),
            "expected_invariants": list(self.expected_invariants),
            "origin": self.origin,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationSpec:
        data = dict(payload)
        data["expected_invariants"] = tuple(data.get("expected_invariants", ()))
        return cls(**data)


@dataclass(frozen=True)
class ExecutionEvidence:
    bundle_id: str
    verification_id: str
    claim_id: str
    experiment_type: str
    status: str
    observations: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        _required_strings(
            bundle_id=self.bundle_id,
            verification_id=self.verification_id,
            claim_id=self.claim_id,
            experiment_type=self.experiment_type,
        )
        if self.status not in {"completed", "failed"}:
            raise ValueError(f"unsupported evidence status: {self.status}")
        if self.status == "failed" and (not isinstance(self.error, str) or not self.error):
            raise ValueError("failed evidence must include an error")
        object.__setattr__(
            self, "observations", _frozen_json_mapping("observations", self.observations)
        )
        object.__setattr__(
            self,
            "artifact_paths",
            _frozen_json_mapping("artifact_paths", self.artifact_paths),
        )

    @classmethod
    def failed(
        cls, bundle_id: str, spec: VerificationSpec, error: Exception
    ) -> ExecutionEvidence:
        return cls(
            bundle_id=bundle_id,
            verification_id=spec.verification_id,
            claim_id=spec.claim_id,
            experiment_type=spec.experiment_type,
            status="failed",
            error=str(error),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "verification_id": self.verification_id,
            "claim_id": self.claim_id,
            "experiment_type": self.experiment_type,
            "status": self.status,
            "observations": deepcopy(dict(self.observations)),
            "artifact_paths": deepcopy(dict(self.artifact_paths)),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionEvidence:
        return cls(**dict(payload))


@dataclass(frozen=True)
class ExperimentVerdict:
    verification_id: str
    claim_id: str
    outcome: str
    reason: str
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_strings(
            verification_id=self.verification_id,
            claim_id=self.claim_id,
            reason=self.reason,
        )
        if self.outcome not in {"counterexample", "passed", "inconclusive"}:
            raise ValueError(f"unsupported experiment outcome: {self.outcome}")
        object.__setattr__(
            self, "measurements", _frozen_json_mapping("measurements", self.measurements)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "claim_id": self.claim_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "measurements": deepcopy(dict(self.measurements)),
        }


@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    status: str
    experiments: tuple[ExperimentVerdict, ...]

    def __post_init__(self) -> None:
        _required_strings(claim_id=self.claim_id)
        if self.status not in {"verified", "not_verified", "inconclusive"}:
            raise ValueError(f"unsupported claim status: {self.status}")
        object.__setattr__(self, "experiments", tuple(self.experiments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }


@dataclass(frozen=True)
class VerificationReport:
    bundle_id: str
    verdict: str
    claims: tuple[ClaimVerdict, ...]
    evidence: tuple[ExecutionEvidence, ...]

    def __post_init__(self) -> None:
        _required_strings(bundle_id=self.bundle_id)
        if self.verdict not in {"verified", "not_verified", "inconclusive"}:
            raise ValueError(f"unsupported report verdict: {self.verdict}")
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def ghosts(self) -> tuple[ExperimentVerdict, ...]:
        return tuple(
            experiment
            for claim in self.claims
            for experiment in claim.experiments
            if experiment.outcome == "counterexample"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "verdict": self.verdict,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [item.to_dict() for item in self.evidence],
            "counterexamples": len(self.ghosts),
            "ghosts": [ghost.to_dict() for ghost in self.ghosts],
        }

