"""Machine-verifiable claims extracted from an agent analysis bundle."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from ghostdata.bundle.analysis import AnalysisBundle


@dataclass(frozen=True)
class Claim:
    claim_id: str
    assertion: str
    evaluator: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    supplied_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "claim_id": self.claim_id,
            "assertion": self.assertion,
            "evaluator": self.evaluator,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(deepcopy(dict(self.parameters))),
        )
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(
            self,
            "supplied_evidence",
            MappingProxyType(deepcopy(dict(self.supplied_evidence))),
        )
        try:
            json.dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("claim fields must be JSON serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "assertion": self.assertion,
            "evaluator": self.evaluator,
            "parameters": deepcopy(dict(self.parameters)),
            "dependencies": list(self.dependencies),
            "supplied_evidence": deepcopy(dict(self.supplied_evidence)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Claim:
        data = dict(payload)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


class ClaimExtractor(Protocol):
    def extract(self, bundle: AnalysisBundle) -> Sequence[Claim]: ...


class BundleClaimExtractor:
    """P0 extractor: accept the structured claims supplied in the bundle."""

    def extract(self, bundle: AnalysisBundle) -> tuple[Claim, ...]:
        return bundle.claims
