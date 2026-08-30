"""Standard input boundary for outputs produced by any data-analysis agent."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from ghostdata.bundle.claim import Claim


def _immutable_string_mapping(name: str, value: Mapping[str, str]) -> Mapping[str, str]:
    copied = deepcopy(dict(value))
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in copied.items()):
        raise ValueError(f"{name} must map strings to strings")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class AgentOutput:
    code: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _immutable_string_mapping("code", self.code))
        object.__setattr__(
            self, "artifacts", _immutable_string_mapping("artifacts", self.artifacts)
        )
        metrics = deepcopy(dict(self.metrics))
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for name, value in metrics.items()
        ):
            raise ValueError("metrics must contain named finite numbers")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": deepcopy(dict(self.code)),
            "artifacts": deepcopy(dict(self.artifacts)),
            "metrics": deepcopy(dict(self.metrics)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentOutput:
        return cls(**dict(payload))


@dataclass(frozen=True)
class AnalysisBundle:
    bundle_id: str
    task: str
    inputs: Mapping[str, str]
    agent_output: AgentOutput
    claims: tuple[Claim, ...]
    tests: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for name, value in {
            "bundle_id": self.bundle_id,
            "task": self.task,
            "schema_version": self.schema_version,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "inputs", _immutable_string_mapping("inputs", self.inputs))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "tests", tuple(self.tests))
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique within a bundle")
        if any(not isinstance(test, str) or not test.strip() for test in self.tests):
            raise ValueError("tests must contain non-empty paths")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "task": self.task,
            "inputs": deepcopy(dict(self.inputs)),
            "agent_output": self.agent_output.to_dict(),
            "claims": [claim.to_dict() for claim in self.claims],
            "tests": list(self.tests),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnalysisBundle:
        data = dict(payload)
        data["agent_output"] = AgentOutput.from_dict(data["agent_output"])
        data["claims"] = tuple(Claim.from_dict(claim) for claim in data.get("claims", ()))
        data["tests"] = tuple(data.get("tests", ()))
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str | bytes) -> AnalysisBundle:
        return cls.from_dict(json.loads(payload))
