"""Evaluator plugin boundary: evidence is interpreted only after execution."""

from __future__ import annotations

from typing import Protocol

from ghostdata.bundle import Claim
from ghostdata.verification import ExecutionEvidence, ExperimentVerdict, VerificationSpec


class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        claim: Claim,
        spec: VerificationSpec,
        evidence: ExecutionEvidence,
    ) -> ExperimentVerdict: ...


class EvaluatorRegistry:
    def __init__(self, evaluators: tuple[Evaluator, ...] = ()) -> None:
        self._evaluators: dict[str, Evaluator] = {}
        for evaluator in evaluators:
            self.register(evaluator)

    def register(self, evaluator: Evaluator) -> None:
        if evaluator.name in self._evaluators:
            raise ValueError(f"evaluator already registered: {evaluator.name}")
        self._evaluators[evaluator.name] = evaluator

    def evaluate(
        self,
        claim: Claim,
        spec: VerificationSpec,
        evidence: ExecutionEvidence,
    ) -> ExperimentVerdict:
        evaluator = self._evaluators.get(claim.evaluator)
        if evaluator is None:
            return ExperimentVerdict(
                verification_id=spec.verification_id,
                claim_id=claim.claim_id,
                outcome="inconclusive",
                reason=f"No evaluator is registered for {claim.evaluator!r}.",
            )
        return evaluator.evaluate(claim, spec, evidence)

