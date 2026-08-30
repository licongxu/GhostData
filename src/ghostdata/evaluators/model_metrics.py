"""Evaluator for an agent's claim that a model metric is preserved."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any, Mapping

from ghostdata.bundle import Claim
from ghostdata.verification import ExecutionEvidence, ExperimentVerdict, VerificationSpec


class ModelMetricPreservationEvaluator:
    name = "model_metric_preservation"

    def evaluate(
        self,
        claim: Claim,
        spec: VerificationSpec,
        evidence: ExecutionEvidence,
    ) -> ExperimentVerdict:
        if evidence.status == "failed":
            return self._verdict(
                claim,
                spec,
                "inconclusive",
                f"Experiment did not complete: {evidence.error}",
            )

        parameters = claim.parameters
        metric = parameters.get("metric")
        max_drop = parameters.get("max_drop")
        direction = parameters.get("direction", "higher_is_better")
        if (
            not isinstance(metric, str)
            or not metric.strip()
            or not isinstance(max_drop, (int, float))
            or not isfinite(max_drop)
            or max_drop < 0
            or direction not in {"higher_is_better", "lower_is_better"}
        ):
            return self._verdict(
                claim,
                spec,
                "inconclusive",
                "Claim has invalid model-metric evaluation parameters.",
            )

        invariants = evidence.observations.get("invariants")
        if (
            not isinstance(invariants, Mapping)
            or not invariants
            or any(type(passed) is not bool for passed in invariants.values())
        ):
            return self._verdict(
                claim, spec, "inconclusive", "Execution did not report valid invariants."
            )
        if not all(invariants.values()):
            return self._verdict(
                claim,
                spec,
                "inconclusive",
                "Experiment violated an expected invariant and is not a valid counterexample.",
                {"invariants": dict(invariants)},
            )

        observation = evidence.observations.get("model_metric")
        parsed = self._parse_metric_observation(observation, metric)
        if parsed is None:
            return self._verdict(
                claim,
                spec,
                "inconclusive",
                f"Execution did not report a valid {metric!r} observation.",
            )
        baseline, candidate = parsed
        degradation = (
            baseline - candidate
            if direction == "higher_is_better"
            else candidate - baseline
        )
        measurements = {
            "metric": metric,
            "baseline": baseline,
            "candidate": candidate,
            "degradation": degradation,
            "max_drop": float(max_drop),
            "affected_fraction": evidence.observations.get("affected_fraction", 0.0),
        }
        if degradation > max_drop and not isclose(
            degradation, float(max_drop), rel_tol=1e-9, abs_tol=1e-12
        ):
            return self._verdict(
                claim,
                spec,
                "counterexample",
                f"{metric} degraded by {degradation:.6f}, beyond tolerance {max_drop:.6f}.",
                measurements,
            )
        return self._verdict(
            claim,
            spec,
            "passed",
            f"{metric} degradation stayed within tolerance.",
            measurements,
        )

    @staticmethod
    def _parse_metric_observation(
        observation: Any, expected_metric: str
    ) -> tuple[float, float] | None:
        if not isinstance(observation, Mapping) or observation.get("name") != expected_metric:
            return None
        baseline = observation.get("baseline")
        candidate = observation.get("candidate")
        if (
            not isinstance(baseline, (int, float))
            or not isinstance(candidate, (int, float))
            or not isfinite(baseline)
            or not isfinite(candidate)
        ):
            return None
        return float(baseline), float(candidate)

    @staticmethod
    def _verdict(
        claim: Claim,
        spec: VerificationSpec,
        outcome: str,
        reason: str,
        measurements: Mapping[str, Any] | None = None,
    ) -> ExperimentVerdict:
        return ExperimentVerdict(
            verification_id=spec.verification_id,
            claim_id=claim.claim_id,
            outcome=outcome,
            reason=reason,
            measurements=measurements or {},
        )
