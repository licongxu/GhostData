"""Coordinate extraction, planning, execution, evaluation, and verdict aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from ghostdata.bundle import AnalysisBundle, Claim, ClaimExtractor
from ghostdata.evaluators import EvaluatorRegistry
from ghostdata.execution.base import VerificationRunner
from ghostdata.planner.base import VerificationPlanner
from ghostdata.verification.models import (
    ClaimVerdict,
    ExecutionEvidence,
    ExperimentVerdict,
    VerificationReport,
    VerificationSpec,
)


class VerificationOrchestrator:
    def __init__(
        self,
        runner: VerificationRunner,
        evaluators: EvaluatorRegistry,
        max_workers: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._runner = runner
        self._evaluators = evaluators
        self._max_workers = max_workers

    def verify(
        self,
        bundle: AnalysisBundle,
        extractor: ClaimExtractor,
        planner: VerificationPlanner,
    ) -> VerificationReport:
        claims = tuple(extractor.extract(bundle))
        self._validate_claims(bundle, claims)
        specs = tuple(planner.propose(bundle, claims))
        self._validate_specs(claims, specs)
        evidence = self._execute(bundle, specs)

        claims_by_id = {claim.claim_id: claim for claim in claims}
        experiments_by_claim: dict[str, list[ExperimentVerdict]] = {
            claim.claim_id: [] for claim in claims
        }
        for spec, item in zip(specs, evidence, strict=True):
            experiment = self._evaluators.evaluate(
                claims_by_id[spec.claim_id], spec, item
            )
            experiments_by_claim[spec.claim_id].append(experiment)

        claim_verdicts = tuple(
            self._claim_verdict(claim, tuple(experiments_by_claim[claim.claim_id]))
            for claim in claims
        )
        statuses = {claim.status for claim in claim_verdicts}
        if "not_verified" in statuses:
            verdict = "not_verified"
        elif statuses and statuses == {"verified"}:
            verdict = "verified"
        else:
            verdict = "inconclusive"
        return VerificationReport(bundle.bundle_id, verdict, claim_verdicts, evidence)

    def _execute(
        self, bundle: AnalysisBundle, specs: tuple[VerificationSpec, ...]
    ) -> tuple[ExecutionEvidence, ...]:
        indexed: dict[int, ExecutionEvidence] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._run_one, bundle, spec): (index, spec)
                for index, spec in enumerate(specs)
            }
            for future in as_completed(futures):
                index, spec = futures[future]
                try:
                    indexed[index] = future.result()
                except Exception as exc:
                    indexed[index] = ExecutionEvidence.failed(bundle.bundle_id, spec, exc)
        return tuple(indexed[index] for index in range(len(specs)))

    def _run_one(
        self, bundle: AnalysisBundle, spec: VerificationSpec
    ) -> ExecutionEvidence:
        evidence = self._runner.run(bundle, spec)
        expected = (
            bundle.bundle_id,
            spec.verification_id,
            spec.claim_id,
            spec.experiment_type,
        )
        observed = (
            evidence.bundle_id,
            evidence.verification_id,
            evidence.claim_id,
            evidence.experiment_type,
        )
        if observed != expected:
            raise ValueError(
                f"evidence identity {observed!r} does not match verification {expected!r}"
            )
        return evidence

    @staticmethod
    def _validate_claims(bundle: AnalysisBundle, claims: tuple[Claim, ...]) -> None:
        bundle_claim_ids = {claim.claim_id for claim in bundle.claims}
        extracted_ids = [claim.claim_id for claim in claims]
        if len(extracted_ids) != len(set(extracted_ids)):
            raise ValueError("claim extractor returned duplicate ids")
        if any(claim_id not in bundle_claim_ids for claim_id in extracted_ids):
            raise ValueError("claim extractor returned a claim outside the bundle")

    @staticmethod
    def _validate_specs(
        claims: tuple[Claim, ...], specs: tuple[VerificationSpec, ...]
    ) -> None:
        claim_ids = {claim.claim_id for claim in claims}
        verification_ids = [spec.verification_id for spec in specs]
        if len(verification_ids) != len(set(verification_ids)):
            raise ValueError("planner returned duplicate verification ids")
        if any(spec.claim_id not in claim_ids for spec in specs):
            raise ValueError("planner returned a verification for an unknown claim")

    @staticmethod
    def _claim_verdict(
        claim: Claim, experiments: tuple[ExperimentVerdict, ...]
    ) -> ClaimVerdict:
        outcomes = {experiment.outcome for experiment in experiments}
        if "counterexample" in outcomes:
            status = "not_verified"
        elif experiments and outcomes == {"passed"}:
            status = "verified"
        else:
            status = "inconclusive"
        return ClaimVerdict(claim.claim_id, status, experiments)

