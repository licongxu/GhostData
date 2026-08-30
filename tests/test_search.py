from dataclasses import replace

import pytest

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.verification import ExecutionEvidence, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


def claim(claim_id: str = "C001", evaluator: str = "model_metric_preservation") -> Claim:
    return Claim(
        claim_id,
        "Metric is preserved",
        evaluator,
        {"metric": "roc_auc", "max_drop": 0.0, "direction": "higher_is_better"},
    )


def bundle(claims: tuple[Claim, ...] | None = None) -> AnalysisBundle:
    return AnalysisBundle(
        "B001",
        "Verify",
        {"dataset": "data.csv"},
        AgentOutput(),
        claims if claims is not None else (claim(),),
    )


def spec(
    verification_id: str = "V001",
    claim_id: str = "C001",
) -> VerificationSpec:
    return VerificationSpec(
        verification_id, claim_id, "entity_alignment", "Misalign"
    )


def evidence(
    experiment: VerificationSpec,
    *,
    candidate: float = 0.7,
    invariants: dict[str, bool] | None = None,
) -> ExecutionEvidence:
    return ExecutionEvidence(
        "B001",
        experiment.verification_id,
        experiment.claim_id,
        experiment.experiment_type,
        "completed",
        {
            "invariants": invariants or {"schema": True},
            "model_metric": {
                "name": "roc_auc",
                "baseline": 0.8,
                "candidate": candidate,
            },
        },
    )


class StaticPlanner:
    def __init__(self, specs: tuple[VerificationSpec, ...]) -> None:
        self.specs = specs

    def propose(self, analysis: AnalysisBundle, claims: tuple[Claim, ...]) -> tuple[VerificationSpec, ...]:
        return self.specs


class StubRunner:
    def __init__(self, candidates: dict[str, float] | None = None) -> None:
        self.candidates = candidates or {}

    def run(
        self, analysis: AnalysisBundle, experiment: VerificationSpec
    ) -> ExecutionEvidence:
        return evidence(
            experiment,
            candidate=self.candidates.get(experiment.verification_id, 0.7),
        )


def orchestrator(runner: object | None = None) -> VerificationOrchestrator:
    return VerificationOrchestrator(
        runner or StubRunner(),
        EvaluatorRegistry((ModelMetricPreservationEvaluator(),)),
        max_workers=3,
    )


def test_orchestrator_builds_not_verified_report_from_counterexample() -> None:
    report = orchestrator().verify(
        bundle(), BundleClaimExtractor(), StaticPlanner((spec(),))
    )

    assert report.verdict == "not_verified"
    assert report.claims[0].status == "not_verified"
    assert report.ghosts[0].verification_id == "V001"
    assert report.evidence[0].to_dict().get("verdict") is None


def test_orchestrator_verifies_claim_when_all_experiments_pass() -> None:
    experiments = (spec("V001"), spec("V002"))
    runner = StubRunner({"V001": 0.8, "V002": 0.81})

    report = orchestrator(runner).verify(
        bundle(), BundleClaimExtractor(), StaticPlanner(experiments)
    )

    assert report.verdict == "verified"
    assert [item.outcome for item in report.claims[0].experiments] == [
        "passed",
        "passed",
    ]


def test_orchestrator_contains_runner_failure_as_inconclusive() -> None:
    class FailingRunner:
        def run(self, analysis: AnalysisBundle, experiment: VerificationSpec) -> ExecutionEvidence:
            raise RuntimeError("sandbox unavailable")

    report = orchestrator(FailingRunner()).verify(
        bundle(), BundleClaimExtractor(), StaticPlanner((spec(),))
    )

    assert report.verdict == "inconclusive"
    assert report.evidence[0].status == "failed"
    assert report.evidence[0].error == "sandbox unavailable"


def test_counterexample_takes_precedence_over_inconclusive_experiment() -> None:
    class PartialRunner:
        def run(self, analysis: AnalysisBundle, experiment: VerificationSpec) -> ExecutionEvidence:
            if experiment.verification_id == "V002":
                raise RuntimeError("failed")
            return evidence(experiment)

    report = orchestrator(PartialRunner()).verify(
        bundle(),
        BundleClaimExtractor(),
        StaticPlanner((spec("V001"), spec("V002"))),
    )

    assert report.verdict == "not_verified"
    assert report.claims[0].status == "not_verified"


def test_claim_without_experiments_and_empty_bundle_are_inconclusive() -> None:
    no_experiments = orchestrator().verify(
        bundle(), BundleClaimExtractor(), StaticPlanner(())
    )
    empty = orchestrator().verify(
        bundle(()), BundleClaimExtractor(), StaticPlanner(())
    )

    assert no_experiments.claims[0].status == "inconclusive"
    assert empty.verdict == "inconclusive"


def test_multiple_claims_aggregate_to_report_verdict() -> None:
    claims = (claim("C001"), claim("C002"))
    experiments = (spec("V001", "C001"), spec("V002", "C002"))
    runner = StubRunner({"V001": 0.8, "V002": 0.7})

    report = orchestrator(runner).verify(
        bundle(claims), BundleClaimExtractor(), StaticPlanner(experiments)
    )

    assert [item.status for item in report.claims] == ["verified", "not_verified"]
    assert report.verdict == "not_verified"


@pytest.mark.parametrize(
    "mutate,error_field",
    [
        (lambda item: replace(item, bundle_id="wrong"), "evidence identity"),
        (lambda item: replace(item, verification_id="wrong"), "evidence identity"),
        (lambda item: replace(item, claim_id="wrong"), "evidence identity"),
        (lambda item: replace(item, experiment_type="wrong"), "evidence identity"),
    ],
)
def test_mismatched_evidence_identity_becomes_failed_evidence(
    mutate: object, error_field: str
) -> None:
    class WrongRunner:
        def run(self, analysis: AnalysisBundle, experiment: VerificationSpec) -> ExecutionEvidence:
            return mutate(evidence(experiment))

    report = orchestrator(WrongRunner()).verify(
        bundle(), BundleClaimExtractor(), StaticPlanner((spec(),))
    )

    assert report.evidence[0].status == "failed"
    assert error_field in report.evidence[0].error


def test_unknown_evaluator_is_inconclusive() -> None:
    unsupported = claim(evaluator="not_registered")
    report = orchestrator().verify(
        bundle((unsupported,)),
        BundleClaimExtractor(),
        StaticPlanner((spec(),)),
    )

    assert report.verdict == "inconclusive"


def test_orchestrator_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        VerificationOrchestrator(StubRunner(), EvaluatorRegistry(), 0)


def test_orchestrator_rejects_invalid_extractor_outputs() -> None:
    analysis = bundle()

    class DuplicateExtractor:
        def extract(self, source: AnalysisBundle) -> tuple[Claim, ...]:
            return (source.claims[0], source.claims[0])

    class ForeignExtractor:
        def extract(self, source: AnalysisBundle) -> tuple[Claim, ...]:
            return (claim("foreign"),)

    with pytest.raises(ValueError, match="duplicate"):
        orchestrator().verify(analysis, DuplicateExtractor(), StaticPlanner(()))
    with pytest.raises(ValueError, match="outside"):
        orchestrator().verify(analysis, ForeignExtractor(), StaticPlanner(()))


def test_orchestrator_rejects_invalid_planner_outputs() -> None:
    analysis = bundle()
    duplicate = (spec("V001"), spec("V001"))
    unknown_claim = (spec("V001", "foreign"),)

    with pytest.raises(ValueError, match="duplicate"):
        orchestrator().verify(analysis, BundleClaimExtractor(), StaticPlanner(duplicate))
    with pytest.raises(ValueError, match="unknown claim"):
        orchestrator().verify(
            analysis, BundleClaimExtractor(), StaticPlanner(unknown_claim)
        )
