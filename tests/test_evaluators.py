import pytest

from ghostdata.bundle import Claim
from ghostdata.evaluators import (
    EvaluatorRegistry,
    ModelMetricPreservationEvaluator,
)
from ghostdata.verification import ExecutionEvidence, VerificationSpec


def claim(**parameter_overrides: object) -> Claim:
    parameters = {
        "metric": "roc_auc",
        "max_drop": 0.02,
        "direction": "higher_is_better",
    }
    parameters.update(parameter_overrides)
    return Claim("C001", "Metric is preserved", "model_metric_preservation", parameters)


def spec() -> VerificationSpec:
    return VerificationSpec("V001", "C001", "entity_alignment", "Misalign")


def evidence(
    *,
    baseline: object = 0.8,
    candidate: object = 0.7,
    metric: str = "roc_auc",
    invariants: object = None,
) -> ExecutionEvidence:
    return ExecutionEvidence(
        "B001",
        "V001",
        "C001",
        "entity_alignment",
        "completed",
        observations={
            "invariants": {"schema": True} if invariants is None else invariants,
            "model_metric": {
                "name": metric,
                "baseline": baseline,
                "candidate": candidate,
            },
            "affected_fraction": 0.25,
        },
    )


def test_higher_is_better_metric_finds_counterexample() -> None:
    verdict = ModelMetricPreservationEvaluator().evaluate(
        claim(), spec(), evidence()
    )

    assert verdict.outcome == "counterexample"
    assert verdict.measurements["degradation"] == pytest.approx(0.1)
    assert verdict.measurements["affected_fraction"] == 0.25


def test_metric_at_tolerance_boundary_passes() -> None:
    verdict = ModelMetricPreservationEvaluator().evaluate(
        claim(max_drop=0.1), spec(), evidence()
    )

    assert verdict.outcome == "passed"


def test_lower_is_better_metric_uses_opposite_direction() -> None:
    verdict = ModelMetricPreservationEvaluator().evaluate(
        claim(metric="loss", max_drop=0.05, direction="lower_is_better"),
        spec(),
        evidence(baseline=0.2, candidate=0.3, metric="loss"),
    )

    assert verdict.outcome == "counterexample"
    assert verdict.measurements["degradation"] == pytest.approx(0.1)


def test_failed_execution_is_inconclusive() -> None:
    failed = ExecutionEvidence.failed("B001", spec(), RuntimeError("sandbox failed"))

    verdict = ModelMetricPreservationEvaluator().evaluate(claim(), spec(), failed)

    assert verdict.outcome == "inconclusive"
    assert "sandbox failed" in verdict.reason


@pytest.mark.parametrize(
    "parameters",
    [
        {"metric": "", "max_drop": 0.1},
        {"metric": "roc_auc", "max_drop": None},
        {"metric": "roc_auc", "max_drop": -0.1},
        {"metric": "roc_auc", "max_drop": 0.1, "direction": "sideways"},
    ],
)
def test_invalid_claim_parameters_are_inconclusive(
    parameters: dict[str, object]
) -> None:
    invalid = Claim("C001", "Metric", "model_metric_preservation", parameters)

    verdict = ModelMetricPreservationEvaluator().evaluate(invalid, spec(), evidence())

    assert verdict.outcome == "inconclusive"
    assert "invalid" in verdict.reason


@pytest.mark.parametrize("invariants", [None, {}, {"schema": 1}])
def test_missing_or_invalid_invariants_are_inconclusive(invariants: object) -> None:
    observations = {
        "model_metric": {"name": "roc_auc", "baseline": 0.8, "candidate": 0.7}
    }
    if invariants is not None:
        observations["invariants"] = invariants
    item = ExecutionEvidence(
        "B001", "V001", "C001", "entity_alignment", "completed", observations
    )

    verdict = ModelMetricPreservationEvaluator().evaluate(claim(), spec(), item)

    assert verdict.outcome == "inconclusive"


def test_failed_invariant_rejects_experiment_as_counterexample() -> None:
    verdict = ModelMetricPreservationEvaluator().evaluate(
        claim(), spec(), evidence(invariants={"schema": False})
    )

    assert verdict.outcome == "inconclusive"
    assert verdict.measurements["invariants"] == {"schema": False}


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {"name": "f1", "baseline": 0.8, "candidate": 0.7},
        {"name": "roc_auc", "baseline": "high", "candidate": 0.7},
    ],
)
def test_invalid_metric_observation_is_inconclusive(observation: object) -> None:
    item = ExecutionEvidence(
        "B001",
        "V001",
        "C001",
        "entity_alignment",
        "completed",
        {"invariants": {"schema": True}, "model_metric": observation},
    )

    verdict = ModelMetricPreservationEvaluator().evaluate(claim(), spec(), item)

    assert verdict.outcome == "inconclusive"


def test_registry_dispatches_and_handles_unknown_or_duplicate_evaluators() -> None:
    evaluator = ModelMetricPreservationEvaluator()
    registry = EvaluatorRegistry((evaluator,))

    assert registry.evaluate(claim(), spec(), evidence()).outcome == "counterexample"
    unknown = Claim("C001", "Unknown", "not_registered")
    assert registry.evaluate(unknown, spec(), evidence()).outcome == "inconclusive"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(evaluator)
