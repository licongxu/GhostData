import pytest

from ghostdata.bundle import AgentOutput, AnalysisBundle, Claim
from ghostdata.planner import KnownFailurePlanner


def claim(claim_id: str = "C001", evaluator: str = "model_metric_preservation") -> Claim:
    return Claim(claim_id, "A claim", evaluator)


def bundle(claims: tuple[Claim, ...]) -> AnalysisBundle:
    return AnalysisBundle("B001", "Verify analysis", {"dataset": "data.csv"}, AgentOutput(), claims)


def test_known_failure_planner_emits_deterministic_credit_experiment() -> None:
    analysis = bundle((claim(),))

    specs = KnownFailurePlanner("MonthlyIncome", 0.25, 7).propose(
        analysis, analysis.claims
    )

    assert [spec.to_dict() for spec in specs] == [
        {
            "verification_id": "V001",
            "claim_id": "C001",
            "experiment_type": "entity_alignment",
            "hypothesis": (
                "Valid MonthlyIncome values become attached to the wrong entities while "
                "the agent's stated invariants remain unchanged."
            ),
            "parameters": {
                "target_feature": "MonthlyIncome",
                "segment": {},
                "mismatch_fraction": 0.25,
                "seed": 7,
            },
            "expected_invariants": [
                "schema",
                "marginal_distribution",
                "missing_rate",
            ],
            "origin": "fixed_library",
        }
    ]


def test_known_failure_planner_skips_claims_for_other_evaluators() -> None:
    analysis = bundle((claim(evaluator="schema"),))

    assert KnownFailurePlanner("income").propose(analysis, analysis.claims) == []


def test_known_failure_planner_rejects_foreign_claim() -> None:
    analysis = bundle((claim(),))

    with pytest.raises(ValueError, match="not part of bundle"):
        KnownFailurePlanner("income").propose(analysis, (claim("foreign"),))


def test_known_failure_planner_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="target_feature"):
        KnownFailurePlanner("")

