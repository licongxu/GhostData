import pandas as pd
import pytest

from ghostdata.bundle import AgentOutput, AnalysisBundle, Claim
from ghostdata.planner import StructuredSpecPlanner
from ghostdata.tabular import profile_table


def _bundle(claims: tuple[Claim, ...]) -> AnalysisBundle:
    return AnalysisBundle("B001", "Verify", {"dataset": "data.csv"}, AgentOutput(), claims)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
            "signal": [1.0, 2.0, 1.5, 9.0, 8.0, 10.0, 2.0, 9.5],
            "noise": [3.0, 4.0, 5.0, 4.0, 3.0, 5.0, 4.5, 3.5],
        }
    )


def test_structured_spec_planner_emits_sandbox_agent_origin() -> None:
    frame = _frame()
    analysis = _bundle((Claim("C001", "kept", "model_metric_preservation"),))
    planner = StructuredSpecPlanner(frame, "label")

    specs = planner.propose(analysis, analysis.claims)

    assert len(specs) >= 2
    spec = specs[0]
    assert spec.origin == "sandbox_agent"
    assert spec.experiment_type == "entity_alignment"
    assert spec.parameters["target_feature"] == "signal"
    assert {item.parameters["target_feature"] for item in specs} <= {"signal", "noise"}
    assert planner.last_analysis is not None
    assert planner.last_analysis["ranked_features"][0] == "signal"
    assert "signal" in planner.last_analysis["inspected_columns"]


def test_structured_spec_planner_skips_other_evaluators_and_rejects_foreign_claims() -> None:
    frame = _frame()
    analysis = _bundle((Claim("C001", "schema", "schema"),))
    planner = StructuredSpecPlanner(frame, "label", profile_table(frame, "label"))

    assert planner.propose(analysis, analysis.claims) == []

    other = _bundle((Claim("C001", "kept", "model_metric_preservation"),))
    with pytest.raises(ValueError, match="not part of bundle"):
        planner.propose(other, (Claim("foreign", "kept", "model_metric_preservation"),))


def test_structured_spec_planner_requires_the_label_column() -> None:
    with pytest.raises(ValueError, match="label column missing"):
        StructuredSpecPlanner(_frame(), "missing")
