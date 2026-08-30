import json

import pandas as pd
import pytest

from ghostdata.demo.charts import attach_visuals, build_visuals
from ghostdata.demo.credit import PROJECT_ROOT, prepare_credit_demo, run_credit_demo
from ghostdata.execution.local import default_compiler
from ghostdata.verification import ExecutionEvidence, VerificationReport, VerificationSpec


def _report(**observations: object) -> VerificationReport:
    return VerificationReport(
        "bundle",
        "not_verified",
        (),
        (
            ExecutionEvidence(
                "bundle",
                "V001",
                "C001",
                "entity_alignment",
                "completed",
                observations=observations,
            ),
        ),
    )


def _spec(feature: str) -> VerificationSpec:
    return VerificationSpec(
        verification_id="V001",
        claim_id="C001",
        experiment_type="entity_alignment",
        hypothesis="entity alignment",
        parameters={
            "target_feature": feature,
            "segment": {},
            "mismatch_fraction": 0.5,
            "seed": 7,
        },
        expected_invariants=("schema",),
    )


def _chart(visuals: dict, chart_id: str) -> dict:
    return next(item for item in visuals["charts"] if item["id"] == chart_id)


def test_credit_visuals_keep_marginal_and_break_relationships() -> None:
    prepared = prepare_credit_demo()
    ghost = default_compiler().execute(prepared.reference, prepared.specs[0]).dataframe
    visuals = build_visuals(
        prepared.reference, ghost, run_credit_demo("local"), prepared.specs[0]
    )

    assert visuals["headline"] == "Same values. Different relationships."
    assert visuals["perturbed_feature"] == prepared.specs[0].parameters["target_feature"]
    marginal = _chart(visuals, "marginal")
    assert marginal["reference"] == marginal["ghost"]
    assert _chart(visuals, "label")["reference"] != _chart(visuals, "label")["ghost"]
    json.dumps(visuals, allow_nan=False)


def test_attach_visuals_accepts_an_explicit_table() -> None:
    frame = pd.DataFrame(
        {
            "churn": [0, 1] * 10,
            "tenure": list(range(20)),
            "spend": list(range(20)),
        }
    )
    payload = attach_visuals(_report(), frame, _spec("tenure"))

    assert payload["visuals"]["perturbed_feature"] == "tenure"
    payload = attach_visuals(run_credit_demo("local"))

    assert payload["verdict"] == "not_verified"
    assert payload["counterexamples"] >= 1
    assert payload["visuals"]["metric"]["candidate"] < payload["visuals"]["metric"]["baseline"]
    assert payload["visuals"]["charts"][0]["id"] == "marginal"


def test_visuals_work_on_a_non_credit_table() -> None:
    frame = pd.DataFrame(
        {
            "churn": [0, 1] * 20,
            "tenure": list(range(40)),
            "spend": [10 + index * 3 for index in range(40)],
        }
    )
    spec = _spec("tenure")
    ghost = default_compiler().execute(frame, spec).dataframe
    visuals = build_visuals(frame, ghost, _report(), spec)

    assert visuals["perturbed_feature"] == "tenure"
    assert visuals["label_column"] == "churn"
    public_text = " ".join(
        f"{chart['title']} {chart['caption']} {chart['x_label']} {chart['y_label']}"
        for chart in visuals["charts"]
    )
    assert "tenure" not in public_text
    assert "churn" not in public_text
    assert "spend" not in public_text
    assert _chart(visuals, "marginal")["x_label"] == "Feature value"
    assert "wrong rows" in _chart(visuals, "label")["caption"]
    assert visuals["paired_column"] == "spend"


def test_visuals_work_after_swapping_in_german_credit() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "data" / "build" / "german_credit.csv")
    spec = _spec("credit_amount")
    ghost = default_compiler().execute(frame, spec).dataframe
    visuals = build_visuals(frame, ghost, _report(), spec)

    assert visuals["perturbed_feature"] == "credit_amount"
    assert visuals["label_column"] == "class"
    public_text = " ".join(
        f"{chart['title']} {chart['caption']} {chart['x_label']} {chart['y_label']}"
        for chart in visuals["charts"]
    )
    assert "credit_amount" not in public_text
    assert "class" not in public_text
    assert {item["id"] for item in visuals["charts"]} >= {"marginal", "label"}


def test_visuals_infer_changed_column_without_a_spec() -> None:
    frame = pd.DataFrame(
        {
            "label": [0, 1] * 12,
            "score": list(range(24)),
            "other": [index * 2 for index in range(24)],
        }
    )
    ghost = default_compiler().execute(frame, _spec("score")).dataframe
    visuals = build_visuals(frame, ghost, _report(), spec=None)

    assert visuals["perturbed_feature"] == "score"


def test_categorical_feature_uses_value_counts() -> None:
    frame = pd.DataFrame(
        {
            "fraud": [0, 1] * 8,
            "color": (["red", "blue", "green"] * 6)[:16],
            "amount": list(range(16)),
        }
    )
    spec = _spec("color")
    ghost = default_compiler().execute(frame, spec).dataframe
    visuals = build_visuals(frame, ghost, _report(), spec)
    marginal = _chart(visuals, "marginal")

    assert marginal["kind"] == "grouped_bars"
    assert marginal["reference"] == marginal["ghost"]


def test_visuals_skip_relationships_when_the_feature_is_constant() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "x": [4000, 4000, 4000, 4000],
            "z": [0.2, 0.3, 0.4, 0.1],
        }
    )
    visuals = build_visuals(
        frame,
        frame,
        VerificationReport("empty", "inconclusive", (), ()),
        _spec("x"),
    )

    assert [item["id"] for item in visuals["charts"]] == ["marginal"]
    assert visuals["metric"]["name"] == "metric"


def test_visuals_use_fewer_bins_when_the_feature_has_ties() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": [1, 1, 2, 2, 3, 3, 1, 2],
            "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.3],
        }
    )
    visuals = build_visuals(frame, frame, _report(invariants={"schema": True}), _spec("x"))

    assert _chart(visuals, "label")["labels"][0].startswith("Bin")


def test_feature_only_table_still_renders_a_marginal() -> None:
    frame = pd.DataFrame({"x": list(range(20))})
    visuals = build_visuals(frame, frame.copy(), _report(), spec=None)

    assert visuals["perturbed_feature"] == "x"
    assert visuals["label_column"] is None
    assert visuals["charts"][0]["id"] == "marginal"


def test_visuals_require_a_shared_feature() -> None:
    with pytest.raises(ValueError, match="feature column"):
        build_visuals(
            pd.DataFrame({"a": ["x"]}),
            pd.DataFrame({"b": ["y"]}),
            _report(),
            spec=None,
        )
