import pytest

from ghostdata.bundle import (
    AgentOutput,
    AnalysisBundle,
    BundleClaimExtractor,
    Claim,
)


def make_claim(**overrides: object) -> Claim:
    values = {
        "claim_id": "C001",
        "assertion": "The preprocessing change preserves model quality.",
        "evaluator": "model_metric_preservation",
        "parameters": {
            "metric": "roc_auc",
            "max_drop": 0.0,
            "direction": "higher_is_better",
        },
    }
    values.update(overrides)
    return Claim(**values)


def make_bundle(**overrides: object) -> AnalysisBundle:
    values = {
        "bundle_id": "bundle-1",
        "task": "Verify preprocessing",
        "inputs": {"dataset": "data.csv"},
        "agent_output": AgentOutput(
            code={"pipeline.py": "def run(data): return data"},
            artifacts={"report": "report.json"},
            metrics={"roc_auc": 0.8},
        ),
        "claims": (make_claim(),),
        "tests": ("tests/test_schema.py",),
    }
    values.update(overrides)
    return AnalysisBundle(**values)


def test_analysis_bundle_round_trip_and_schema_version() -> None:
    bundle = make_bundle()

    restored = AnalysisBundle.from_json(bundle.to_json())

    assert restored.to_dict() == bundle.to_dict()
    assert restored.schema_version == "1.0"


def test_bundle_claim_extractor_returns_structured_manual_claims() -> None:
    bundle = make_bundle()

    assert BundleClaimExtractor().extract(bundle) == bundle.claims


@pytest.mark.parametrize(
    "field,value",
    [("claim_id", ""), ("assertion", " "), ("evaluator", None)],
)
def test_claim_rejects_invalid_required_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_claim(**{field: value})


@pytest.mark.parametrize("invalid", [object(), float("nan")])
def test_claim_rejects_non_json_parameters(invalid: object) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        make_claim(parameters={"invalid": invalid})


def test_claim_copies_caller_mappings_and_restores_dependencies() -> None:
    parameters = {"metric": "roc_auc"}
    supplied = {"source": "agent"}
    claim = make_claim(
        parameters=parameters,
        supplied_evidence=supplied,
        dependencies=["dataset"],
    )
    parameters["metric"] = "f1"
    supplied["source"] = "changed"

    restored = Claim.from_dict(claim.to_dict())

    assert claim.parameters == {"metric": "roc_auc"}
    assert claim.supplied_evidence == {"source": "agent"}
    assert restored.dependencies == ("dataset",)
    with pytest.raises(TypeError):
        claim.parameters["metric"] = "f1"


def test_agent_output_copies_mappings() -> None:
    code = {"analysis.py": "print(1)"}
    output = AgentOutput(code=code, metrics={"score": 1})
    code["analysis.py"] = "changed"

    restored = AgentOutput.from_dict(output.to_dict())

    assert output.code["analysis.py"] == "print(1)"
    assert restored.to_dict() == output.to_dict()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"code": {1: "source"}}, "code"),
        ({"artifacts": {"result": 42}}, "artifacts"),
        ({"metrics": {"": 1.0}}, "metrics"),
        ({"metrics": {"score": float("inf")}}, "metrics"),
        ({"metrics": {"score": "high"}}, "metrics"),
    ],
)
def test_agent_output_rejects_invalid_payloads(
    kwargs: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        AgentOutput(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [("bundle_id", ""), ("task", None), ("schema_version", "")],
)
def test_bundle_rejects_invalid_required_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_bundle(**{field: value})


def test_bundle_rejects_duplicate_claim_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_bundle(claims=(make_claim(), make_claim()))


def test_bundle_rejects_invalid_inputs_and_tests() -> None:
    with pytest.raises(ValueError, match="inputs"):
        make_bundle(inputs={"dataset": 42})
    with pytest.raises(ValueError, match="tests"):
        make_bundle(tests=("",))
