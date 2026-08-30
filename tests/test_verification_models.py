import pytest

from ghostdata.verification import (
    ClaimVerdict,
    ExecutionEvidence,
    ExperimentVerdict,
    VerificationReport,
    VerificationSpec,
)


def make_spec(**overrides: object) -> VerificationSpec:
    values = {
        "verification_id": "V001",
        "claim_id": "C001",
        "experiment_type": "entity_alignment",
        "hypothesis": "Misalign valid values",
        "parameters": {"target_feature": "income"},
        "expected_invariants": ("schema",),
    }
    values.update(overrides)
    return VerificationSpec(**values)


def make_evidence(**overrides: object) -> ExecutionEvidence:
    values = {
        "bundle_id": "B001",
        "verification_id": "V001",
        "claim_id": "C001",
        "experiment_type": "entity_alignment",
        "status": "completed",
        "observations": {"invariants": {"schema": True}},
    }
    values.update(overrides)
    return ExecutionEvidence(**values)


def test_verification_spec_round_trip_and_mapping_ownership() -> None:
    parameters = {"target_feature": "income"}
    spec = make_spec(parameters=parameters, expected_invariants=["schema"])
    parameters["target_feature"] = "debt"

    restored = VerificationSpec.from_dict(spec.to_dict())

    assert restored.to_dict() == spec.to_dict()
    assert spec.parameters["target_feature"] == "income"
    assert spec.expected_invariants == ("schema",)
    assert "V001" in spec.to_json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("verification_id", ""),
        ("claim_id", None),
        ("experiment_type", " "),
        ("hypothesis", 42),
        ("origin", ""),
    ],
)
def test_verification_spec_rejects_invalid_required_fields(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        make_spec(**{field: value})


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"parameters": {"bad": object()}}, "JSON serializable"),
        ({"parameters": {"bad": float("nan")}}, "JSON serializable"),
        ({"expected_invariants": ("",)}, "expected_invariants"),
    ],
)
def test_verification_spec_rejects_invalid_structured_fields(
    overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        make_spec(**overrides)


def test_execution_evidence_round_trip_and_failed_factory() -> None:
    completed = make_evidence(artifact_paths={"report": "report.json"})
    restored = ExecutionEvidence.from_dict(completed.to_dict())
    failed = ExecutionEvidence.failed("B001", make_spec(), RuntimeError("boom"))

    assert restored.to_dict() == completed.to_dict()
    assert failed.status == "failed"
    assert failed.error == "boom"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"bundle_id": ""}, "bundle_id"),
        ({"status": "running"}, "status"),
        ({"status": "failed", "error": None}, "include an error"),
        ({"observations": {"bad": object()}}, "observations"),
        ({"artifact_paths": {"bad": float("nan")}}, "artifact_paths"),
    ],
)
def test_execution_evidence_rejects_invalid_contracts(
    overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        make_evidence(**overrides)


def test_verdict_models_and_report_expose_ghosts() -> None:
    ghost = ExperimentVerdict(
        "V001", "C001", "counterexample", "metric degraded", {"damage": 0.1}
    )
    passed = ExperimentVerdict("V002", "C001", "passed", "within tolerance")
    claim = ClaimVerdict("C001", "not_verified", (ghost, passed))
    report = VerificationReport("B001", "not_verified", (claim,), (make_evidence(),))

    payload = report.to_dict()

    assert report.ghosts == (ghost,)
    assert payload["counterexamples"] == 1
    assert payload["claims"][0]["experiments"][0]["measurements"] == {
        "damage": 0.1
    }


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: ExperimentVerdict("", "C", "passed", "ok"), "verification_id"),
        (lambda: ExperimentVerdict("V", "C", "unknown", "ok"), "outcome"),
        (
            lambda: ExperimentVerdict("V", "C", "passed", "ok", {"bad": object()}),
            "measurements",
        ),
        (lambda: ClaimVerdict("", "verified", ()), "claim_id"),
        (lambda: ClaimVerdict("C", "unknown", ()), "claim status"),
        (lambda: VerificationReport("", "verified", (), ()), "bundle_id"),
        (lambda: VerificationReport("B", "unknown", (), ()), "report verdict"),
    ],
)
def test_verdict_models_reject_invalid_contracts(factory: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        factory()
