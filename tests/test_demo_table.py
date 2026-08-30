import json
from pathlib import Path

import pandas as pd
import pytest

import ghostdata.demo.table as table
from ghostdata.demo.table import (
    FrozenSpecsPlanner,
    build_executor_job,
    build_local_runner,
    package_sandbox_files,
    run_table_demo,
)
from ghostdata.tabular import load_table
from ghostdata.verification import VerificationSpec


def _churn_csv(tmp_path: Path) -> Path:
    path = tmp_path / "churn.csv"
    pd.DataFrame(
        {
            "churned": [0, 0, 0, 0, 1, 1, 1, 1] * 8,
            "tenure": [20, 18, 19, 17, 1, 2, 0, 3] * 8,
            "spend": [90, 80, 85, 75, 15, 20, 10, 25] * 8,
        }
    ).to_csv(path, index=False)
    return path


def test_package_and_executor_job_are_table_agnostic(tmp_path: Path) -> None:
    path = _churn_csv(tmp_path)
    files = package_sandbox_files(path, {"worker.py": b"pass"})
    job = build_executor_job(path, "churned")

    assert files["dataset.csv"] == path.read_bytes()
    assert "src/ghostdata/tabular.py" in files
    assert job.role == "executor"
    task = json.loads(job.files["task.json"])
    assert task["label_column"] == "churned"
    assert task["dataset"] == "/data/dataset.csv"
    assert job.volume_files["dataset.csv"] == path.read_bytes()
    assert "model.joblib" in job.volume_files
    assert "dataset.csv" not in job.files
    assert job.code_run
    assert "MonthlyIncome" not in job.files["worker.py"].decode()


def test_run_table_demo_finds_a_ghost_on_churn_and_german(tmp_path: Path) -> None:
    churn_report, churn_spec, churn_analysis = run_table_demo(
        _churn_csv(tmp_path), "churned", "local"
    )
    assert churn_report.verdict == "not_verified"
    assert churn_spec.origin == "sandbox_agent"
    assert churn_spec.parameters["target_feature"] in {"tenure", "spend"}
    assert "tenure" in churn_analysis["inspected_columns"]
    assert churn_analysis["executed_spec_count"] >= 2

    german = Path("data/build/german_credit.csv")
    german_report, german_spec, _analysis = run_table_demo(german, "class", "local")
    assert german_report.verdict == "not_verified"
    assert german_spec.parameters["target_feature"] in load_table(german, "class").columns


def test_run_table_demo_daytona_uses_injected_proposer_and_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _churn_csv(tmp_path)
    reference = load_table(path, "churned")
    local_report, spec, analysis = run_table_demo(path, "churned", "local")
    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id, volume_files=None):
            del bundle, files, label_column, claim_id, volume_files
            return [spec], analysis

    fake_runner = build_local_runner(reference, "churned")
    monkeypatch.setattr(table, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(
        table,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )

    report, observed_spec, observed_analysis = run_table_demo(path, "churned", "daytona")
    assert report.verdict == local_report.verdict
    assert observed_spec.verification_id == spec.verification_id
    assert observed_analysis["inspected_columns"] == analysis["inspected_columns"]


def test_run_table_demo_lazily_imports_daytona_runners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ghostdata.execution.daytona as daytona

    path = _churn_csv(tmp_path)
    reference = load_table(path, "churned")
    _report, spec, analysis = run_table_demo(path, "churned", "local")
    feature = str(spec.parameters["target_feature"])

    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id, volume_files=None):
            del bundle, files, label_column, claim_id, volume_files
            return [spec], analysis

    fake_runner = build_local_runner(reference, "churned")
    monkeypatch.setattr(table, "DaytonaProposalRunner", None)
    monkeypatch.setattr(table, "DaytonaVerificationRunner", None)
    monkeypatch.setattr(daytona, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(
        daytona,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )

    report, _spec, _analysis = run_table_demo(path, "churned", "daytona")
    assert report.verdict == "not_verified"


def test_run_table_demo_rejects_empty_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _churn_csv(tmp_path)

    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id, volume_files=None):
            del bundle, files, label_column, claim_id, volume_files
            return [], {"inspected_columns": ["tenure"]}

    monkeypatch.setattr(table, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(
        table,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: None,
    )
    with pytest.raises(RuntimeError, match="no compilable VerificationSpec"):
        run_table_demo(path, "churned", "daytona")


def test_run_table_demo_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported demo backend"):
        run_table_demo(_churn_csv(tmp_path), "churned", "unknown")


def test_ghost_artifacts_on_a_mixed_table(tmp_path: Path) -> None:
    from ghostdata.demo.artifacts import build_ghost_artifacts
    from ghostdata.demo.table import build_table_bundle
    from ghostdata.tabular import load_table
    from ghostdata.verification import VerificationSpec as Spec

    path = tmp_path / "churn.csv"
    pd.DataFrame(
        {
            "churned": [0, 0, 0, 0, 1, 1, 1, 1] * 8,
            "tenure": [20, 18, 19, 17, 1, 2, 0, 3] * 8,
            "plan": ["pro", "basic"] * 32,
        }
    ).to_csv(path, index=False)
    reference = load_table(path, "churned")
    bundle, _planner, _baseline = build_table_bundle(
        reference, "churned", "discovery-mixed", "task"
    )
    report, spec, _analysis = run_table_demo(path, "churned", "local")
    evidence = next(
        item for item in report.evidence if item.verification_id == spec.verification_id
    )
    parameters = dict(spec.parameters)
    parameters["discovery_id"] = "mixed"
    parameters["execution_backend"] = "local"
    promoted = Spec(
        spec.verification_id,
        spec.claim_id,
        spec.experiment_type,
        spec.hypothesis,
        parameters,
        spec.expected_invariants,
        spec.origin,
    )
    built = build_ghost_artifacts(
        path,
        "churned",
        bundle,
        promoted,
        evidence,
        {"agents": [], "proposal": {}, "verification_report": report.to_dict()},
        tmp_path / "out",
    )
    assert built["label_column"] == "churned"
    assert "plan" in built["model"]["features"]


def test_frozen_specs_planner_returns_the_supplied_spec() -> None:
    spec = VerificationSpec("V001", "C001", "entity_alignment", "Misalign")
    planner = FrozenSpecsPlanner((spec,), {"inspected_columns": ["x"]})
    assert planner.propose(None, None) == [spec]
