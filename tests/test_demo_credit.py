import json
from pathlib import Path

import pandas as pd
import pytest

import ghostdata.demo.credit as credit
import ghostdata.demo.table as table
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.tabular import feature_invariants, feature_score


def test_prepare_credit_demo_uses_real_dataset_and_measured_baseline() -> None:
    prepared = credit.prepare_credit_demo()
    spec = prepared.specs[0]

    assert prepared.data_path == credit.DEFAULT_DATA_PATH.resolve()
    assert prepared.reference.shape == (3000, 11)
    assert prepared.bundle.agent_output.metrics["roc_auc"] == prepared.baseline
    assert prepared.bundle.inputs == {"dataset": "dataset.csv"}
    assert spec.origin == "sandbox_agent"
    assert spec.experiment_type == "entity_alignment"
    assert spec.parameters["target_feature"] in prepared.reference.columns
    assert spec.parameters["target_feature"] != credit.TARGET_COLUMN


def test_credit_invariants_detect_preserved_and_changed_values() -> None:
    prepared = credit.prepare_credit_demo()
    before = prepared.reference.head(20).copy()

    assert all(credit.credit_invariants(before, before.copy()).values())
    assert credit.credit_score(prepared.reference)(prepared.reference) > 0.5

    changed = before.copy()
    changed.loc[changed.index[0], "MonthlyIncome"] = 999999
    assert not credit.credit_invariants(before, changed)["marginal_distribution"]


def test_fitted_credit_model_uses_fixed_holdout_and_requires_same_rows() -> None:
    prepared = credit.prepare_credit_demo()
    score = credit.fitted_credit_model_score(prepared.reference)

    assert score(prepared.reference) == pytest.approx(0.5423511904761905)
    with pytest.raises(ValueError, match="row count"):
        score(prepared.reference.iloc[:-1])


def test_load_credit_data_rejects_missing_columns_and_empty_data(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"MonthlyIncome": [1]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        credit.load_credit_data(missing)

    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=[credit.TARGET_COLUMN, credit.TARGET_FEATURE]).to_csv(
        empty, index=False
    )
    with pytest.raises(ValueError, match="at least one row"):
        credit.load_credit_data(empty)


def test_build_daytona_job_contains_dataset_worker_and_package() -> None:
    prepared = credit.prepare_credit_demo()

    job = credit.build_daytona_job(
        prepared, prepared.bundle, prepared.specs[0]
    )

    assert job.command == "PYTHONPATH=src python worker.py"
    assert job.role == "executor"
    assert job.files["dataset.csv"] == prepared.data_path.read_bytes()
    assert job.files["worker.py"] == credit.WORKER_PATH.read_bytes()
    assert b"MonthlyIncome" not in job.files["worker.py"]
    assert "src/ghostdata/bundle/analysis.py" in job.files
    task = json.loads(job.files["task.json"])
    assert task["label_column"] == credit.TARGET_COLUMN


def test_run_credit_demo_local_returns_real_counterexample() -> None:
    report = credit.run_credit_demo("local")

    assert report.verdict == "not_verified"
    assert len(report.ghosts) == 1
    measurements = report.ghosts[0].measurements
    assert measurements["candidate"] < measurements["baseline"]


def _stub_daytona(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = credit.prepare_credit_demo()
    spec = prepared.specs[0]
    feature = str(spec.parameters["target_feature"])
    analysis = dict(prepared.planner.last_analysis or {})
    fake_runner = LocalVerificationRunner(
        prepared.reference,
        default_compiler(),
        feature_invariants(feature),
        feature_score(prepared.reference, credit.TARGET_COLUMN, feature),
        "roc_auc",
    )

    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id):
            del bundle, files, label_column, claim_id
            return [spec], analysis

    monkeypatch.setattr(table, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(
        table,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )


def test_run_credit_demo_daytona_branch_uses_injected_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_daytona(monkeypatch)
    assert credit.run_credit_demo("daytona").verdict == "not_verified"


def test_run_credit_demo_lazily_imports_daytona_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ghostdata.execution.daytona as daytona

    prepared = credit.prepare_credit_demo()
    spec = prepared.specs[0]
    feature = str(spec.parameters["target_feature"])
    analysis = dict(prepared.planner.last_analysis or {})
    fake_runner = LocalVerificationRunner(
        prepared.reference,
        default_compiler(),
        feature_invariants(feature),
        feature_score(prepared.reference, credit.TARGET_COLUMN, feature),
        "roc_auc",
    )

    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id):
            del bundle, files, label_column, claim_id
            return [spec], analysis

    monkeypatch.setattr(table, "DaytonaProposalRunner", None)
    monkeypatch.setattr(table, "DaytonaVerificationRunner", None)
    monkeypatch.setattr(daytona, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(
        daytona,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )

    assert credit.run_credit_demo("daytona").verdict == "not_verified"


def test_run_credit_demo_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported demo backend"):
        credit.run_credit_demo("unknown")
