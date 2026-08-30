from pathlib import Path

import pandas as pd
import pytest

import ghostdata.demo.credit as credit
from ghostdata.execution.local import LocalVerificationRunner, default_compiler


def test_prepare_credit_demo_uses_real_dataset_and_measured_baseline() -> None:
    prepared = credit.prepare_credit_demo()

    assert prepared.data_path == credit.DEFAULT_DATA_PATH.resolve()
    assert prepared.reference.shape == (3000, 11)
    assert prepared.baseline == pytest.approx(0.5536616071428572)
    assert prepared.bundle.agent_output.metrics["roc_auc"] == prepared.baseline
    assert prepared.bundle.inputs == {"dataset": "dataset.csv"}
    assert prepared.specs[0].parameters["target_feature"] == "MonthlyIncome"


def test_credit_invariants_detect_preserved_and_changed_values() -> None:
    prepared = credit.prepare_credit_demo()
    before = prepared.reference.head(20).copy()

    assert all(credit.credit_invariants(before, before.copy()).values())

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
    assert job.files["dataset.csv"] == prepared.data_path.read_bytes()
    assert job.files["worker.py"] == credit.WORKER_PATH.read_bytes()
    assert "src/ghostdata/bundle/analysis.py" in job.files


def test_run_credit_demo_local_returns_real_counterexample() -> None:
    report = credit.run_credit_demo("local")

    assert report.verdict == "not_verified"
    assert len(report.ghosts) == 1
    measurements = report.ghosts[0].measurements
    assert measurements["baseline"] == pytest.approx(0.5536616071428572)
    assert measurements["candidate"] == pytest.approx(0.5316732142857142)


def test_run_credit_demo_daytona_branch_uses_injected_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = credit.prepare_credit_demo()
    fake_runner = LocalVerificationRunner(
        prepared.reference,
        default_compiler(),
        credit.credit_invariants,
        credit.credit_score(prepared.reference),
        "roc_auc",
    )
    monkeypatch.setattr(
        credit,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )

    report = credit.run_credit_demo("daytona")

    assert report.verdict == "not_verified"


def test_run_credit_demo_lazily_imports_daytona_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ghostdata.execution.daytona as daytona

    prepared = credit.prepare_credit_demo()
    fake_runner = LocalVerificationRunner(
        prepared.reference,
        default_compiler(),
        credit.credit_invariants,
        credit.credit_score(prepared.reference),
        "roc_auc",
    )
    monkeypatch.setattr(credit, "DaytonaVerificationRunner", None)
    monkeypatch.setattr(
        daytona,
        "DaytonaVerificationRunner",
        lambda job_factory, settings=None: fake_runner,
    )

    assert credit.run_credit_demo("daytona").verdict == "not_verified"


def test_run_credit_demo_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported demo backend"):
        credit.run_credit_demo("unknown")
