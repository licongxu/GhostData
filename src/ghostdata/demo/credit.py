"""Reusable vertical slice for the Give Me Some Credit demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.planner import KnownFailurePlanner
from ghostdata.verification import VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "build" / "givemesomecredit_debug_3k.csv"
WORKER_PATH = PROJECT_ROOT / "demo" / "credit_pipeline" / "worker.py"
TARGET_COLUMN = "SeriousDlqin2yrs"
TARGET_FEATURE = "MonthlyIncome"
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.3
DaytonaVerificationRunner = None


@dataclass(frozen=True)
class PreparedCreditDemo:
    data_path: Path
    reference: pd.DataFrame
    bundle: AnalysisBundle
    planner: KnownFailurePlanner
    baseline: float

    @property
    def specs(self) -> tuple[VerificationSpec, ...]:
        return tuple(self.planner.propose(self.bundle, self.bundle.claims))


def load_credit_data(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(data_path).resolve()
    dataframe = pd.read_csv(path)
    missing = {TARGET_COLUMN, TARGET_FEATURE} - set(dataframe.columns)
    if missing:
        raise ValueError(f"credit dataset is missing required columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError("credit dataset must contain at least one row")
    return dataframe


def credit_score(reference: pd.DataFrame):
    median = float(reference[TARGET_FEATURE].median())

    def score(dataframe: pd.DataFrame) -> float:
        values = dataframe[TARGET_FEATURE].fillna(median)
        return float(roc_auc_score(dataframe[TARGET_COLUMN], -values))

    return score


def fitted_credit_model_score(reference: pd.DataFrame):
    """Fit once on the reference data and score a fixed stratified holdout."""
    positions = np.arange(len(reference))
    train_positions, test_positions = train_test_split(
        positions,
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=reference[TARGET_COLUMN],
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=1000),
    )
    model.fit(
        reference.iloc[train_positions][[TARGET_FEATURE]],
        reference.iloc[train_positions][TARGET_COLUMN],
    )

    def score(dataframe: pd.DataFrame) -> float:
        if len(dataframe) != len(reference):
            raise ValueError("candidate dataset must preserve the reference row count")
        probabilities = model.predict_proba(
            dataframe.iloc[test_positions][[TARGET_FEATURE]]
        )[:, 1]
        return float(
            roc_auc_score(
                dataframe.iloc[test_positions][TARGET_COLUMN], probabilities
            )
        )

    return score


def credit_invariants(
    before: pd.DataFrame, after: pd.DataFrame
) -> Mapping[str, bool]:
    before_values = (
        before[TARGET_FEATURE].sort_values(na_position="first").reset_index(drop=True)
    )
    after_values = (
        after[TARGET_FEATURE].sort_values(na_position="first").reset_index(drop=True)
    )
    return {
        "schema": tuple(before.columns) == tuple(after.columns),
        "missing_rate": before.isna().sum().equals(after.isna().sum()),
        "marginal_distribution": before_values.equals(after_values),
    }


def prepare_credit_demo(
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> PreparedCreditDemo:
    path = Path(data_path).resolve()
    reference = load_credit_data(path)
    baseline = credit_score(reference)(reference)
    claim = Claim(
        claim_id="C001",
        assertion="The preprocessing change preserves model quality.",
        evaluator="model_metric_preservation",
        parameters={
            "metric": "roc_auc",
            "max_drop": 0.0,
            "direction": "higher_is_better",
        },
        supplied_evidence={"roc_auc": baseline},
    )
    bundle = AnalysisBundle(
        bundle_id="credit-preprocessing-demo",
        task="Verify an agent-generated credit preprocessing change.",
        inputs={"dataset": "dataset.csv"},
        agent_output=AgentOutput(metrics={"roc_auc": baseline}),
        claims=(claim,),
    )
    return PreparedCreditDemo(
        data_path=path,
        reference=reference,
        bundle=bundle,
        planner=KnownFailurePlanner(TARGET_FEATURE),
        baseline=baseline,
    )


def build_daytona_job(
    prepared: PreparedCreditDemo,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
) -> "DaytonaJob":
    from ghostdata.execution.daytona import DaytonaJob

    files: dict[str, bytes] = {
        "dataset.csv": prepared.data_path.read_bytes(),
        "worker.py": WORKER_PATH.read_bytes(),
    }
    package_root = PROJECT_ROOT / "src" / "ghostdata"
    for source_path in package_root.rglob("*.py"):
        remote_path = (Path("src") / source_path.relative_to(PROJECT_ROOT / "src")).as_posix()
        files[remote_path] = source_path.read_bytes()
    return DaytonaJob(
        command="PYTHONPATH=src python worker.py",
        files=files,
        evidence_path="evidence.json",
    )


def run_credit_demo(
    backend: Literal["local", "daytona"] = "local",
    data_path: Path | str = DEFAULT_DATA_PATH,
    daytona_settings: "DaytonaSettings | None" = None,
) -> VerificationReport:
    prepared = prepare_credit_demo(data_path)
    if backend == "local":
        runner = LocalVerificationRunner(
            prepared.reference,
            default_compiler(),
            credit_invariants,
            credit_score(prepared.reference),
            metric="roc_auc",
        )
    elif backend == "daytona":
        runner_class = DaytonaVerificationRunner
        if runner_class is None:
            from ghostdata.execution.daytona import DaytonaVerificationRunner as runner_class

        load_dotenv(PROJECT_ROOT / ".env")
        runner = runner_class(
            lambda bundle, spec: build_daytona_job(prepared, bundle, spec),
            settings=daytona_settings,
        )
    else:
        raise ValueError(f"unsupported demo backend: {backend}")

    evaluators = EvaluatorRegistry((ModelMetricPreservationEvaluator(),))
    orchestrator = VerificationOrchestrator(runner, evaluators, max_workers=1)
    return orchestrator.verify(
        prepared.bundle,
        BundleClaimExtractor(),
        prepared.planner,
    )
