"""Reusable vertical slice for the Give Me Some Credit demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd
from sklearn.metrics import roc_auc_score

from ghostdata.bundle import AnalysisBundle
from ghostdata.demo.table import (
    WORKER_PATH,
    build_executor_job,
    build_table_bundle,
    run_table_demo,
)
from ghostdata.planner.agent import StructuredSpecPlanner
from ghostdata.tabular import frozen_model_score
from ghostdata.verification import VerificationReport, VerificationSpec


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "build" / "givemesomecredit_debug_3k.csv"
TARGET_COLUMN = "SeriousDlqin2yrs"
TARGET_FEATURE = "MonthlyIncome"


@dataclass(frozen=True)
class PreparedCreditDemo:
    data_path: Path
    reference: pd.DataFrame
    bundle: AnalysisBundle
    planner: StructuredSpecPlanner
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
    """Fit once on MonthlyIncome only. Fixture helper, not the general path."""
    return frozen_model_score(reference, TARGET_COLUMN, (TARGET_FEATURE,))


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
    bundle, planner, baseline = build_table_bundle(
        reference,
        TARGET_COLUMN,
        "credit-preprocessing-demo",
        "Verify an agent-generated credit preprocessing change.",
    )
    return PreparedCreditDemo(
        data_path=path,
        reference=reference,
        bundle=bundle,
        planner=planner,
        baseline=baseline,
    )


def build_daytona_job(
    prepared: PreparedCreditDemo,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
) -> "DaytonaJob":
    del bundle, spec
    return build_executor_job(prepared.data_path, TARGET_COLUMN)


def run_credit_demo(
    backend: Literal["local", "daytona"] = "local",
    data_path: Path | str = DEFAULT_DATA_PATH,
    daytona_settings: "DaytonaSettings | None" = None,
) -> VerificationReport:
    report, _spec, _analysis = run_table_demo(
        data_path,
        TARGET_COLUMN,
        backend,
        daytona_settings,
        bundle_id="credit-preprocessing-demo",
    )
    return report
