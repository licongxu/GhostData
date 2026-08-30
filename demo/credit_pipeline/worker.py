"""Worker uploaded into a Daytona sandbox for the credit verification demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.metrics import roc_auc_score

from ghostdata.bundle import AnalysisBundle
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.verification import VerificationSpec


WORK_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "SeriousDlqin2yrs"


def main() -> None:
    bundle = AnalysisBundle.from_json((WORK_DIR / "bundle.json").read_text())
    spec = VerificationSpec.from_dict(
        json.loads((WORK_DIR / "verification.json").read_text())
    )
    dataset_path = WORK_DIR / bundle.inputs["dataset"]
    reference = pd.read_csv(dataset_path)
    target_feature = str(spec.parameters["target_feature"])
    median = float(reference[target_feature].median())

    def score(dataframe: pd.DataFrame) -> float:
        values = dataframe[target_feature].fillna(median)
        return float(roc_auc_score(dataframe[TARGET_COLUMN], -values))

    def checks(before: pd.DataFrame, after: pd.DataFrame) -> Mapping[str, bool]:
        before_values = (
            before[target_feature].sort_values(na_position="first").reset_index(drop=True)
        )
        after_values = (
            after[target_feature].sort_values(na_position="first").reset_index(drop=True)
        )
        return {
            "schema": tuple(before.columns) == tuple(after.columns),
            "missing_rate": before.isna().sum().equals(after.isna().sum()),
            "marginal_distribution": before_values.equals(after_values),
        }

    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        checks,
        score,
        metric="roc_auc",
    )
    evidence = runner.run(bundle, spec)
    (WORK_DIR / "evidence.json").write_text(
        json.dumps(evidence.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
