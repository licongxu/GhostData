"""Promote one measured counterexample into exactly four client artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.demo.artifacts import build_ghost_artifacts
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.tabular import (
    frozen_model_score,
    load_frozen_model,
    score_frozen_model,
    table_invariants,
)
from ghostdata.verification import VerificationSpec


WORK_DIR = Path(__file__).resolve().parent


def _dataset_path(task: dict[str, object], bundle: AnalysisBundle) -> Path:
    dataset = str(task.get("dataset") or bundle.inputs["dataset"])
    path = Path(dataset)
    return path if path.is_absolute() else WORK_DIR / path


def _scorer(task: dict[str, object], reference: pd.DataFrame, label_column: str):
    model_path = Path(str(task.get("model_path") or "/data/model.joblib"))
    if model_path.is_file():
        payload = load_frozen_model(model_path.read_bytes())
        return lambda dataframe: score_frozen_model(payload, dataframe)
    return frozen_model_score(reference, label_column)


def main() -> None:
    bundle = AnalysisBundle.from_json((WORK_DIR / "bundle.json").read_text())
    spec = VerificationSpec.from_dict(
        json.loads((WORK_DIR / "verification.json").read_text())
    )
    task = json.loads((WORK_DIR / "task.json").read_text(encoding="utf-8"))
    label_column = str(task["label_column"])
    reference_path = _dataset_path(task, bundle)
    reference = pd.read_csv(reference_path)
    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        table_invariants,
        _scorer(task, reference, label_column),
        metric="roc_auc",
    )
    evidence = runner.run(bundle, spec)
    discovery = json.loads((WORK_DIR / "discovery_report.json").read_text())
    build_ghost_artifacts(
        reference_path,
        label_column,
        bundle,
        spec,
        evidence,
        discovery,
        WORK_DIR / "outputs",
    )
    (WORK_DIR / "promotion_evidence.json").write_text(
        json.dumps(evidence.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
