"""Generic executor worker: apply a VerificationSpec, run checks, score a frozen model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.tabular import feature_invariants, feature_score
from ghostdata.verification import VerificationSpec


WORK_DIR = Path(__file__).resolve().parent


def main() -> None:
    bundle = AnalysisBundle.from_json((WORK_DIR / "bundle.json").read_text())
    spec = VerificationSpec.from_dict(
        json.loads((WORK_DIR / "verification.json").read_text())
    )
    task = json.loads((WORK_DIR / "task.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(WORK_DIR / bundle.inputs["dataset"])
    label_column = str(task["label_column"])
    target_feature = str(spec.parameters["target_feature"])
    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        feature_invariants(target_feature),
        feature_score(reference, label_column, target_feature),
        metric="roc_auc",
    )
    evidence = runner.run(bundle, spec)
    (WORK_DIR / "evidence.json").write_text(
        json.dumps(evidence.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
