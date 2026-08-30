"""Run one fitted-model discovery experiment inside a Daytona sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.demo.credit import credit_invariants, fitted_credit_model_score
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.verification import VerificationSpec


WORK_DIR = Path(__file__).resolve().parent


def main() -> None:
    bundle = AnalysisBundle.from_json((WORK_DIR / "bundle.json").read_text())
    spec = VerificationSpec.from_dict(
        json.loads((WORK_DIR / "verification.json").read_text())
    )
    reference = pd.read_csv(WORK_DIR / bundle.inputs["dataset"])
    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        credit_invariants,
        fitted_credit_model_score(reference),
        metric="roc_auc",
    )
    evidence = runner.run(bundle, spec)
    (WORK_DIR / "evidence.json").write_text(
        json.dumps(evidence.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
