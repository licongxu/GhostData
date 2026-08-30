"""Promote one measured counterexample into exactly four client artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.demo.artifacts import build_credit_artifacts
from ghostdata.demo.credit import credit_invariants, fitted_credit_model_score
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.verification import VerificationSpec


WORK_DIR = Path(__file__).resolve().parent


def main() -> None:
    bundle = AnalysisBundle.from_json((WORK_DIR / "bundle.json").read_text())
    spec = VerificationSpec.from_dict(
        json.loads((WORK_DIR / "verification.json").read_text())
    )
    reference_path = WORK_DIR / bundle.inputs["dataset"]
    reference = pd.read_csv(reference_path)
    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        credit_invariants,
        fitted_credit_model_score(reference),
        metric="roc_auc",
    )
    evidence = runner.run(bundle, spec)
    discovery = json.loads((WORK_DIR / "discovery_report.json").read_text())
    build_credit_artifacts(
        reference_path,
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
