"""Build and validate the four client-facing artifacts for one credit Ghost."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from math import isclose
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.demo.credit import (
    MODEL_RANDOM_STATE,
    MODEL_TEST_SIZE,
    TARGET_COLUMN,
    TARGET_FEATURE,
    credit_invariants,
    fitted_credit_model_score,
    load_credit_data,
)
from ghostdata.execution.local import default_compiler
from ghostdata.verification import ExecutionEvidence, VerificationSpec


ARTIFACT_NAMES = {
    "transform_code": "transform.py",
    "degraded_dataset": "ghost_dataset.csv",
    "model_report": "model_report.json",
    "regression_contract": "regression_contract.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_source(spec: VerificationSpec) -> str:
    target = json.dumps(spec.parameters["target_feature"])
    segment = json.dumps(spec.parameters.get("segment", {}), sort_keys=True)
    fraction = float(spec.parameters["mismatch_fraction"])
    seed = int(spec.parameters["seed"])
    return dedent(
        f'''\
        """Reproduce GhostData counterexample {spec.verification_id}."""

        import argparse
        import json

        import numpy as np
        import pandas as pd


        TARGET_FEATURE = {target}
        SEGMENT = json.loads({json.dumps(segment)})
        MISMATCH_FRACTION = {fraction!r}
        SEED = {seed}


        def transform(dataframe: pd.DataFrame) -> pd.DataFrame:
            eligible = np.ones(len(dataframe), dtype=bool)
            for column, expected in SEGMENT.items():
                eligible &= dataframe[column].eq(expected).to_numpy()
            candidates = np.flatnonzero(eligible)
            count = min(len(candidates), round(len(candidates) * MISMATCH_FRACTION))
            transformed = dataframe.copy(deep=True)
            if count < 2:
                return transformed
            selected = np.random.default_rng(SEED).choice(
                candidates, size=count, replace=False
            )
            position = dataframe.columns.get_loc(TARGET_FEATURE)
            original = transformed.iloc[selected, position].to_numpy(copy=True)
            transformed.iloc[selected, position] = np.roll(original, 1)
            return transformed


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("input_csv")
            parser.add_argument("output_csv")
            args = parser.parse_args()
            transform(pd.read_csv(args.input_csv)).to_csv(args.output_csv, index=False)


        if __name__ == "__main__":
            main()
        '''
    )


def _contract_source(max_drop: float) -> str:
    return dedent(
        f'''\
        """Regression contract promoted from a measured GhostData counterexample."""

        import argparse
        import json
        import sys

        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline


        TARGET_COLUMN = {TARGET_COLUMN!r}
        TARGET_FEATURE = {TARGET_FEATURE!r}
        MAX_AUC_DROP = {max_drop!r}
        RANDOM_STATE = {MODEL_RANDOM_STATE}
        TEST_SIZE = {MODEL_TEST_SIZE!r}


        def evaluate(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
            if list(candidate.columns) != list(reference.columns):
                raise ValueError("candidate schema differs from the reference")
            if len(candidate) != len(reference):
                raise ValueError("candidate row count differs from the reference")
            positions = np.arange(len(reference))
            train_positions, test_positions = train_test_split(
                positions,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=reference[TARGET_COLUMN],
            )
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            )
            model.fit(
                reference.iloc[train_positions][[TARGET_FEATURE]],
                reference.iloc[train_positions][TARGET_COLUMN],
            )

            def auc(dataframe: pd.DataFrame) -> float:
                probabilities = model.predict_proba(
                    dataframe.iloc[test_positions][[TARGET_FEATURE]]
                )[:, 1]
                return float(
                    roc_auc_score(
                        dataframe.iloc[test_positions][TARGET_COLUMN], probabilities
                    )
                )

            baseline = auc(reference)
            candidate_auc = auc(candidate)
            return {{
                "baseline_auc": baseline,
                "candidate_auc": candidate_auc,
                "auc_drop": baseline - candidate_auc,
                "max_auc_drop": MAX_AUC_DROP,
            }}


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("reference_csv")
            parser.add_argument("candidate_csv")
            args = parser.parse_args()
            result = evaluate(
                pd.read_csv(args.reference_csv), pd.read_csv(args.candidate_csv)
            )
            print(json.dumps(result, sort_keys=True))
            return int(result["auc_drop"] > MAX_AUC_DROP + 1e-12)


        if __name__ == "__main__":
            sys.exit(main())
        '''
    )


def build_credit_artifacts(
    reference_path: Path,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
    evidence: ExecutionEvidence,
    discovery: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Recompute the winner and write exactly four files into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("artifact destination must be empty")

    reference = load_credit_data(reference_path)
    transformed = default_compiler().execute(reference, spec)
    scorer = fitted_credit_model_score(reference)
    baseline = scorer(reference)
    candidate = scorer(transformed.dataframe)
    invariants = dict(credit_invariants(reference, transformed.dataframe))
    observed_metric = evidence.observations.get("model_metric", {})
    if (
        evidence.status != "completed"
        or not all(invariants.values())
        or not isclose(float(observed_metric.get("baseline", -1)), baseline, abs_tol=1e-12)
        or not isclose(float(observed_metric.get("candidate", -1)), candidate, abs_tol=1e-12)
    ):
        raise ValueError("promotion measurements do not match discovery evidence")

    transform_path = destination / ARTIFACT_NAMES["transform_code"]
    dataset_path = destination / ARTIFACT_NAMES["degraded_dataset"]
    contract_path = destination / ARTIFACT_NAMES["regression_contract"]
    report_path = destination / ARTIFACT_NAMES["model_report"]
    transform_path.write_text(_transform_source(spec), encoding="utf-8")
    transformed.dataframe.to_csv(dataset_path, index=False)
    max_drop = float(bundle.claims[0].parameters["max_drop"])
    contract_path.write_text(_contract_source(max_drop), encoding="utf-8")

    before = reference[TARGET_FEATURE]
    after = transformed.dataframe[TARGET_FEATURE]
    unchanged = before.eq(after) | (before.isna() & after.isna())
    report = {
        "discovery_id": str(spec.parameters["discovery_id"]),
        "backend": str(spec.parameters["execution_backend"]),
        "status": "completed",
        "selected_agent": str(spec.parameters["agent_id"]),
        "winning_spec": spec.to_dict(),
        "winning_evidence": evidence.to_dict(),
        "agents": list(discovery["agents"]),
        "verification_report": discovery["verification_report"],
        "model": {
            "type": "sklearn.linear_model.LogisticRegression",
            "features": [TARGET_FEATURE],
            "split": "stratified_holdout",
            "test_size": MODEL_TEST_SIZE,
            "random_state": MODEL_RANDOM_STATE,
        },
        "metric": "roc_auc",
        "baseline_auc": baseline,
        "candidate_auc": candidate,
        "auc_drop": baseline - candidate,
        "max_auc_drop": max_drop,
        "rows": len(reference),
        "affected_rows": int((~unchanged).sum()),
        "affected_fraction": transformed.affected_fraction,
        "invariants": invariants,
        "source_dataset": reference_path.name,
        "source_dataset_sha256": _sha256(reference_path),
        "ghost_dataset_sha256": _sha256(dataset_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    validate_credit_artifacts(reference_path, destination)
    return report


def validate_credit_artifacts(
    reference_path: Path, artifact_dir: Path
) -> dict[str, Any]:
    """Independently audit a published four-file Ghost delivery."""
    observed_names = {path.name for path in artifact_dir.iterdir() if path.is_file()}
    if observed_names != set(ARTIFACT_NAMES.values()):
        raise ValueError("a Ghost delivery must contain exactly four named artifacts")

    report = json.loads(
        (artifact_dir / ARTIFACT_NAMES["model_report"]).read_text(encoding="utf-8")
    )
    spec = VerificationSpec.from_dict(report["winning_spec"])
    evidence = ExecutionEvidence.from_dict(report["winning_evidence"])
    reference = load_credit_data(reference_path)
    ghost_path = artifact_dir / ARTIFACT_NAMES["degraded_dataset"]
    ghost = load_credit_data(ghost_path)
    if list(ghost.columns) != list(reference.columns) or len(ghost) != len(reference):
        raise ValueError("Ghost dataset does not preserve schema and row count")
    if not ghost[TARGET_COLUMN].equals(reference[TARGET_COLUMN]):
        raise ValueError("Ghost dataset changed the target labels")
    invariants = dict(credit_invariants(reference, ghost))
    if not all(invariants.values()) or invariants != report["invariants"]:
        raise ValueError("Ghost dataset does not satisfy the reported invariants")
    if _sha256(reference_path) != report["source_dataset_sha256"]:
        raise ValueError("source dataset hash does not match the report")
    if _sha256(ghost_path) != report["ghost_dataset_sha256"]:
        raise ValueError("Ghost dataset hash does not match the report")

    scorer = fitted_credit_model_score(reference)
    baseline = scorer(reference)
    candidate = scorer(ghost)
    observed = evidence.observations["model_metric"]
    expected_numbers = (
        (baseline, report["baseline_auc"]),
        (candidate, report["candidate_auc"]),
        (baseline - candidate, report["auc_drop"]),
        (baseline, observed["baseline"]),
        (candidate, observed["candidate"]),
    )
    if any(not isclose(float(left), float(right), abs_tol=1e-12) for left, right in expected_numbers):
        raise ValueError("independent AUC recomputation does not match the report")

    with tempfile.TemporaryDirectory() as temp_dir:
        reproduced_path = Path(temp_dir) / "reproduced.csv"
        transform = subprocess.run(
            [
                sys.executable,
                str(artifact_dir / ARTIFACT_NAMES["transform_code"]),
                str(reference_path),
                str(reproduced_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if transform.returncode != 0:
            raise ValueError(f"transform artifact failed: {transform.stderr.strip()}")
        reproduced = load_credit_data(reproduced_path)
        try:
            pd.testing.assert_frame_equal(ghost, reproduced)
        except AssertionError as exc:
            raise ValueError("transform artifact does not reproduce the Ghost dataset") from exc

    contract_path = artifact_dir / ARTIFACT_NAMES["regression_contract"]
    clean = subprocess.run(
        [sys.executable, str(contract_path), str(reference_path), str(reference_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    rejected = subprocess.run(
        [sys.executable, str(contract_path), str(reference_path), str(ghost_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if clean.returncode != 0 or rejected.returncode != 1:
        raise ValueError("regression contract must pass reference and reject the Ghost")
    return report
