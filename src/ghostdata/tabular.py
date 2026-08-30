"""Table-agnostic loading, profiling, scoring, and invariant checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from sklearn.metrics import roc_auc_score


def load_table(data_path: Path | str, label_column: str) -> pd.DataFrame:
    path = Path(data_path).resolve()
    dataframe = pd.read_csv(path)
    if label_column not in dataframe.columns:
        raise ValueError(f"label column missing: {label_column}")
    if dataframe.empty:
        raise ValueError("dataset must contain at least one row")
    if dataframe[label_column].nunique(dropna=True) < 2:
        raise ValueError("label column must contain at least two classes")
    return dataframe


def encode_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        values = set(numeric.dropna().unique().tolist())
        if values <= {0, 1} or values <= {0.0, 1.0}:
            return numeric.astype(float)
    codes, _ = pd.factorize(series, sort=True)
    mapped = pd.Series(codes, index=series.index, dtype=float)
    return mapped.mask(mapped < 0)


def profile_table(dataframe: pd.DataFrame, label_column: str) -> dict[str, Any]:
    if label_column not in dataframe.columns:
        raise ValueError(f"label column missing: {label_column}")
    label = encode_label(dataframe[label_column])
    columns: list[dict[str, Any]] = []
    correlations: dict[str, float] = {}
    for name in dataframe.columns:
        series = dataframe[name]
        numeric = pd.to_numeric(series, errors="coerce") if name != label_column else None
        entry = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "nunique": int(series.nunique(dropna=True)),
            "numeric": bool(pd.api.types.is_numeric_dtype(series)),
        }
        columns.append(entry)
        if name == label_column or numeric is None:
            continue
        if numeric.nunique(dropna=True) < 2:
            continue
        aligned = pd.concat([numeric, label], axis=1).dropna()
        if len(aligned) < 4:
            continue
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if corr == corr:
            correlations[str(name)] = corr
    ranked = sorted(correlations, key=lambda name: abs(correlations[name]), reverse=True)
    return {
        "label_column": label_column,
        "row_count": int(len(dataframe)),
        "columns": columns,
        "inspected_columns": [str(name) for name in dataframe.columns],
        "correlations_with_label": correlations,
        "ranked_features": ranked,
        "operator_library": ["entity_alignment"],
    }


def spec_from_profile(
    profile: Mapping[str, Any],
    claim_id: str,
    verification_id: str = "V001",
    mismatch_fraction: float = 0.25,
    seed: int = 7,
) -> dict[str, Any]:
    ranked = list(profile.get("ranked_features") or [])
    if not ranked:
        raise ValueError("profile has no numeric feature correlated with the label")
    feature = str(ranked[0])
    corr = float((profile.get("correlations_with_label") or {}).get(feature, 0.0))
    return {
        "verification_id": verification_id,
        "claim_id": claim_id,
        "experiment_type": "entity_alignment",
        "hypothesis": (
            "Valid feature values attached to the wrong rows still pass distribution "
            "checks while the label relationship breaks."
        ),
        "parameters": {
            "target_feature": feature,
            "segment": {},
            "mismatch_fraction": mismatch_fraction,
            "seed": seed,
            "label_correlation": corr,
        },
        "expected_invariants": [
            "schema",
            "marginal_distribution",
            "missing_rate",
        ],
        "origin": "sandbox_agent",
    }


def feature_invariants(target_feature: str) -> Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, bool]]:
    def check(before: pd.DataFrame, after: pd.DataFrame) -> Mapping[str, bool]:
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

    return check


def feature_score(reference: pd.DataFrame, label_column: str, target_feature: str):
    labels = encode_label(reference[label_column])
    values = pd.to_numeric(reference[target_feature], errors="coerce")
    median = float(values.median()) if values.notna().any() else 0.0
    filled = values.fillna(median)
    baseline_positive = float(roc_auc_score(labels, filled))
    sign = 1.0 if baseline_positive >= 0.5 else -1.0

    def score(dataframe: pd.DataFrame) -> float:
        if len(dataframe) != len(reference):
            raise ValueError("candidate dataset must preserve the reference row count")
        candidate = pd.to_numeric(dataframe[target_feature], errors="coerce").fillna(median)
        return float(roc_auc_score(encode_label(dataframe[label_column]), sign * candidate))

    return score
