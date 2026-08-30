"""Table-agnostic loading, profiling, scoring, and invariant checks."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


COMPILABLE_PRIMITIVES = frozenset({"entity_alignment"})
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.3
MAX_CATEGORY_CARDINALITY = 40
DEFAULT_MAX_SPECS = 4


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
    missing = sorted(
        (item for item in columns if item["name"] != label_column),
        key=lambda item: item["missing_rate"],
        reverse=True,
    )
    return {
        "label_column": label_column,
        "row_count": int(len(dataframe)),
        "columns": columns,
        "inspected_columns": [str(name) for name in dataframe.columns],
        "correlations_with_label": correlations,
        "ranked_features": ranked,
        "missing_ranked": [item["name"] for item in missing if item["missing_rate"] > 0],
        "operator_library": sorted(COMPILABLE_PRIMITIVES),
        "planner": "table_profile",
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _entity_alignment_spec(
    *,
    verification_id: str,
    claim_id: str,
    feature: str,
    fraction: float,
    seed: int,
    correlation: float,
    missing_rate: float,
    segment: Mapping[str, Any],
) -> dict[str, Any]:
    if segment:
        where = ", ".join(f"{key}={value!r}" for key, value in segment.items())
        hypothesis = (
            f"Permuting {feature} inside {where} keeps schema and marginals while "
            "the frozen model should lose the local association."
        )
        agent_id = f"{feature}|{where}|f{fraction:.2f}"
    else:
        hypothesis = (
            f"{feature} is associated with the label (corr={correlation:.3f}, "
            f"missing={missing_rate:.3f}); attaching valid values to the wrong rows "
            "should pass distribution checks and drop model quality."
        )
        agent_id = f"{feature}|f{fraction:.2f}"
    return {
        "verification_id": verification_id,
        "claim_id": claim_id,
        "experiment_type": "entity_alignment",
        "hypothesis": hypothesis,
        "parameters": {
            "target_feature": feature,
            "segment": dict(segment),
            "mismatch_fraction": fraction,
            "seed": seed,
            "label_correlation": correlation,
            "agent_id": agent_id,
        },
        "expected_invariants": [
            "schema",
            "marginal_distribution",
            "missing_rate",
        ],
        "origin": "sandbox_agent",
    }


def specs_from_profile(
    profile: Mapping[str, Any],
    claim_id: str,
    max_specs: int = DEFAULT_MAX_SPECS,
    seed: int = 7,
    segments: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if max_specs < 1:
        raise ValueError("max_specs must be at least 1")
    ranked = list(profile.get("ranked_features") or [])
    if not ranked:
        raise ValueError("profile has no numeric feature correlated with the label")
    missing_rates = {
        str(item["name"]): float(item.get("missing_rate") or 0.0)
        for item in profile.get("columns") or []
        if isinstance(item, Mapping) and "name" in item
    }
    correlations = {
        str(name): float(value)
        for name, value in (profile.get("correlations_with_label") or {}).items()
    }
    fractions = (0.50, 0.35, 0.65, 0.75)
    payloads: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(feature: str, fraction: float, segment: Mapping[str, Any], spec_seed: int) -> None:
        if len(payloads) >= max_specs:
            return
        key = (feature, round(fraction, 4), tuple(sorted(segment.items())))
        if key in seen:
            return
        seen.add(key)
        payloads.append(
            _entity_alignment_spec(
                verification_id=f"V{len(payloads) + 1:03d}",
                claim_id=claim_id,
                feature=feature,
                fraction=fraction,
                seed=spec_seed,
                correlation=correlations.get(feature, 0.0),
                missing_rate=missing_rates.get(feature, 0.0),
                segment=segment,
            )
        )

    for index, feature in enumerate(ranked[: max(1, max_specs - 1)]):
        add(feature, fractions[index % len(fractions)], {}, seed + index)
    if segments:
        add(ranked[0], 0.50, dict(segments[0]), seed + 20)
    if len(payloads) < min(2, max_specs):
        add(ranked[0], 0.75, {}, seed + 30)
    if len(payloads) < max_specs and len(ranked) == 1:
        for extra_fraction in (0.25, 0.90):
            add(ranked[0], extra_fraction, {}, seed + int(extra_fraction * 100))
    return payloads


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
    return _entity_alignment_spec(
        verification_id=verification_id,
        claim_id=claim_id,
        feature=feature,
        fraction=mismatch_fraction,
        seed=seed,
        correlation=float((profile.get("correlations_with_label") or {}).get(feature, 0.0)),
        missing_rate=0.0,
        segment={},
    )


def choose_segment(dataframe: pd.DataFrame, label_column: str) -> dict[str, Any] | None:
    for name in dataframe.columns:
        if name == label_column:
            continue
        series = dataframe[name]
        nunique = int(series.nunique(dropna=True))
        if nunique < 2 or nunique > 8:
            continue
        counts = series.value_counts(dropna=True)
        if counts.empty or int(counts.iloc[0]) < 8:
            continue
        return {str(name): _jsonable(counts.index[0])}
    return None


def propose_from_table(
    dataframe: pd.DataFrame,
    label_column: str,
    claim_id: str,
    max_specs: int = DEFAULT_MAX_SPECS,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = profile_table(dataframe, label_column)
    segment = choose_segment(dataframe, label_column)
    payloads = specs_from_profile(
        profile,
        claim_id,
        max_specs=max_specs,
        seed=seed,
        segments=None if segment is None else (segment,),
    )
    dropped: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("experiment_type") not in COMPILABLE_PRIMITIVES:
            dropped.append({"reason": "uncompilable_primitive", "payload": payload})
            continue
        if "transform_code" in (payload.get("parameters") or {}):
            dropped.append({"reason": "untrusted_transform", "payload": payload})
            continue
        accepted.append(payload)
    analysis = dict(profile)
    analysis["hypotheses"] = [
        {
            "verification_id": item["verification_id"],
            "target_feature": item["parameters"]["target_feature"],
            "mismatch_fraction": item["parameters"]["mismatch_fraction"],
            "segment": item["parameters"]["segment"],
            "hypothesis": item["hypothesis"],
        }
        for item in accepted
    ]
    analysis["dropped"] = dropped
    analysis["emitted_spec_count"] = len(accepted)
    return accepted, analysis


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


def table_invariants(before: pd.DataFrame, after: pd.DataFrame) -> Mapping[str, bool]:
    schema = tuple(before.columns) == tuple(after.columns)
    missing_rate = before.isna().sum().equals(after.isna().sum())
    marginal = False
    if schema:
        marginal = True
        for column in before.columns:
            before_values = (
                before[column].sort_values(na_position="first").reset_index(drop=True)
            )
            after_values = after[column].sort_values(na_position="first").reset_index(drop=True)
            if not before_values.equals(after_values):
                marginal = False
                break
    return {
        "schema": schema,
        "missing_rate": missing_rate,
        "marginal_distribution": marginal,
    }


def infer_feature_columns(
    dataframe: pd.DataFrame,
    label_column: str,
    allowlist: Sequence[str] | None = None,
) -> list[str]:
    allowed = None if allowlist is None else set(allowlist)
    columns: list[str] = []
    for name in dataframe.columns:
        if name == label_column:
            continue
        if allowed is not None and name not in allowed:
            continue
        series = dataframe[name]
        if pd.api.types.is_numeric_dtype(series):
            columns.append(str(name))
            continue
        if int(series.nunique(dropna=True)) <= MAX_CATEGORY_CARDINALITY:
            columns.append(str(name))
    if not columns:
        raise ValueError("no usable model features besides the label")
    return columns


def _split_feature_types(
    dataframe: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for name in feature_columns:
        series = dataframe[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(str(name))
        else:
            categorical.append(str(name))
    return numeric, categorical


def build_frozen_pipeline(
    dataframe: pd.DataFrame, feature_columns: Sequence[str]
) -> Any:
    numeric, categorical = _split_feature_types(dataframe, feature_columns)
    classifier = LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=1000)
    if not categorical:
        if not numeric:
            raise ValueError("no usable model features besides the label")
        return make_pipeline(SimpleImputer(strategy="median"), classifier)
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    transformers.append(
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical,
        )
    )
    return make_pipeline(
        ColumnTransformer(transformers, remainder="drop"),
        classifier,
    )


def fit_frozen_model(
    reference: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    features = list(feature_columns or infer_feature_columns(reference, label_column))
    labels = encode_label(reference[label_column])
    if labels.nunique(dropna=True) < 2:
        raise ValueError("label column must contain at least two classes")
    positions = np.arange(len(reference))
    train_positions, test_positions = train_test_split(
        positions,
        test_size=MODEL_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=labels.fillna(0),
    )
    pipeline = build_frozen_pipeline(reference, features)
    pipeline.fit(
        reference.iloc[train_positions][features],
        labels.iloc[train_positions],
    )
    return {
        "pipeline": pipeline,
        "test_positions": np.asarray(test_positions),
        "feature_columns": features,
        "label_column": label_column,
        "row_count": int(len(reference)),
    }


def score_frozen_model(payload: Mapping[str, Any], dataframe: pd.DataFrame) -> float:
    expected = int(payload["row_count"])
    if len(dataframe) != expected:
        raise ValueError("candidate dataset must preserve the reference row count")
    features = list(payload["feature_columns"])
    label_column = str(payload["label_column"])
    test_positions = payload["test_positions"]
    probabilities = payload["pipeline"].predict_proba(
        dataframe.iloc[test_positions][features]
    )[:, 1]
    return float(
        roc_auc_score(encode_label(dataframe.iloc[test_positions][label_column]), probabilities)
    )


def frozen_model_score(
    reference: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str] | None = None,
) -> Callable[[pd.DataFrame], float]:
    payload = fit_frozen_model(reference, label_column, feature_columns)

    def score(dataframe: pd.DataFrame) -> float:
        return score_frozen_model(payload, dataframe)

    return score


def dump_frozen_model(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(dict(payload), buffer)
    return buffer.getvalue()


def load_frozen_model(data: bytes) -> dict[str, Any]:
    loaded = joblib.load(io.BytesIO(data))
    if not isinstance(loaded, Mapping):
        raise ValueError("frozen model payload is invalid")
    return dict(loaded)


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
