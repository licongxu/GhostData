"""Dataset-agnostic chart payloads for a Ghost versus its reference table."""

from __future__ import annotations

import json
from math import isfinite
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ghostdata.demo.credit import prepare_credit_demo
from ghostdata.execution.local import default_compiler
from ghostdata.verification import VerificationReport, VerificationSpec


HISTOGRAM_BINS = 16
FEATURE_QUANTILES = 5
QUANTILE_LABELS = ("Lowest", "Q2", "Q3", "Q4", "Highest")


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or not isfinite(value):
        return None
    return float(value)


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    codes, _ = pd.factorize(series, sort=True)
    mapped = pd.Series(codes, index=series.index, dtype=float)
    return mapped.mask(mapped < 0)


def _bin_edges(values: pd.Series) -> np.ndarray:
    observed = _to_numeric(values).dropna().to_numpy(dtype=float)
    low, high = np.quantile(observed, [0.01, 0.99])
    if not isfinite(low) or not isfinite(high) or high <= low:
        high = low + 1.0
    return np.linspace(float(low), float(high), HISTOGRAM_BINS + 1)


def _histogram(values: pd.Series, edges: np.ndarray) -> list[int]:
    clipped = _to_numeric(values).dropna().clip(edges[0], edges[-1]).to_numpy(dtype=float)
    counts, _ = np.histogram(clipped, bins=edges)
    return [int(count) for count in counts]


def _midpoints(edges: np.ndarray) -> list[float]:
    return [float((left + right) / 2) for left, right in zip(edges[:-1], edges[1:])]


def _quantile_bins(values: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    observed = _to_numeric(values).dropna()
    if observed.empty:
        return np.asarray([0.0], dtype=float), ()
    if observed.nunique() < 2:
        return np.asarray([float(observed.iloc[0])], dtype=float), ()
    _, edges = pd.qcut(observed, FEATURE_QUANTILES, retbins=True, duplicates="drop")
    if len(edges) - 1 < 2 and int(observed.nunique()) >= 2:
        unique = np.sort(observed.unique())
        width = float(unique[-1] - unique[-2]) or 1.0
        edges = np.concatenate([unique, [unique[-1] + width]])
        labels = tuple(f"Bin {index + 1}" for index in range(len(unique)))
        return np.asarray(edges, dtype=float), labels
    count = len(edges) - 1
    labels = QUANTILE_LABELS if count == FEATURE_QUANTILES else tuple(
        f"Bin {index + 1}" for index in range(count)
    )
    return np.asarray(edges, dtype=float), labels


def _grouped_values(
    dataframe: pd.DataFrame,
    feature: str,
    edges: np.ndarray,
    labels: tuple[str, ...],
    column: str,
    reducer: str,
) -> list[float | None]:
    feature_values = _to_numeric(dataframe[feature])
    column_values = _to_numeric(dataframe[column])
    valid = feature_values.notna() & column_values.notna()
    assigned = pd.cut(
        feature_values[valid],
        bins=edges,
        labels=list(labels),
        include_lowest=True,
    )
    grouped = column_values[valid].groupby(assigned, observed=False)
    reduced = grouped.median() if reducer == "median" else grouped.mean()
    return [_finite_float(reduced.get(label)) for label in labels]


def _perturbed_feature(
    reference: pd.DataFrame,
    ghost: pd.DataFrame,
    spec: VerificationSpec | None,
) -> str:
    hinted = spec.parameters.get("target_feature") if spec is not None else None
    if isinstance(hinted, str) and hinted.strip():
        if hinted not in reference.columns or hinted not in ghost.columns:
            raise ValueError(f"perturbed feature missing: {hinted}")
        return hinted
    for column in reference.columns:
        if column in ghost.columns and not reference[column].equals(ghost[column]):
            return str(column)
    shared = [
        str(column)
        for column in reference.columns
        if column in ghost.columns and pd.api.types.is_numeric_dtype(reference[column])
    ]
    if shared:
        return shared[0]
    raise ValueError("visuals need a feature column present in both datasets")


def _is_binary(series: pd.Series) -> bool:
    return int(series.nunique(dropna=True)) == 2


def _is_unit_interval(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = {
        float(value)
        for value in pd.to_numeric(series, errors="coerce").dropna().unique()
    }
    return bool(values) and values <= {0.0, 1.0}


def _infer_label(reference: pd.DataFrame, feature: str) -> str | None:
    binaries = [
        str(column)
        for column in reference.columns
        if column != feature and _is_binary(reference[column])
    ]
    if not binaries:
        return None
    for column in binaries:
        if _is_unit_interval(reference[column]):
            return column
    return binaries[-1]


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    numeric_left = _to_numeric(left)
    numeric_right = _to_numeric(right)
    if numeric_left.nunique(dropna=True) < 2 or numeric_right.nunique(dropna=True) < 2:
        return None
    return _finite_float(numeric_left.corr(numeric_right))


def _infer_paired(
    reference: pd.DataFrame,
    ghost: pd.DataFrame,
    feature: str,
    label: str | None,
) -> str | None:
    excluded = {feature, label}
    best_column: str | None = None
    best_score = -1.0
    for column in reference.columns:
        name = str(column)
        if name in excluded or name not in ghost.columns:
            continue
        before = _correlation(reference[feature], reference[column])
        after = _correlation(ghost[feature], ghost[column])
        if before is None:
            continue
        delta = abs(before - (after if after is not None else 0.0))
        score = delta if delta > 0 else abs(before)
        if score > best_score:
            best_score = score
            best_column = name
    return best_column


def _winning_experiment(report: VerificationReport):
    if not report.ghosts:
        return None

    def damage(item: object) -> float:
        value = getattr(item, "measurements", {}).get("degradation")
        if isinstance(value, (int, float)) and isfinite(value):
            return float(value)
        return float("-inf")

    return max(report.ghosts, key=damage)


def _evidence_for_visuals(report: VerificationReport):
    winner = _winning_experiment(report)
    if winner is not None:
        for item in report.evidence:
            if item.verification_id == winner.verification_id:
                return item
    if report.evidence:
        return report.evidence[0]
    return None


def _metric_block(report: VerificationReport) -> dict[str, Any]:
    evidence = _evidence_for_visuals(report)
    observations = evidence.observations if evidence is not None else {}
    metric = observations.get("model_metric", {})
    if not isinstance(metric, Mapping):
        metric = {}
    baseline = _finite_float(metric.get("baseline"))
    candidate = _finite_float(metric.get("candidate"))
    degradation = None
    if baseline is not None and candidate is not None:
        degradation = baseline - candidate
    winner = _winning_experiment(report)
    if winner is not None:
        measured = winner.measurements
        measured_baseline = _finite_float(measured.get("baseline"))
        measured_candidate = _finite_float(measured.get("candidate"))
        measured_drop = _finite_float(measured.get("degradation"))
        if measured_baseline is not None:
            baseline = measured_baseline
        if measured_candidate is not None:
            candidate = measured_candidate
        if measured_drop is not None:
            degradation = measured_drop
    return {
        "name": str(metric.get("name") or "metric"),
        "baseline": baseline,
        "candidate": candidate,
        "degradation": degradation,
        "affected_fraction": _finite_float(observations.get("affected_fraction")) or 0.0,
    }


def _continuous(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and int(series.nunique(dropna=True)) > FEATURE_QUANTILES


def _marginal_chart(reference: pd.DataFrame, ghost: pd.DataFrame, feature: str) -> dict[str, Any]:
    if _continuous(reference[feature]):
        edges = _bin_edges(reference[feature])
        return {
            "id": "marginal",
            "kind": "overlay_histogram",
            "title": "What existing tests see",
            "caption": "The value distribution is unchanged, so schema and histogram checks still pass.",
            "x_label": "Feature value",
            "y_label": "Rows",
            "bin_midpoints": _midpoints(edges),
            "reference": _histogram(reference[feature], edges),
            "ghost": _histogram(ghost[feature], edges),
        }
    categories = list(
        reference[feature].astype("string").fillna("(missing)").value_counts().index[:12]
    )
    ref_counts = reference[feature].astype("string").fillna("(missing)").value_counts()
    ghost_counts = ghost[feature].astype("string").fillna("(missing)").value_counts()
    return {
        "id": "marginal",
        "kind": "grouped_bars",
        "title": "What existing tests see",
        "caption": "The value distribution is unchanged, so schema and histogram checks still pass.",
        "x_label": "Feature value",
        "y_label": "Rows",
        "y_format": "number",
        "labels": [str(item) for item in categories],
        "reference": [int(ref_counts.get(item, 0)) for item in categories],
        "ghost": [int(ghost_counts.get(item, 0)) for item in categories],
    }


def _relationship_chart(
    *,
    chart_id: str,
    title: str,
    caption: str,
    y_label: str,
    reference: pd.DataFrame,
    ghost: pd.DataFrame,
    feature: str,
    column: str,
    reducer: str,
    y_format: str,
) -> dict[str, Any] | None:
    edges, labels = _quantile_bins(reference[feature])
    if len(labels) < 2:
        return None
    return {
        "id": chart_id,
        "kind": "grouped_bars",
        "title": title,
        "caption": caption,
        "x_label": "Feature quantile",
        "y_label": y_label,
        "y_format": y_format,
        "labels": list(labels),
        "reference": _grouped_values(reference, feature, edges, labels, column, reducer),
        "ghost": _grouped_values(ghost, feature, edges, labels, column, reducer),
    }


def build_visuals(
    reference: pd.DataFrame,
    ghost: pd.DataFrame,
    report: VerificationReport,
    spec: VerificationSpec | None = None,
) -> dict[str, Any]:
    feature = _perturbed_feature(reference, ghost, spec)
    evidence = _evidence_for_visuals(report)
    observations = evidence.observations if evidence is not None else {}
    invariants = observations.get("invariants", {})
    if not isinstance(invariants, Mapping):
        invariants = {}
    label = _infer_label(reference, feature)
    paired = _infer_paired(reference, ghost, feature, label)
    charts: list[dict[str, Any]] = [_marginal_chart(reference, ghost, feature)]
    if label is not None:
        label_chart = _relationship_chart(
            chart_id="label",
            title="What the model depended on",
            caption="After values move to the wrong rows, the outcome pattern flattens.",
            y_label="Outcome",
            reference=reference,
            ghost=ghost,
            feature=feature,
            column=label,
            reducer="mean",
            y_format="percent" if _is_binary(reference[label]) else "number",
        )
        if label_chart is not None:
            charts.append(label_chart)
    if paired is not None:
        paired_chart = _relationship_chart(
            chart_id="paired",
            title="The pairing broke",
            caption="The numbers are still valid. They now belong to the wrong rows.",
            y_label="Related feature",
            reference=reference,
            ghost=ghost,
            feature=feature,
            column=paired,
            reducer="median",
            y_format="number",
        )
        if paired_chart is not None:
            charts.append(paired_chart)
    payload = {
        "headline": "Same values. Different relationships.",
        "verdict": report.verdict,
        "perturbed_feature": feature,
        "label_column": label,
        "paired_column": paired,
        "invariants": {str(name): bool(passed) for name, passed in invariants.items()},
        "metric": _metric_block(report),
        "charts": charts,
    }
    json.dumps(payload, allow_nan=False)
    return payload


def attach_visuals(
    report: VerificationReport,
    reference: pd.DataFrame | None = None,
    spec: VerificationSpec | None = None,
) -> dict[str, Any]:
    if reference is None or spec is None:
        prepared = prepare_credit_demo()
        reference = prepared.reference
        spec = prepared.specs[0]
    ghost = default_compiler().execute(reference, spec).dataframe
    return {
        **report.to_dict(),
        "visuals": build_visuals(reference, ghost, report, spec),
    }
