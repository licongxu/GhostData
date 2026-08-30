"""Local execution backend for deterministic development and the credit demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ghostdata.bundle import AnalysisBundle
from ghostdata.verification import ExecutionEvidence, VerificationSpec


@dataclass(frozen=True)
class TransformResult:
    dataframe: pd.DataFrame
    affected_fraction: float


Transform = Callable[[pd.DataFrame, VerificationSpec], TransformResult]
InvariantCheck = Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, bool]]
ScoreFunction = Callable[[pd.DataFrame], float]


class ExperimentCompiler:
    def __init__(self) -> None:
        self._transforms: dict[str, Transform] = {}

    def register(self, experiment_type: str, transform: Transform) -> None:
        if experiment_type in self._transforms:
            raise ValueError(f"transform already registered: {experiment_type}")
        self._transforms[experiment_type] = transform

    def execute(
        self, dataframe: pd.DataFrame, spec: VerificationSpec
    ) -> TransformResult:
        try:
            transform = self._transforms[spec.experiment_type]
        except KeyError as exc:
            raise ValueError(f"unsupported experiment type: {spec.experiment_type}") from exc
        return transform(dataframe, spec)


def entity_alignment_transform(
    dataframe: pd.DataFrame, spec: VerificationSpec
) -> TransformResult:
    target_feature = spec.parameters.get("target_feature")
    if not isinstance(target_feature, str) or target_feature not in dataframe.columns:
        raise ValueError(f"missing target feature: {target_feature}")

    eligible = np.ones(len(dataframe), dtype=bool)
    segment = spec.parameters.get("segment", {})
    if not isinstance(segment, Mapping):
        raise ValueError("segment must be a mapping")
    for column, expected in segment.items():
        if column not in dataframe.columns:
            raise ValueError(f"missing segment column: {column}")
        eligible &= dataframe[column].eq(expected).to_numpy()

    mismatch_fraction = float(spec.parameters.get("mismatch_fraction", 1.0))
    if not 0.0 <= mismatch_fraction <= 1.0:
        raise ValueError("mismatch_fraction must be between 0 and 1")

    candidates = np.flatnonzero(eligible)
    count = min(len(candidates), round(len(candidates) * mismatch_fraction))
    if count < 2:
        return TransformResult(dataframe.copy(deep=True), 0.0)

    rng = np.random.default_rng(int(spec.parameters.get("seed", 0)))
    selected = rng.choice(candidates, size=count, replace=False)
    transformed = dataframe.copy(deep=True)
    target_position = dataframe.columns.get_loc(target_feature)
    if not isinstance(target_position, int):
        raise ValueError(f"target feature must be unique: {target_feature}")
    original = transformed.iloc[selected, target_position].to_numpy(copy=True)
    transformed.iloc[selected, target_position] = np.roll(original, 1)

    before = dataframe[target_feature]
    after = transformed[target_feature]
    unchanged = before.eq(after) | (before.isna() & after.isna())
    changed = (~unchanged).sum()
    return TransformResult(transformed, float(changed / len(dataframe)))


def default_compiler() -> ExperimentCompiler:
    compiler = ExperimentCompiler()
    compiler.register("entity_alignment", entity_alignment_transform)
    return compiler


class LocalVerificationRunner:
    def __init__(
        self,
        reference: pd.DataFrame,
        compiler: ExperimentCompiler,
        check: InvariantCheck,
        score: ScoreFunction,
        metric: str,
    ) -> None:
        if not metric.strip():
            raise ValueError("metric must be a non-empty string")
        self._reference = reference.copy(deep=True)
        self._compiler = compiler
        self._check = check
        self._score = score
        self._metric = metric
        self._baseline = float(score(self._reference))

    def run(
        self, bundle: AnalysisBundle, spec: VerificationSpec
    ) -> ExecutionEvidence:
        transformed = self._compiler.execute(self._reference, spec)
        invariants = dict(self._check(self._reference, transformed.dataframe))
        if any(not isinstance(name, str) or type(passed) is not bool for name, passed in invariants.items()):
            raise ValueError("invariant checks must map names to booleans")
        candidate = float(self._score(transformed.dataframe))
        return ExecutionEvidence(
            bundle_id=bundle.bundle_id,
            verification_id=spec.verification_id,
            claim_id=spec.claim_id,
            experiment_type=spec.experiment_type,
            status="completed",
            observations={
                "invariants": invariants,
                "model_metric": {
                    "name": self._metric,
                    "baseline": self._baseline,
                    "candidate": candidate,
                },
                "affected_fraction": transformed.affected_fraction,
            },
        )

