"""Structured-output proposal agent. It never writes transforms or Ghost CSVs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from ghostdata.bundle import AnalysisBundle, Claim
from ghostdata.tabular import profile_table, spec_from_profile
from ghostdata.verification import VerificationSpec


class StructuredSpecPlanner:
    """Turn a table profile into a VerificationSpec. No transform code."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_column: str,
        analysis: Mapping[str, Any] | None = None,
    ) -> None:
        if label_column not in dataframe.columns:
            raise ValueError(f"label column missing: {label_column}")
        self._dataframe = dataframe
        self._label_column = label_column
        self.last_analysis: dict[str, Any] | None = (
            dict(analysis) if analysis is not None else None
        )

    def propose(
        self, bundle: AnalysisBundle, claims: Sequence[Claim]
    ) -> list[VerificationSpec]:
        bundle_claim_ids = {claim.claim_id for claim in bundle.claims}
        profile = self.last_analysis or profile_table(self._dataframe, self._label_column)
        self.last_analysis = dict(profile)
        specs: list[VerificationSpec] = []
        for claim in claims:
            if claim.claim_id not in bundle_claim_ids:
                raise ValueError(f"claim is not part of bundle: {claim.claim_id}")
            if claim.evaluator != "model_metric_preservation":
                continue
            payload = spec_from_profile(profile, claim.claim_id, f"V{len(specs) + 1:03d}")
            specs.append(VerificationSpec.from_dict(payload))
        return specs
