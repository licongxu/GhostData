"""Structured-output proposal agent. It never writes transforms or Ghost CSVs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from ghostdata.bundle import AnalysisBundle, Claim
from ghostdata.tabular import DEFAULT_MAX_SPECS, propose_from_table
from ghostdata.verification import VerificationSpec


class StructuredSpecPlanner:
    """Inspect a table and emit compilable VerificationSpecs. No transform code."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_column: str,
        analysis: Mapping[str, Any] | None = None,
        max_specs: int = DEFAULT_MAX_SPECS,
    ) -> None:
        if label_column not in dataframe.columns:
            raise ValueError(f"label column missing: {label_column}")
        self._dataframe = dataframe
        self._label_column = label_column
        self._max_specs = max_specs
        self.last_analysis: dict[str, Any] | None = (
            dict(analysis) if analysis is not None else None
        )

    def propose(
        self, bundle: AnalysisBundle, claims: Sequence[Claim]
    ) -> list[VerificationSpec]:
        bundle_claim_ids = {claim.claim_id for claim in bundle.claims}
        payloads, analysis = propose_from_table(
            self._dataframe,
            self._label_column,
            next(
                (claim.claim_id for claim in claims if claim.claim_id in bundle_claim_ids),
                "C001",
            ),
            max_specs=self._max_specs,
        )
        if self.last_analysis is None:
            self.last_analysis = analysis
        else:
            self.last_analysis.update(analysis)
        specs: list[VerificationSpec] = []
        for claim in claims:
            if claim.claim_id not in bundle_claim_ids:
                raise ValueError(f"claim is not part of bundle: {claim.claim_id}")
            if claim.evaluator != "model_metric_preservation":
                continue
            for payload in payloads:
                item = dict(payload)
                item["claim_id"] = claim.claim_id
                if specs:
                    item["verification_id"] = f"V{len(specs) + 1:03d}"
                specs.append(VerificationSpec.from_dict(item))
        return specs
