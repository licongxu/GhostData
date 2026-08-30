"""Known production-failure experiments used by the credit demo."""

from __future__ import annotations

from typing import Sequence

from ghostdata.bundle import AnalysisBundle, Claim
from ghostdata.verification import VerificationSpec


class KnownFailurePlanner:
    """P0 planner backed by a deterministic library, not an AI judge."""

    def __init__(
        self,
        target_feature: str,
        mismatch_fraction: float = 0.25,
        seed: int = 7,
    ) -> None:
        if not target_feature.strip():
            raise ValueError("target_feature must be a non-empty string")
        self._target_feature = target_feature
        self._mismatch_fraction = mismatch_fraction
        self._seed = seed

    def propose(
        self, bundle: AnalysisBundle, claims: Sequence[Claim]
    ) -> list[VerificationSpec]:
        bundle_claim_ids = {claim.claim_id for claim in bundle.claims}
        specs: list[VerificationSpec] = []
        for claim in claims:
            if claim.claim_id not in bundle_claim_ids:
                raise ValueError(f"claim is not part of bundle: {claim.claim_id}")
            if claim.evaluator != "model_metric_preservation":
                continue
            specs.append(
                VerificationSpec(
                    verification_id=f"V{len(specs) + 1:03d}",
                    claim_id=claim.claim_id,
                    experiment_type="entity_alignment",
                    hypothesis=(
                        f"Valid {self._target_feature} values become attached to the wrong "
                        "entities while the agent's stated invariants remain unchanged."
                    ),
                    parameters={
                        "target_feature": self._target_feature,
                        "segment": {},
                        "mismatch_fraction": self._mismatch_fraction,
                        "seed": self._seed,
                    },
                    expected_invariants=(
                        "schema",
                        "marginal_distribution",
                        "missing_rate",
                    ),
                )
            )
        return specs

