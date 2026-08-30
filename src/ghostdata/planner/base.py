"""Trust boundary for components that propose falsification experiments."""

from __future__ import annotations

from typing import Protocol, Sequence

from ghostdata.bundle import AnalysisBundle, Claim
from ghostdata.verification import VerificationSpec


class VerificationPlanner(Protocol):
    def propose(
        self, bundle: AnalysisBundle, claims: Sequence[Claim]
    ) -> Sequence[VerificationSpec]: ...

