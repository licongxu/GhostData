"""Execution substrate contract; execution records facts and never decides verdicts."""

from __future__ import annotations

from typing import Protocol

from ghostdata.bundle import AnalysisBundle
from ghostdata.verification import ExecutionEvidence, VerificationSpec


class VerificationRunner(Protocol):
    def run(
        self, bundle: AnalysisBundle, spec: VerificationSpec
    ) -> ExecutionEvidence: ...

