"""Generic labelled-table demo. Credit is one fixture, not the product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.planner.agent import StructuredSpecPlanner
from ghostdata.tabular import (
    feature_invariants,
    feature_score,
    load_table,
    profile_table,
    spec_from_profile,
)
from ghostdata.verification import VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = PROJECT_ROOT / "demo" / "pipeline" / "worker.py"
PROPOSER_PATH = PROJECT_ROOT / "demo" / "pipeline" / "proposer.py"
DaytonaVerificationRunner = None
DaytonaProposalRunner = None


@dataclass(frozen=True)
class FrozenSpecsPlanner:
    specs: tuple[VerificationSpec, ...]
    last_analysis: dict[str, Any] | None = None

    def propose(self, bundle: AnalysisBundle, claims: object) -> list[VerificationSpec]:
        del bundle, claims
        return list(self.specs)


def package_sandbox_files(
    data_path: Path,
    extra: dict[str, bytes],
) -> dict[str, bytes]:
    files = dict(extra)
    files["dataset.csv"] = data_path.read_bytes()
    package_root = PROJECT_ROOT / "src" / "ghostdata"
    for source_path in package_root.rglob("*.py"):
        remote = (Path("src") / source_path.relative_to(PROJECT_ROOT / "src")).as_posix()
        files[remote] = source_path.read_bytes()
    return files


def build_table_bundle(
    reference,
    label_column: str,
    bundle_id: str,
    task: str,
) -> tuple[AnalysisBundle, StructuredSpecPlanner, float]:
    profile = profile_table(reference, label_column)
    planner = StructuredSpecPlanner(reference, label_column, profile)
    payload = spec_from_profile(profile, "C001")
    feature = str(payload["parameters"]["target_feature"])
    baseline = feature_score(reference, label_column, feature)(reference)
    claim = Claim(
        claim_id="C001",
        assertion="The preprocessing change preserves model quality.",
        evaluator="model_metric_preservation",
        parameters={
            "metric": "roc_auc",
            "max_drop": 0.0,
            "direction": "higher_is_better",
        },
        supplied_evidence={"roc_auc": baseline},
    )
    bundle = AnalysisBundle(
        bundle_id=bundle_id,
        task=task,
        inputs={"dataset": "dataset.csv"},
        agent_output=AgentOutput(metrics={"roc_auc": baseline}),
        claims=(claim,),
    )
    return bundle, planner, baseline


def build_executor_job(data_path: Path, label_column: str):
    from ghostdata.execution.daytona import DaytonaJob

    files = package_sandbox_files(
        data_path,
        {
            "worker.py": WORKER_PATH.read_bytes(),
            "task.json": json.dumps(
                {"label_column": label_column, "dataset": "dataset.csv"}
            ).encode(),
        },
    )
    return DaytonaJob(
        command="PYTHONPATH=src python worker.py",
        files=files,
        evidence_path="evidence.json",
        role="executor",
    )


def run_table_demo(
    data_path: Path | str,
    label_column: str,
    backend: Literal["local", "daytona"] = "local",
    daytona_settings: object | None = None,
    bundle_id: str = "table-preprocessing-demo",
) -> tuple[VerificationReport, VerificationSpec, dict[str, Any]]:
    path = Path(data_path).resolve()
    reference = load_table(path, label_column)
    bundle, planner, _baseline = build_table_bundle(
        reference,
        label_column,
        bundle_id,
        "Verify an agent-generated preprocessing change.",
    )
    if backend == "local":
        spec = planner.propose(bundle, bundle.claims)[0]
        feature = str(spec.parameters["target_feature"])
        runner = LocalVerificationRunner(
            reference,
            default_compiler(),
            feature_invariants(feature),
            feature_score(reference, label_column, feature),
            metric="roc_auc",
        )
        analysis = dict(planner.last_analysis or {})
    elif backend == "daytona":
        from ghostdata.execution.daytona import DaytonaSettings

        load_dotenv(PROJECT_ROOT / ".env")
        settings = daytona_settings or DaytonaSettings(volume_name="ghostdata-data")
        proposal_cls = DaytonaProposalRunner
        if proposal_cls is None:
            from ghostdata.execution.daytona import (
                DaytonaProposalRunner as proposal_cls,
            )
        runner_cls = DaytonaVerificationRunner
        if runner_cls is None:
            from ghostdata.execution.daytona import (
                DaytonaVerificationRunner as runner_cls,
            )
        proposer_files = package_sandbox_files(
            path,
            {
                "proposer.py": PROPOSER_PATH.read_bytes(),
                "task.json": json.dumps(
                    {
                        "label_column": label_column,
                        "dataset": "dataset.csv",
                        "claim_id": "C001",
                    }
                ).encode(),
            },
        )
        specs, analysis = proposal_cls(settings).propose(
            bundle, proposer_files, label_column, "C001"
        )
        planner = FrozenSpecsPlanner(tuple(specs), analysis)
        spec = specs[0]
        feature = str(spec.parameters["target_feature"])
        runner = runner_cls(
            lambda _bundle, _spec: build_executor_job(path, label_column),
            settings=settings,
        )
    else:
        raise ValueError(f"unsupported demo backend: {backend}")

    evaluators = EvaluatorRegistry((ModelMetricPreservationEvaluator(),))
    orchestrator = VerificationOrchestrator(runner, evaluators, max_workers=1)
    report = orchestrator.verify(bundle, BundleClaimExtractor(), planner)
    return report, spec, dict(analysis)
