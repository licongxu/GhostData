"""Generic labelled-table demo. Credit is one fixture, not the product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.planner.agent import StructuredSpecPlanner
from ghostdata.tabular import (
    DEFAULT_MAX_SPECS,
    dump_frozen_model,
    fit_frozen_model,
    frozen_model_score,
    load_table,
    table_invariants,
)
from ghostdata.verification import ExecutionEvidence, VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKER_PATH = PROJECT_ROOT / "demo" / "pipeline" / "worker.py"
PROPOSER_PATH = PROJECT_ROOT / "demo" / "pipeline" / "proposer.py"
PROMOTE_PATH = PROJECT_ROOT / "demo" / "pipeline" / "promote.py"
MAX_LIVE_WORLDS = 6
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
    include_package: bool = True,
    include_dataset: bool = True,
) -> dict[str, bytes]:
    files = dict(extra)
    if include_dataset:
        files["dataset.csv"] = data_path.read_bytes()
    if include_package:
        package_root = PROJECT_ROOT / "src" / "ghostdata"
        for source_path in package_root.rglob("*.py"):
            remote = (Path("src") / source_path.relative_to(PROJECT_ROOT / "src")).as_posix()
            files[remote] = source_path.read_bytes()
    return files


def build_local_runner(reference, label_column: str) -> LocalVerificationRunner:
    return LocalVerificationRunner(
        reference,
        default_compiler(),
        table_invariants,
        frozen_model_score(reference, label_column),
        metric="roc_auc",
    )


def build_table_bundle(
    reference,
    label_column: str,
    bundle_id: str,
    task: str,
    max_specs: int = DEFAULT_MAX_SPECS,
) -> tuple[AnalysisBundle, StructuredSpecPlanner, float]:
    planner = StructuredSpecPlanner(reference, label_column, max_specs=max_specs)
    scorer = frozen_model_score(reference, label_column)
    baseline = scorer(reference)
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


def _task_payload(
    label_column: str,
    *,
    dataset: str,
    claim_id: str = "C001",
    max_specs: int = DEFAULT_MAX_SPECS,
    model_path: str | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "label_column": label_column,
        "dataset": dataset,
        "claim_id": claim_id,
        "max_specs": max_specs,
    }
    if model_path:
        payload["model_path"] = model_path
    return json.dumps(payload).encode()


def build_executor_job(
    data_path: Path,
    label_column: str,
    settings: object | None = None,
    include_dataset: bool | None = None,
):
    from ghostdata.execution.daytona import (
        PROPOSER_CHART_CODE,
        DaytonaJob,
        DaytonaSettings,
        sandbox_pythonpath,
        uses_baked_package,
    )

    resolved = settings if isinstance(settings, DaytonaSettings) else DaytonaSettings()
    baked = uses_baked_package(resolved)
    volume = bool(resolved.volume_name)
    upload_dataset = (not volume) if include_dataset is None else include_dataset
    dataset = "/data/dataset.csv" if volume else "dataset.csv"
    model_path = "/data/model.joblib" if volume else None
    files = package_sandbox_files(
        data_path,
        {
            "worker.py": WORKER_PATH.read_bytes(),
            "task.json": _task_payload(
                label_column, dataset=dataset, model_path=model_path
            ),
        },
        include_package=not baked,
        include_dataset=upload_dataset,
    )
    volume_files: dict[str, bytes] = {}
    if volume:
        volume_files = {
            "dataset.csv": data_path.read_bytes(),
            "model.joblib": dump_frozen_model(
                fit_frozen_model(load_table(data_path, label_column), label_column)
            ),
        }
    return DaytonaJob(
        command=f"PYTHONPATH={sandbox_pythonpath(resolved)} python worker.py",
        files=files,
        evidence_path="evidence.json",
        role="executor",
        extra_labels={"stage": "execute"},
        network_block_all=True,
        volume_files=volume_files,
        code_run=PROPOSER_CHART_CODE,
    )


def build_promote_job(
    data_path: Path,
    label_column: str,
    discovery: dict[str, object],
    settings: object | None = None,
):
    from ghostdata.demo.artifacts import ARTIFACT_NAMES
    from ghostdata.execution.daytona import (
        PROPOSER_CHART_CODE,
        DaytonaJob,
        DaytonaSettings,
        sandbox_pythonpath,
        uses_baked_package,
    )

    resolved = settings if isinstance(settings, DaytonaSettings) else DaytonaSettings()
    baked = uses_baked_package(resolved)
    volume = bool(resolved.volume_name)
    dataset = "/data/dataset.csv" if volume else "dataset.csv"
    files = package_sandbox_files(
        data_path,
        {
            "worker.py": PROMOTE_PATH.read_bytes(),
            "task.json": _task_payload(
                label_column,
                dataset=dataset,
                model_path="/data/model.joblib" if volume else None,
            ),
            "discovery_report.json": json.dumps(
                discovery, sort_keys=True, allow_nan=False, default=str
            ).encode(),
        },
        include_package=not baked,
        include_dataset=not volume,
    )
    volume_files: dict[str, bytes] = {}
    if volume:
        volume_files = {
            "dataset.csv": data_path.read_bytes(),
            "model.joblib": dump_frozen_model(
                fit_frozen_model(load_table(data_path, label_column), label_column)
            ),
        }
    return DaytonaJob(
        command=f"PYTHONPATH={sandbox_pythonpath(resolved)} python worker.py",
        files=files,
        evidence_path="promotion_evidence.json",
        download_paths={
            role: f"outputs/{filename}" for role, filename in ARTIFACT_NAMES.items()
        },
        role="promoter",
        extra_labels={"stage": "promote"},
        network_block_all=True,
        volume_files=volume_files,
        code_run=PROPOSER_CHART_CODE,
    )


def select_winner(
    report: VerificationReport, specs: list[VerificationSpec] | tuple[VerificationSpec, ...]
) -> tuple[VerificationSpec, ExecutionEvidence] | None:
    specs_by_id = {spec.verification_id: spec for spec in specs}
    evidence_by_id = {item.verification_id: item for item in report.evidence}
    eligible = []
    for ghost in report.ghosts:
        degradation = ghost.measurements.get("degradation")
        if isinstance(degradation, (int, float)) and isfinite(degradation):
            eligible.append((float(degradation), ghost.verification_id))
    if not eligible:
        return None
    _, verification_id = max(eligible, key=lambda item: item[0])
    return specs_by_id[verification_id], evidence_by_id[verification_id]


def run_table_demo(
    data_path: Path | str,
    label_column: str,
    backend: Literal["local", "daytona"] = "local",
    daytona_settings: object | None = None,
    bundle_id: str = "table-preprocessing-demo",
    max_specs: int = DEFAULT_MAX_SPECS,
) -> tuple[VerificationReport, VerificationSpec, dict[str, Any]]:
    path = Path(data_path).resolve()
    reference = load_table(path, label_column)
    bundle, planner, _baseline = build_table_bundle(
        reference,
        label_column,
        bundle_id,
        "Verify an agent-generated preprocessing change.",
        max_specs=max_specs,
    )
    if backend == "local":
        specs = planner.propose(bundle, bundle.claims)
        runner = build_local_runner(reference, label_column)
        analysis = dict(planner.last_analysis or {})
    elif backend == "daytona":
        from ghostdata.execution.daytona import DaytonaSettings, uses_baked_package

        load_dotenv(PROJECT_ROOT / ".env")
        settings = daytona_settings or DaytonaSettings()
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
        baked = uses_baked_package(settings)
        volume = bool(getattr(settings, "volume_name", None))
        proposer_files = package_sandbox_files(
            path,
            {
                "proposer.py": PROPOSER_PATH.read_bytes(),
                "task.json": _task_payload(
                    label_column,
                    dataset="/data/dataset.csv" if volume else "dataset.csv",
                    max_specs=max_specs,
                ),
            },
            include_package=not baked,
            include_dataset=not volume,
        )
        volume_files = None
        if volume:
            volume_files = {
                "dataset.csv": path.read_bytes(),
                "model.joblib": dump_frozen_model(
                    fit_frozen_model(reference, label_column)
                ),
            }
        specs, analysis = proposal_cls(settings).propose(
            bundle,
            proposer_files,
            label_column,
            "C001",
            volume_files=volume_files,
        )
        planner = FrozenSpecsPlanner(tuple(specs), analysis)
        runner = runner_cls(
            lambda _bundle, _spec: build_executor_job(path, label_column, settings),
            settings=settings,
        )
    else:
        raise ValueError(f"unsupported demo backend: {backend}")

    if not specs:
        raise RuntimeError("planner emitted no compilable VerificationSpec")
    workers = min(max(len(specs), 1), MAX_LIVE_WORLDS)
    evaluators = EvaluatorRegistry((ModelMetricPreservationEvaluator(),))
    orchestrator = VerificationOrchestrator(runner, evaluators, max_workers=workers)
    report = orchestrator.verify(bundle, BundleClaimExtractor(), planner)
    winner = select_winner(report, specs)
    spec = winner[0] if winner is not None else specs[0]
    analysis = dict(analysis)
    analysis["executed_spec_count"] = len(specs)
    analysis["winning_verification_id"] = spec.verification_id
    return report, spec, analysis
