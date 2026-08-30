"""Multi-sandbox Ghost discovery and promotion. Credit is one fixture."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import pandas as pd
from dotenv import load_dotenv

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.demo.artifacts import (
    ARTIFACT_NAMES,
    build_ghost_artifacts,
    validate_ghost_artifacts,
)
from ghostdata.demo.credit import PROJECT_ROOT, TARGET_COLUMN, TARGET_FEATURE
from ghostdata.demo.table import (
    FrozenSpecsPlanner,
    MAX_LIVE_WORLDS,
    build_executor_job,
    build_local_runner,
    build_promote_job,
    build_table_bundle,
    package_sandbox_files,
    select_winner,
)
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.daytona import (
    DaytonaJob,
    DaytonaProposalRunner,
    DaytonaSettings,
    DaytonaVerificationRunner,
)
from ghostdata.tabular import DEFAULT_MAX_SPECS, load_table
from ghostdata.verification import ExecutionEvidence, VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


FULL_DATA_PATH = PROJECT_ROOT / "data" / "build" / "givemesomecredit.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "discovery"
DISCOVERY_WORKER = PROJECT_ROOT / "demo" / "pipeline" / "worker.py"
ARTIFACT_WORKER = PROJECT_ROOT / "demo" / "pipeline" / "promote.py"


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    mismatch_fraction: float
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        if not 0.0 <= self.mismatch_fraction <= 1.0:
            raise ValueError("mismatch_fraction must be between 0 and 1")


DEFAULT_AGENT_PROFILES = (
    AgentProfile("alignment_scout", 0.10),
    AgentProfile("invariant_breaker", 0.25),
    AgentProfile("broad_stress_tester", 0.50),
    AgentProfile("relationship_hunter", 0.75),
)


class CreditDiscoveryPlanner:
    def __init__(
        self,
        profiles: Sequence[AgentProfile],
        discovery_id: str,
        backend: str,
    ) -> None:
        self._profiles = tuple(profiles)
        if not self._profiles:
            raise ValueError("at least one agent profile is required")
        ids = [profile.agent_id for profile in self._profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("agent profile ids must be unique")
        self._discovery_id = discovery_id
        self._backend = backend

    def propose(
        self, bundle: AnalysisBundle, claims: Sequence[Claim]
    ) -> list[VerificationSpec]:
        claim_ids = {claim.claim_id for claim in bundle.claims}
        specs: list[VerificationSpec] = []
        for claim in claims:
            if claim.claim_id not in claim_ids:
                raise ValueError(f"claim is not part of bundle: {claim.claim_id}")
            if claim.evaluator != "model_metric_preservation":
                continue
            for profile in self._profiles:
                specs.append(
                    VerificationSpec(
                        verification_id=f"V{len(specs) + 1:03d}",
                        claim_id=claim.claim_id,
                        experiment_type="entity_alignment",
                        hypothesis=(
                            f"Agent {profile.agent_id} tests whether valid {TARGET_FEATURE} "
                            "values can become attached to the wrong entities while the "
                            "declared invariants still pass."
                        ),
                        parameters={
                            "target_feature": TARGET_FEATURE,
                            "segment": {},
                            "mismatch_fraction": profile.mismatch_fraction,
                            "seed": profile.seed,
                            "agent_id": profile.agent_id,
                            "discovery_id": self._discovery_id,
                            "execution_backend": self._backend,
                        },
                        expected_invariants=(
                            "schema",
                            "marginal_distribution",
                            "missing_rate",
                        ),
                        origin="simulated_agent",
                    )
                )
        return specs


@dataclass(frozen=True)
class PreparedCreditDiscovery:
    data_path: Path
    reference: pd.DataFrame
    bundle: AnalysisBundle
    planner: CreditDiscoveryPlanner
    scorer: Callable[[pd.DataFrame], float]

    @property
    def specs(self) -> tuple[VerificationSpec, ...]:
        return tuple(self.planner.propose(self.bundle, self.bundle.claims))


def prepare_credit_discovery(
    data_path: Path | str,
    profiles: Sequence[AgentProfile],
    discovery_id: str,
    backend: str,
) -> PreparedCreditDiscovery:
    from ghostdata.demo.credit import fitted_credit_model_score, load_credit_data

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", discovery_id):
        raise ValueError("discovery_id contains unsafe characters")
    path = Path(data_path).resolve()
    reference = load_credit_data(path)
    scorer = fitted_credit_model_score(reference)
    baseline = scorer(reference)
    claim = Claim(
        claim_id="C001",
        assertion="The preprocessing change preserves fitted-model ROC AUC.",
        evaluator="model_metric_preservation",
        parameters={
            "metric": "roc_auc",
            "max_drop": 0.0,
            "direction": "higher_is_better",
        },
        supplied_evidence={"roc_auc": baseline},
    )
    bundle = AnalysisBundle(
        bundle_id=f"credit-discovery-{discovery_id}",
        task="Find executable failures in an agent-generated credit data pipeline.",
        inputs={"dataset": "dataset.csv"},
        agent_output=AgentOutput(metrics={"roc_auc": baseline}),
        claims=(claim,),
    )
    return PreparedCreditDiscovery(
        path,
        reference,
        bundle,
        CreditDiscoveryPlanner(profiles, discovery_id, backend),
        scorer,
    )


def _job_files(data_path: Path, worker_path: Path) -> dict[str, bytes]:
    return package_sandbox_files(data_path, {"worker.py": worker_path.read_bytes()})


def build_discovery_job(
    prepared: PreparedCreditDiscovery,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
) -> DaytonaJob:
    return DaytonaJob(
        command="PYTHONPATH=src python worker.py",
        files=_job_files(prepared.data_path, DISCOVERY_WORKER),
    )


def _agent_results(
    report: VerificationReport, specs: Sequence[VerificationSpec]
) -> list[dict[str, object]]:
    verdicts = {
        experiment.verification_id: experiment
        for claim in report.claims
        for experiment in claim.experiments
    }
    return [
        {
            "agent_id": spec.parameters.get("agent_id") or spec.verification_id,
            "verification_id": spec.verification_id,
            "target_feature": spec.parameters.get("target_feature"),
            "mismatch_fraction": spec.parameters.get("mismatch_fraction"),
            "hypothesis": spec.hypothesis,
            "outcome": verdicts[spec.verification_id].outcome,
            "reason": verdicts[spec.verification_id].reason,
            "measurements": dict(verdicts[spec.verification_id].measurements),
        }
        for spec in specs
    ]


def _winner(
    report: VerificationReport, specs: Sequence[VerificationSpec]
) -> tuple[VerificationSpec, ExecutionEvidence]:
    selected = select_winner(report, specs)
    if selected is None:
        raise RuntimeError("discovery completed without a promotable Ghost")
    return selected


def _promote_local(
    data_path: Path,
    label_column: str,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
    evidence: ExecutionEvidence,
    discovery: Mapping[str, object],
    destination: Path,
) -> None:
    build_ghost_artifacts(
        data_path, label_column, bundle, spec, evidence, discovery, destination
    )


def _promote_daytona(
    data_path: Path,
    label_column: str,
    bundle: AnalysisBundle,
    spec: VerificationSpec,
    discovery: Mapping[str, object],
    destination: Path,
    settings: DaytonaSettings | None,
) -> None:
    def sink(
        _bundle: AnalysisBundle,
        promoted_spec: VerificationSpec,
        artifacts: Mapping[str, bytes],
    ) -> Mapping[str, str]:
        del promoted_spec
        if set(artifacts) != set(ARTIFACT_NAMES):
            raise ValueError("promotion sandbox returned an invalid artifact set")
        destination.mkdir(parents=True, exist_ok=True)
        for role, contents in artifacts.items():
            (destination / ARTIFACT_NAMES[role]).write_bytes(contents)
        return dict(ARTIFACT_NAMES)

    runner = DaytonaVerificationRunner(
        lambda _bundle, _spec: build_promote_job(
            data_path, label_column, dict(discovery), settings
        ),
        settings=settings,
        artifact_sink=sink,
    )
    runner.run(bundle, spec)
    validate_ghost_artifacts(data_path, destination, label_column)


def run_table_discovery(
    data_path: Path | str,
    label_column: str,
    backend: Literal["local", "daytona"] = "local",
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    discovery_id: str | None = None,
    max_specs: int = DEFAULT_MAX_SPECS,
    daytona_settings: DaytonaSettings | None = None,
    planner: object | None = None,
) -> dict[str, object]:
    discovery_id = discovery_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", discovery_id):
        raise ValueError("discovery_id contains unsafe characters")
    path = Path(data_path).resolve()
    reference = load_table(path, label_column)
    bundle, table_planner, _baseline = build_table_bundle(
        reference,
        label_column,
        f"discovery-{discovery_id}",
        "Find executable failures in an agent-generated data pipeline.",
        max_specs=max_specs,
    )
    active_planner = planner or table_planner
    if backend == "local":
        specs = list(active_planner.propose(bundle, bundle.claims))
        runner = build_local_runner(reference, label_column)
        analysis = dict(getattr(active_planner, "last_analysis", None) or {})
    elif backend == "daytona":
        from ghostdata.execution.daytona import uses_baked_package
        from ghostdata.tabular import dump_frozen_model, fit_frozen_model

        load_dotenv(PROJECT_ROOT / ".env")
        settings = daytona_settings or DaytonaSettings()
        proposal_cls = DaytonaProposalRunner
        runner_cls = DaytonaVerificationRunner
        baked = uses_baked_package(settings)
        volume = bool(settings.volume_name)
        proposer_files = package_sandbox_files(
            path,
            {
                "proposer.py": (
                    PROJECT_ROOT / "demo" / "pipeline" / "proposer.py"
                ).read_bytes(),
                "task.json": json.dumps(
                    {
                        "label_column": label_column,
                        "dataset": "/data/dataset.csv" if volume else "dataset.csv",
                        "claim_id": "C001",
                        "max_specs": max_specs,
                    }
                ).encode(),
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
        active_planner = FrozenSpecsPlanner(tuple(specs), analysis)
        runner = runner_cls(
            lambda _bundle, _spec: build_executor_job(path, label_column, settings),
            settings=settings,
        )
    else:
        raise ValueError(f"unsupported discovery backend: {backend}")

    annotated: list[VerificationSpec] = []
    for spec in specs:
        parameters = dict(spec.parameters)
        parameters["discovery_id"] = discovery_id
        parameters["execution_backend"] = backend
        parameters.setdefault("agent_id", spec.verification_id)
        annotated.append(
            VerificationSpec(
                verification_id=spec.verification_id,
                claim_id=spec.claim_id,
                experiment_type=spec.experiment_type,
                hypothesis=spec.hypothesis,
                parameters=parameters,
                expected_invariants=spec.expected_invariants,
                origin=spec.origin,
            )
        )
    active_planner = FrozenSpecsPlanner(tuple(annotated), dict(analysis))
    workers = min(max(len(annotated), 1), MAX_LIVE_WORLDS)
    orchestrator = VerificationOrchestrator(
        runner,
        EvaluatorRegistry((ModelMetricPreservationEvaluator(),)),
        max_workers=workers,
    )
    report = orchestrator.verify(bundle, BundleClaimExtractor(), active_planner)
    winning_spec, winning_evidence = _winner(report, annotated)
    discovery = {
        "discovery_id": discovery_id,
        "backend": backend,
        "label_column": label_column,
        "agents": _agent_results(report, annotated),
        "proposal": dict(analysis),
        "verification_report": report.to_dict(),
    }

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / discovery_id
    if run_dir.exists():
        raise FileExistsError(f"discovery already exists: {discovery_id}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{discovery_id}-", dir=root))
    try:
        if backend == "local":
            _promote_local(
                path,
                label_column,
                bundle,
                winning_spec,
                winning_evidence,
                discovery,
                temporary,
            )
        else:
            _promote_daytona(
                path,
                label_column,
                bundle,
                winning_spec,
                discovery,
                temporary,
                daytona_settings,
            )
        temporary.rename(run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_ghost_artifacts(path, run_dir, label_column)


def run_credit_discovery(
    backend: Literal["local", "daytona"] = "local",
    data_path: Path | str = FULL_DATA_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    profiles: Sequence[AgentProfile] = DEFAULT_AGENT_PROFILES,
    discovery_id: str | None = None,
    daytona_settings: DaytonaSettings | None = None,
    label_column: str = TARGET_COLUMN,
    max_specs: int | None = None,
) -> dict[str, object]:
    del profiles
    return run_table_discovery(
        data_path,
        label_column,
        backend=backend,
        output_root=output_root,
        discovery_id=discovery_id,
        max_specs=max_specs or DEFAULT_MAX_SPECS,
        daytona_settings=daytona_settings,
    )


def load_discovery_run(
    discovery_id: str, output_root: Path | str = DEFAULT_OUTPUT_ROOT
) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", discovery_id):
        raise ValueError("discovery_id contains unsafe characters")
    report_path = (
        Path(output_root).resolve()
        / discovery_id
        / ARTIFACT_NAMES["model_report"]
    )
    if not report_path.is_file():
        raise FileNotFoundError(discovery_id)
    return json.loads(report_path.read_text(encoding="utf-8"))


def list_discovery_runs(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> list[dict[str, object]]:
    root = Path(output_root).resolve()
    if not root.exists():
        return []
    reports = []
    for report_path in sorted(root.glob(f"*/{ARTIFACT_NAMES['model_report']}")):
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    return reports


def discovery_artifact_path(
    discovery_id: str,
    role: str,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if role not in ARTIFACT_NAMES:
        raise KeyError(role)
    load_discovery_run(discovery_id, output_root)
    return Path(output_root).resolve() / discovery_id / ARTIFACT_NAMES[role]
