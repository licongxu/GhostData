"""Multi-sandbox credit discovery and promotion for the hackathon backend."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import pandas as pd
from dotenv import load_dotenv

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.demo.artifacts import (
    ARTIFACT_NAMES,
    build_credit_artifacts,
    validate_credit_artifacts,
)
from ghostdata.demo.credit import (
    PROJECT_ROOT,
    TARGET_FEATURE,
    credit_invariants,
    fitted_credit_model_score,
    load_credit_data,
)
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.daytona import (
    DaytonaJob,
    DaytonaSettings,
    DaytonaVerificationRunner,
)
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.verification import ExecutionEvidence, VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


FULL_DATA_PATH = PROJECT_ROOT / "data" / "build" / "givemesomecredit.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "discovery"
DISCOVERY_WORKER = PROJECT_ROOT / "demo" / "credit_pipeline" / "discovery_worker.py"
ARTIFACT_WORKER = PROJECT_ROOT / "demo" / "credit_pipeline" / "artifact_worker.py"


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
    files = {
        "dataset.csv": data_path.read_bytes(),
        "worker.py": worker_path.read_bytes(),
    }
    package_root = PROJECT_ROOT / "src" / "ghostdata"
    for source_path in package_root.rglob("*.py"):
        remote = (Path("src") / source_path.relative_to(PROJECT_ROOT / "src")).as_posix()
        files[remote] = source_path.read_bytes()
    return files


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
            "agent_id": spec.parameters["agent_id"],
            "verification_id": spec.verification_id,
            "outcome": verdicts[spec.verification_id].outcome,
            "reason": verdicts[spec.verification_id].reason,
            "measurements": dict(verdicts[spec.verification_id].measurements),
        }
        for spec in specs
    ]


def _winner(
    report: VerificationReport, specs: Sequence[VerificationSpec]
) -> tuple[VerificationSpec, ExecutionEvidence]:
    specs_by_id = {spec.verification_id: spec for spec in specs}
    evidence_by_id = {item.verification_id: item for item in report.evidence}
    eligible = []
    for ghost in report.ghosts:
        degradation = ghost.measurements.get("degradation")
        if isinstance(degradation, (int, float)) and isfinite(degradation):
            eligible.append((float(degradation), ghost.verification_id))
    if not eligible:
        raise RuntimeError("discovery completed without a promotable Ghost")
    _, verification_id = max(eligible, key=lambda item: item[0])
    return specs_by_id[verification_id], evidence_by_id[verification_id]


def _promote_daytona(
    prepared: PreparedCreditDiscovery,
    spec: VerificationSpec,
    discovery: Mapping[str, object],
    destination: Path,
    settings: DaytonaSettings | None,
) -> None:
    def sink(
        bundle: AnalysisBundle,
        promoted_spec: VerificationSpec,
        artifacts: Mapping[str, bytes],
    ) -> Mapping[str, str]:
        if set(artifacts) != set(ARTIFACT_NAMES):
            raise ValueError("promotion sandbox returned an invalid artifact set")
        destination.mkdir(parents=True, exist_ok=True)
        for role, contents in artifacts.items():
            (destination / ARTIFACT_NAMES[role]).write_bytes(contents)
        return dict(ARTIFACT_NAMES)

    files = _job_files(prepared.data_path, ARTIFACT_WORKER)
    files["discovery_report.json"] = json.dumps(
        discovery, sort_keys=True, allow_nan=False
    ).encode()
    runner = DaytonaVerificationRunner(
        lambda bundle, promoted_spec: DaytonaJob(
            command="PYTHONPATH=src python worker.py",
            files=files,
            evidence_path="promotion_evidence.json",
            download_paths={
                role: f"outputs/{filename}"
                for role, filename in ARTIFACT_NAMES.items()
            },
        ),
        settings=settings,
        artifact_sink=sink,
    )
    runner.run(prepared.bundle, spec)
    validate_credit_artifacts(prepared.data_path, destination)


def run_credit_discovery(
    backend: Literal["local", "daytona"] = "local",
    data_path: Path | str = FULL_DATA_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    profiles: Sequence[AgentProfile] = DEFAULT_AGENT_PROFILES,
    discovery_id: str | None = None,
    daytona_settings: DaytonaSettings | None = None,
) -> dict[str, object]:
    discovery_id = discovery_id or uuid.uuid4().hex[:12]
    prepared = prepare_credit_discovery(
        data_path, profiles, discovery_id, backend
    )
    if backend == "local":
        runner = LocalVerificationRunner(
            prepared.reference,
            default_compiler(),
            credit_invariants,
            prepared.scorer,
            metric="roc_auc",
        )
    elif backend == "daytona":
        load_dotenv(PROJECT_ROOT / ".env")
        runner = DaytonaVerificationRunner(
            lambda bundle, spec: build_discovery_job(prepared, bundle, spec),
            settings=daytona_settings,
        )
    else:
        raise ValueError(f"unsupported discovery backend: {backend}")

    specs = prepared.specs
    orchestrator = VerificationOrchestrator(
        runner,
        EvaluatorRegistry((ModelMetricPreservationEvaluator(),)),
        max_workers=len(specs),
    )
    report = orchestrator.verify(
        prepared.bundle, BundleClaimExtractor(), prepared.planner
    )
    winning_spec, winning_evidence = _winner(report, specs)
    discovery = {
        "discovery_id": discovery_id,
        "backend": backend,
        "agents": _agent_results(report, specs),
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
            build_credit_artifacts(
                prepared.data_path,
                prepared.bundle,
                winning_spec,
                winning_evidence,
                discovery,
                temporary,
            )
        else:
            _promote_daytona(
                prepared, winning_spec, discovery, temporary, daytona_settings
            )
        temporary.rename(run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_credit_artifacts(prepared.data_path, run_dir)


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

