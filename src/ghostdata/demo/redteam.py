"""CSV + prompt → Daytona analyst → parallel Daytona verifiers → Ghost report.

The host is the control plane. It does not decide the Ghost. Analysts propose;
verifier sandboxes measure; the host evaluator ranks measured evidence.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import load_dotenv

from ghostdata.bundle import AgentOutput, AnalysisBundle, BundleClaimExtractor, Claim
from ghostdata.demo.artifacts import ARTIFACT_NAMES, build_ghost_artifacts
from ghostdata.demo.credit import PROJECT_ROOT
from ghostdata.demo.table import (
    MAX_LIVE_WORLDS,
    select_winner,
)
from ghostdata.evaluators import EvaluatorRegistry, ModelMetricPreservationEvaluator
from ghostdata.execution.daytona import (
    DaytonaJob,
    DaytonaSettings,
    DaytonaVerificationRunner,
    leftover_sandboxes,
)
from ghostdata.execution.local import LocalVerificationRunner, default_compiler
from ghostdata.tabular import frozen_model_score, table_invariants
from ghostdata.verification import ExecutionEvidence, VerificationReport, VerificationSpec
from ghostdata.verification.search import VerificationOrchestrator


DAYTONA_ROOT = "/home/daytona/workspace"
RUNNER_SNAPSHOT = "ghostdata-runner"
MAX_WORLDS = 4
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "runs"


EventSink = Callable[[dict[str, Any]], None]


@dataclass
class ProposedWorld:
    world_id: str
    title: str
    hypothesis: str
    transform_source: str
    target_feature: str | None = None
    mismatch_fraction: float | None = None
    seed: int | None = None
    expected_invariants: tuple[str, ...] = (
        "schema",
        "marginal_distribution",
        "missing_rate",
    )


@dataclass
class AnalystOutput:
    label_column: str
    analysis: dict[str, Any]
    worlds: list[ProposedWorld]
    sandbox_id: str = ""


@dataclass
class RunState:
    run_id: str
    status: str = "running"
    events: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    error: str | None = None


_RUNS: dict[str, RunState] = {}
_RUNS_LOCK = threading.Lock()


def _emit(state: RunState, **event: Any) -> None:
    payload = {"run_id": state.run_id, **event}
    state.events.append(payload)


def get_run(run_id: str) -> dict[str, Any]:
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
    if state is None:
        raise FileNotFoundError(run_id)
    return {
        "run_id": state.run_id,
        "status": state.status,
        "events": list(state.events),
        "report": state.report,
        "error": state.error,
    }


def list_runs() -> list[dict[str, Any]]:
    with _RUNS_LOCK:
        states = list(_RUNS.values())
    return [
        {
            "run_id": state.run_id,
            "status": state.status,
            "error": state.error,
            "ghosts": (state.report or {}).get("ghosts"),
        }
        for state in states
    ]


def _world_id(index: int, raw: str | None) -> str:
    if isinstance(raw, str) and re.fullmatch(r"W\d{3}", raw):
        return raw
    return f"W{index:03d}"


def _parse_worlds(analysis: Mapping[str, Any], files: Mapping[str, str]) -> list[ProposedWorld]:
    worlds: list[ProposedWorld] = []
    listed = list(analysis.get("hypotheses") or [])
    transform_paths = sorted(
        path for path in files if path.endswith("/transform.py") or path.endswith("transform.py")
    )
    if not transform_paths:
        raise RuntimeError("analyst did not write any worlds/transform.py")
    for index, path in enumerate(transform_paths[:MAX_WORLDS], start=1):
        parent = str(Path(path).parent)
        hypothesis_path = f"{parent}/hypothesis.json"
        meta: dict[str, Any] = {}
        if hypothesis_path in files:
            try:
                loaded = json.loads(files[hypothesis_path])
                if isinstance(loaded, dict):
                    meta = loaded
            except json.JSONDecodeError:
                meta = {}
        listed_item = listed[index - 1] if index - 1 < len(listed) and isinstance(listed[index - 1], dict) else {}
        world_id = _world_id(
            index,
            str(meta.get("world_id") or listed_item.get("world_id") or ""),
        )
        source = files[path]
        if "def transform" not in source:
            continue
        worlds.append(
            ProposedWorld(
                world_id=world_id,
                title=str(meta.get("title") or listed_item.get("title") or world_id),
                hypothesis=str(
                    meta.get("hypothesis")
                    or listed_item.get("hypothesis")
                    or "Executable preprocessing failure."
                ),
                transform_source=source,
                target_feature=(
                    str(meta["target_feature"])
                    if isinstance(meta.get("target_feature"), str)
                    else None
                ),
                mismatch_fraction=(
                    float(meta["mismatch_fraction"])
                    if isinstance(meta.get("mismatch_fraction"), (int, float))
                    else None
                ),
                seed=int(meta["seed"]) if isinstance(meta.get("seed"), int) else None,
            )
        )
    if not worlds:
        raise RuntimeError("analyst worlds were missing transform() functions")
    return worlds


def stub_analyst(csv_bytes: bytes, prompt: str, filename: str) -> AnalystOutput:
    import io

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(csv_bytes))
    label = None
    lowered = prompt.lower()
    for column in frame.columns:
        if str(column).lower() in lowered:
            label = str(column)
            break
    if label is None:
        binaries = [
            str(column)
            for column in frame.columns
            if int(frame[column].nunique(dropna=True)) == 2
        ]
        label = binaries[-1] if binaries else str(frame.columns[-1])
    numeric = [
        str(column)
        for column in frame.columns
        if column != label and pd.api.types.is_numeric_dtype(frame[column])
    ]
    feature = numeric[0] if numeric else str(next(c for c in frame.columns if c != label))
    source = (
        "import numpy as np\n"
        "import pandas as pd\n\n"
        f"FEATURES = {numeric or [feature]!r}\n\n"
        "def transform(dataframe: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = dataframe.copy(deep=True)\n"
        "    rng = np.random.default_rng(7)\n"
        "    for name in FEATURES:\n"
        "        if name in out.columns and len(out) > 1:\n"
        "            out[name] = rng.permutation(out[name].to_numpy(copy=True))\n"
        "    return out\n"
    )
    analysis = {
        "label_column": label,
        "inspected_columns": [str(column) for column in frame.columns],
        "fragile_assumptions": [f"{feature} may be used as an entity-aligned signal"],
        "hypotheses": [
            {
                "world_id": "W001",
                "title": f"Permute {feature}",
                "hypothesis": (
                    f"Valid {feature} values attached to the wrong rows can pass "
                    "marginal checks while the label relationship breaks."
                ),
            }
        ],
        "prompt": prompt,
        "filename": filename,
        "planner": "stub_analyst",
    }
    return AnalystOutput(
        label_column=label,
        analysis=analysis,
        worlds=[
            ProposedWorld(
                "W001",
                f"Permute {feature}",
                str(analysis["hypotheses"][0]["hypothesis"]),
                source,
            )
        ],
        sandbox_id="stub",
    )


def _proposal_from_payload(
    payload: Mapping[str, Any],
    prompt: str,
    filename: str,
    on_event: EventSink | None = None,
) -> AnalystOutput:
    files = payload.get("files")
    if not isinstance(files, Mapping) or "analysis.json" not in files:
        raise RuntimeError("analyst did not write analysis.json")
    analysis = json.loads(files["analysis.json"])
    if not isinstance(analysis, dict):
        raise RuntimeError("analysis.json is not an object")
    label_column = str(analysis.get("label_column") or "").strip()
    if not label_column:
        raise RuntimeError("analyst did not identify a label_column")
    worlds = _parse_worlds(analysis, files)
    analysis["prompt"] = prompt
    analysis["filename"] = filename
    if payload.get("planner"):
        analysis["planner"] = payload["planner"]
    if payload.get("metrics"):
        analysis["sandbox_metrics"] = payload["metrics"]
    if payload.get("daytona_charts"):
        analysis["daytona_charts"] = payload["daytona_charts"]
    if on_event is not None:
        on_event(
            {
                "kind": "analyst",
                "status": "done",
                "sandbox_id": payload.get("sandbox_id", ""),
                "text": f"✓ {len(worlds)} failure hypotheses proposed",
            }
        )
    return AnalystOutput(
        label_column, analysis, worlds, str(payload.get("sandbox_id") or "")
    )


async def daytona_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink,
    settings: DaytonaSettings | None = None,
) -> AnalystOutput:
    sys.path.insert(0, str(PROJECT_ROOT))
    from demo.pipeline.analyst import run_sandbox_analyst

    payload = await run_sandbox_analyst(csv_bytes, prompt, filename, on_event, settings)
    return _proposal_from_payload(payload, prompt, filename, on_event)


def _world_spec(world: ProposedWorld, run_id: str) -> VerificationSpec:
    parameters: dict[str, Any] = {
        "transform_source": world.transform_source,
        "agent_id": world.world_id,
        "title": world.title,
        "discovery_id": run_id,
        "execution_backend": "daytona",
    }
    experiment_type = "generated_transform"
    if world.target_feature:
        experiment_type = "entity_alignment"
        parameters["target_feature"] = world.target_feature
        parameters["segment"] = {}
        parameters["mismatch_fraction"] = (
            world.mismatch_fraction if world.mismatch_fraction is not None else 0.5
        )
        parameters["seed"] = world.seed if world.seed is not None else 7
    return VerificationSpec(
        verification_id=world.world_id,
        claim_id="C001",
        experiment_type=experiment_type,
        hypothesis=world.hypothesis,
        parameters=parameters,
        expected_invariants=world.expected_invariants,
        origin="sandbox_agent",
    )


def _bundle(run_id: str) -> AnalysisBundle:
    claim = Claim(
        claim_id="C001",
        assertion="The preprocessing change preserves model quality.",
        evaluator="model_metric_preservation",
        parameters={
            "metric": "roc_auc",
            "max_drop": 0.0,
            "direction": "higher_is_better",
        },
        supplied_evidence={"roc_auc": 0.0},
    )
    return AnalysisBundle(
        bundle_id=f"run-{run_id}",
        task="Red-team an agent-generated data pipeline.",
        inputs={"dataset": "dataset.csv"},
        agent_output=AgentOutput(metrics={"roc_auc": 0.0}),
        claims=(claim,),
    )


def _verify_daytona(
    csv_path: Path,
    csv_bytes: bytes,
    label_column: str,
    bundle: AnalysisBundle,
    specs: list[VerificationSpec],
    on_event: EventSink,
    settings: DaytonaSettings,
) -> VerificationReport:
    from dataclasses import replace as replace_job

    from ghostdata.demo.table import build_executor_job

    del csv_bytes
    runner_box: list[DaytonaVerificationRunner] = []
    base_job = build_executor_job(csv_path, label_column, settings)

    def factory(_bundle: AnalysisBundle, spec: VerificationSpec) -> DaytonaJob:
        on_event(
            {
                "kind": "world",
                "status": "running",
                "world_id": spec.verification_id,
                "text": f"WORLD {spec.verification_id} VERIFYING…",
            }
        )
        if runner_box:
            live = runner_box[0].list_executions()
            on_event(
                {
                    "kind": "sandboxes",
                    "status": "running",
                    "world_id": spec.verification_id,
                    "text": f"{len(live)} GhostData sandbox(es) listed",
                    "sandboxes": live,
                }
            )
        return replace_job(
            base_job,
            extra_labels={
                **dict(base_job.extra_labels),
                "stage": "verify",
                "world_id": spec.verification_id,
            },
        )

    class Frozen:
        def propose(self, _bundle: AnalysisBundle, _claims: object) -> list[VerificationSpec]:
            return list(specs)

    runner = DaytonaVerificationRunner(factory, settings=settings)
    runner_box.append(runner)
    orchestrator = VerificationOrchestrator(
        runner,
        EvaluatorRegistry((ModelMetricPreservationEvaluator(),)),
        max_workers=min(max(len(specs), 1), MAX_LIVE_WORLDS),
    )
    return orchestrator.verify(bundle, BundleClaimExtractor(), Frozen())


def _verify_local(
    csv_bytes: bytes,
    label_column: str,
    bundle: AnalysisBundle,
    specs: list[VerificationSpec],
) -> VerificationReport:
    import io

    import pandas as pd

    reference = pd.read_csv(io.BytesIO(csv_bytes))
    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        table_invariants,
        frozen_model_score(reference, label_column),
        "roc_auc",
    )

    class Frozen:
        def propose(self, _bundle: AnalysisBundle, _claims: object) -> list[VerificationSpec]:
            return list(specs)

    return VerificationOrchestrator(
        runner,
        EvaluatorRegistry((ModelMetricPreservationEvaluator(),)),
        max_workers=min(max(len(specs), 1), MAX_LIVE_WORLDS),
    ).verify(bundle, BundleClaimExtractor(), Frozen())


def _world_summaries(
    specs: list[VerificationSpec],
    report: VerificationReport,
) -> list[dict[str, Any]]:
    verdicts = {
        experiment.verification_id: experiment
        for claim in report.claims
        for experiment in claim.experiments
    }
    rows = []
    for spec in specs:
        verdict = verdicts[spec.verification_id]
        measurements = dict(verdict.measurements)
        rows.append(
            {
                "world_id": spec.verification_id,
                "title": spec.parameters.get("title") or spec.verification_id,
                "hypothesis": spec.hypothesis,
                "outcome": verdict.outcome,
                "reason": verdict.reason,
                "baseline": measurements.get("baseline"),
                "candidate": measurements.get("candidate"),
                "degradation": measurements.get("degradation"),
                "invariants": {},
            }
        )
    evidence_by_id = {item.verification_id: item for item in report.evidence}
    for row in rows:
        evidence = evidence_by_id.get(str(row["world_id"]))
        if evidence is not None:
            invariants = evidence.observations.get("invariants")
            if isinstance(invariants, Mapping):
                row["invariants"] = dict(invariants)
    return rows


def run_redteam(
    csv_bytes: bytes,
    prompt: str,
    filename: str = "dataset.csv",
    run_id: str | None = None,
    on_event: EventSink | None = None,
    analyst: Callable[[bytes, str, str], AnalystOutput] | None = None,
    backend: str = "daytona",
) -> dict[str, Any]:
    run_id = run_id or uuid.uuid4().hex[:12]
    load_dotenv(PROJECT_ROOT / ".env")
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    emit(
        {
            "kind": "dataset",
            "status": "done",
            "text": f"✓ Dataset loaded ({filename}, {len(csv_bytes)} bytes)",
        }
    )
    settings = (
        DaytonaSettings(volume_subpath=run_id) if backend == "daytona" else None
    )
    if analyst is not None:
        proposal = analyst(csv_bytes, prompt, filename)
    elif backend == "daytona":
        proposal = asyncio.run(
            daytona_analyst(csv_bytes, prompt, filename, emit, settings)
        )
    else:
        from ghostdata.demo.codex_analyst import try_run_codex_analyst

        payload = try_run_codex_analyst(csv_bytes, prompt, filename, emit)
        proposal = (
            _proposal_from_payload(payload, prompt, filename, emit)
            if payload is not None
            else stub_analyst(csv_bytes, prompt, filename)
        )

    specs = [_world_spec(world, run_id) for world in proposal.worlds]
    bundle = _bundle(run_id)
    emit(
        {
            "kind": "hypotheses",
            "status": "done",
            "text": f"✓ {len(specs)} failure hypotheses proposed",
            "worlds": [world.world_id for world in proposal.worlds],
        }
    )
    with tempfile.TemporaryDirectory() as temp:
        csv_path = Path(temp) / "dataset.csv"
        csv_path.write_bytes(csv_bytes)
        if backend == "daytona":
            assert settings is not None
            report = _verify_daytona(
                csv_path,
                csv_bytes,
                proposal.label_column,
                bundle,
                specs,
                emit,
                settings,
            )
        else:
            report = _verify_local(csv_bytes, proposal.label_column, bundle, specs)

        summaries = _world_summaries(specs, report)
        for row in summaries:
            if row["outcome"] == "counterexample":
                kind, mark = "ghost", "🔥"
                text = (
                    f"{mark} WORLD {row['world_id']} VERIFIED GHOST  "
                    f"checks PASS · AUC {row['baseline']:.3f} → {row['candidate']:.3f}"
                )
            elif row["outcome"] == "inconclusive":
                kind, mark = "rejected", "×"
                text = f"{mark} WORLD {row['world_id']} REJECTED — {row['reason']}"
            else:
                kind, mark = "harmless", "✓"
                text = f"{mark} WORLD {row['world_id']} checks PASS · little or no drop"
            emit({"kind": kind, "status": "done", "world_id": row["world_id"], "text": text})

        winner = select_winner(report, specs)
        artifacts = None
        if winner is not None:
            spec, evidence = winner
            dest = Path(temp) / "out"
            dest.mkdir()
            discovery_payload = {
                "agents": summaries,
                "proposal": proposal.analysis,
                "verification_report": report.to_dict(),
            }
            build_ghost_artifacts(
                csv_path,
                proposal.label_column,
                bundle,
                spec,
                evidence,
                discovery_payload,
                dest,
            )
            published = OUTPUT_ROOT / run_id
            published.parent.mkdir(parents=True, exist_ok=True)
            if published.exists():
                raise FileExistsError(run_id)
            dest.rename(published)
            artifacts = {
                role: f"/api/runs/{run_id}/artifacts/{role}" for role in ARTIFACT_NAMES
            }

    leftover = None
    if backend == "daytona":
        time.sleep(2)
        leftover = _leftover_count()
        emit(
            {
                "kind": "cleanup",
                "status": "done",
                "text": f"leftover sandboxes: {leftover}",
            }
        )

    winning = None
    if winner is not None:
        spec, evidence = winner
        metric = evidence.observations.get("model_metric") or {}
        winning = {
            "world_id": spec.verification_id,
            "hypothesis": spec.hypothesis,
            "title": spec.parameters.get("title"),
            "baseline": metric.get("baseline"),
            "candidate": metric.get("candidate"),
            "invariants": dict(evidence.observations.get("invariants") or {}),
            "transform_source": spec.parameters.get("transform_source"),
        }
        emit(
            {
                "kind": "report",
                "status": "done",
                "text": "GHOST FOUND" if report.ghosts else "no Ghost",
            }
        )
    payload = {
        "run_id": run_id,
        "status": "completed",
        "backend": backend,
        "filename": filename,
        "prompt": prompt,
        "label_column": proposal.label_column,
        "analyst_sandbox_id": proposal.sandbox_id,
        "analysis": proposal.analysis,
        "worlds": summaries,
        "verification_report": report.to_dict(),
        "ghosts": len(report.ghosts),
        "winner": winning,
        "artifacts": artifacts,
        "leftover_sandboxes": leftover,
        "events": events,
        "verdict": report.verdict,
    }
    return payload


def start_run(
    csv_bytes: bytes,
    prompt: str,
    filename: str = "dataset.csv",
    backend: str = "daytona",
    analyst: Callable[[bytes, str, str], AnalystOutput] | None = None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(run_id=run_id)
    with _RUNS_LOCK:
        _RUNS[run_id] = state

    def worker() -> None:
        try:
            def on_event(event: dict[str, Any]) -> None:
                _emit(state, **event)

            report = run_redteam(
                csv_bytes,
                prompt,
                filename,
                run_id=run_id,
                on_event=on_event,
                analyst=analyst,
                backend=backend,
            )
            state.report = report
            state.status = "completed"
        except Exception as exc:
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            _emit(state, kind="error", status="failed", text=state.error)

    threading.Thread(target=worker, daemon=True).start()
    return run_id


def _leftover_count() -> int:
    from daytona import Daytona, DaytonaConfig

    return len(leftover_sandboxes(Daytona(DaytonaConfig(use_deprecated_polling=False))))


def run_artifact_path(run_id: str, role: str) -> Path:
    if role not in ARTIFACT_NAMES:
        raise KeyError(role)
    path = OUTPUT_ROOT / run_id / ARTIFACT_NAMES[role]
    if not path.is_file():
        raise FileNotFoundError(run_id)
    return path
