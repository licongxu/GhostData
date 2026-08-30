"""Daytona execution lab for isolated verification experiments."""

from __future__ import annotations

import json
import posixpath
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from daytona import (
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    ListSandboxesQuery,
)

from ghostdata.bundle import AnalysisBundle
from ghostdata.verification import ExecutionEvidence, VerificationSpec


RUNNER_SNAPSHOT = "ghostdata-runner"
RUNNER_VOLUME = "ghostdata-data"
RUNNER_PYTHONPATH = "/opt/ghostdata"
PROPOSER_CHART_CODE = """
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

task = json.loads(Path("task.json").read_text())
frame = pd.read_csv(task["dataset"])
column = next((name for name in frame.columns if name != task.get("label_column")), frame.columns[0])
plt.figure()
frame[column].dropna().hist(bins=16)
plt.title("Same values. Different relationships.")
plt.xlabel(str(column))
plt.ylabel("Rows")
plt.show()
print("ghostdata-chart-ready")
"""


def interpreter_inspect_code(work_dir: str) -> str:
    return (
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        f"work = Path({work_dir!r})\n"
        "task = {}\n"
        "task_path = work / 'task.json'\n"
        "if task_path.is_file():\n"
        "    task = json.loads(task_path.read_text())\n"
        "candidates = []\n"
        "dataset = task.get('dataset')\n"
        "if dataset:\n"
        "    candidates.append(Path(dataset))\n"
        "candidates.extend([work / 'dataset.csv', Path('/data/dataset.csv')])\n"
        "path = next((item for item in candidates if item.is_file()), None)\n"
        "if path is None:\n"
        "    print('ghostdata-interpreter: no dataset')\n"
        "else:\n"
        "    frame = pd.read_csv(path)\n"
        "    print('ghostdata-interpreter', {'rows': int(len(frame)), "
        "'columns': [str(name) for name in frame.columns[:16]]})\n"
    )


def serialize_code_run_charts(response: Any) -> list[dict[str, Any]]:
    artifacts = getattr(response, "artifacts", None)
    charts = getattr(artifacts, "charts", None) or []
    payload: list[dict[str, Any]] = []
    for chart in charts:
        if hasattr(chart, "model_dump"):
            item = chart.model_dump(mode="json")
        elif isinstance(chart, Mapping):
            item = dict(chart)
        else:
            item = {
                "title": getattr(chart, "title", None),
                "type": str(getattr(chart, "type", "") or ""),
            }
        payload.append(json.loads(json.dumps(item, default=str, allow_nan=False)))
    return payload


def attach_charts(evidence: ExecutionEvidence, charts: list[dict[str, Any]]) -> ExecutionEvidence:
    if not charts:
        return evidence
    observations = dict(evidence.observations)
    observations["daytona_charts"] = charts
    return replace(evidence, observations=observations)


def exec_in_sandbox(
    sandbox: Any,
    command: str,
    *,
    cwd: str,
    timeout: int,
    session_id: str | None = None,
) -> Any:
    execute = getattr(sandbox.process, "execute_session_command", None)
    if session_id and callable(execute):
        from daytona import SessionExecuteRequest

        request = SessionExecuteRequest(command=f"cd {shlex.quote(cwd)} && {command}")
        response = execute(session_id, request, timeout=timeout)
        result = (
            getattr(response, "output", None)
            or getattr(response, "stdout", None)
            or getattr(response, "result", "")
            or ""
        )
        return SimpleNamespace(
            exit_code=getattr(response, "exit_code", 0) or 0,
            result=str(result),
        )
    return sandbox.process.exec(command, cwd=cwd, timeout=timeout)


def maybe_create_session(sandbox: Any, session_id: str) -> str | None:
    create = getattr(sandbox.process, "create_session", None)
    if not callable(create):
        return None
    create(session_id)
    return session_id


def maybe_delete_session(sandbox: Any, session_id: str | None) -> None:
    if not session_id:
        return
    delete = getattr(sandbox.process, "delete_session", None)
    if callable(delete):
        delete(session_id)


def maybe_inspect_table(sandbox: Any, work_dir: str) -> None:
    interpreter = getattr(sandbox, "code_interpreter", None)
    run_code = getattr(interpreter, "run_code", None) if interpreter is not None else None
    if callable(run_code):
        run_code(interpreter_inspect_code(work_dir))


@dataclass(frozen=True)
class DaytonaSettings:
    snapshot: str = RUNNER_SNAPSHOT
    work_dir: str = "/home/daytona/ghostdata"
    create_timeout_seconds: int = 180
    command_timeout_seconds: int = 180
    auto_stop_minutes: int = 10
    network_block_all: bool = True
    volume_name: str | None = RUNNER_VOLUME
    volume_mount: str = "/data"
    network_allow_list: str | None = None
    volume_subpath: str | None = None


@dataclass(frozen=True)
class DaytonaJob:
    command: str
    files: Mapping[str, bytes] = field(default_factory=dict)
    evidence_path: str = "evidence.json"
    download_paths: Mapping[str, str] = field(default_factory=dict)
    role: str = "executor"
    extra_labels: Mapping[str, str] = field(default_factory=dict)
    env_vars: Mapping[str, str] = field(default_factory=dict)
    network_block_all: bool | None = None
    volume_files: Mapping[str, bytes] = field(default_factory=dict)
    code_run: str | None = None


JobFactory = Callable[[AnalysisBundle, VerificationSpec], DaytonaJob]
ArtifactSink = Callable[
    [AnalysisBundle, VerificationSpec, Mapping[str, bytes]], Mapping[str, str]
]


def sandbox_pythonpath(settings: DaytonaSettings) -> str:
    return RUNNER_PYTHONPATH if settings.snapshot == RUNNER_SNAPSHOT else "src"


def uses_baked_package(settings: DaytonaSettings) -> bool:
    return settings.snapshot == RUNNER_SNAPSHOT


def ghostdata_runner_image() -> Any:
    from daytona import Image

    package = Path(__file__).resolve().parents[1]
    return (
        Image.debian_slim("3.12")
        .pip_install(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "pyarrow",
                "matplotlib",
                "joblib",
            ]
        )
        .add_local_dir(str(package), "/opt/ghostdata/ghostdata")
        .env({"PYTHONPATH": RUNNER_PYTHONPATH})
        .workdir("/home/daytona")
    )


def ensure_runner_snapshot(
    client: Any, name: str = RUNNER_SNAPSHOT, create: bool = True
) -> str:
    snapshot_service = getattr(client, "snapshot", None)
    if snapshot_service is None:
        raise RuntimeError("Daytona client has no snapshot service")
    try:
        snapshot = snapshot_service.get(name)
        state = str(getattr(snapshot, "state", "active")).lower()
        if "error" not in state and "fail" not in state:
            return name
    except Exception:
        snapshot = None
    if not create:
        raise RuntimeError(f"Daytona snapshot missing: {name}")
    from daytona import CreateSnapshotParams

    snapshot_service.create(
        CreateSnapshotParams(name=name, image=ghostdata_runner_image()),
        timeout=600,
    )
    return name


def leftover_sandboxes(client: Any, labels: Mapping[str, str] | None = None) -> list[Any]:
    query = ListSandboxesQuery(labels=dict(labels or {"project": "ghostdata"}))
    return list(client.list(query))


def _maybe_ensure_snapshot(client: Any, settings: DaytonaSettings) -> None:
    if getattr(client, "snapshot", None) is None:
        return
    ensure_runner_snapshot(client, settings.snapshot)


def _safe_relative_path(relative_path: str) -> str:
    normalized = posixpath.normpath(relative_path)
    if (
        normalized in {"", ".", ".."}
        or normalized.startswith("/")
        or normalized.startswith("../")
    ):
        raise ValueError(f"job path must stay inside the work directory: {relative_path}")
    return normalized


def _volume_mounts(client: Any, settings: DaytonaSettings) -> list[Any] | None:
    volume_service = getattr(client, "volume", None)
    if not settings.volume_name or volume_service is None:
        return None
    from daytona import VolumeMount

    volume = volume_service.get(settings.volume_name, create=True)
    return [
        VolumeMount(
            volume_id=volume.id,
            mount_path=settings.volume_mount,
            subpath=settings.volume_subpath,
        )
    ]


def _upload_tree(sandbox: Any, files: Mapping[str, bytes]) -> None:
    for remote_path, contents in files.items():
        parent = posixpath.dirname(remote_path)
        if parent:
            sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
        sandbox.fs.upload_file(contents, remote_path)


def _seed_volume(sandbox: Any, settings: DaytonaSettings, files: Mapping[str, bytes]) -> None:
    if not files:
        return
    sandbox.process.exec(f"mkdir -p {shlex.quote(settings.volume_mount)}")
    for name, contents in files.items():
        remote = posixpath.join(settings.volume_mount, _safe_relative_path(name))
        sandbox.fs.upload_file(contents, remote)


class DaytonaVerificationRunner:
    """Execute one experiment per ephemeral sandbox and return only raw evidence."""

    _RESERVED_INPUTS = {"bundle.json", "verification.json"}

    def __init__(
        self,
        job_factory: JobFactory,
        settings: DaytonaSettings | None = None,
        client: Any | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self._job_factory = job_factory
        self._settings = settings or DaytonaSettings()
        self._client = client or Daytona(DaytonaConfig(use_deprecated_polling=False))
        self._artifact_sink = artifact_sink

    def run(
        self, bundle: AnalysisBundle, spec: VerificationSpec
    ) -> ExecutionEvidence:
        job = self._job_factory(bundle, spec)
        remote_files: dict[str, bytes] = {}
        for relative_path, contents in job.files.items():
            normalized = posixpath.normpath(relative_path)
            if normalized in self._RESERVED_INPUTS:
                raise ValueError(f"job path is reserved: {normalized}")
            remote_files[self._remote_path(relative_path)] = contents
        remote_evidence_path = self._remote_path(job.evidence_path)
        remote_downloads: dict[str, str] = {}
        for role, relative_path in job.download_paths.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("download roles must be non-empty strings")
            remote_downloads[role] = self._remote_path(relative_path)
        if remote_downloads and self._artifact_sink is None:
            raise ValueError("artifact downloads require an artifact sink")

        sandbox = None
        try:
            _maybe_ensure_snapshot(self._client, self._settings)
            labels = {
                "project": "ghostdata",
                "role": job.role,
                "bundle_id": bundle.bundle_id,
                "claim_id": spec.claim_id,
                "verification_id": spec.verification_id,
                "experiment_type": spec.experiment_type,
            }
            for key in ("agent_id", "discovery_id"):
                value = spec.parameters.get(key)
                if isinstance(value, str) and value:
                    labels[key] = value
            for key, value in job.extra_labels.items():
                if isinstance(key, str) and key and isinstance(value, str) and value:
                    labels[key] = value
            blocked = (
                self._settings.network_block_all
                if job.network_block_all is None
                else job.network_block_all
            )
            sandbox = self._client.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self._settings.snapshot,
                    language="python",
                    ephemeral=True,
                    labels=labels,
                    auto_stop_interval=self._settings.auto_stop_minutes,
                    network_block_all=blocked,
                    env_vars=dict(job.env_vars) or None,
                    volumes=_volume_mounts(self._client, self._settings),
                ),
                timeout=self._settings.create_timeout_seconds,
            )
            sandbox.process.exec(f"mkdir -p {shlex.quote(self._settings.work_dir)}")
            sandbox.fs.upload_file(
                bundle.to_json().encode(),
                posixpath.join(self._settings.work_dir, "bundle.json"),
            )
            sandbox.fs.upload_file(
                spec.to_json().encode(),
                posixpath.join(self._settings.work_dir, "verification.json"),
            )
            _upload_tree(sandbox, remote_files)
            _seed_volume(sandbox, self._settings, job.volume_files)
            maybe_inspect_table(sandbox, self._settings.work_dir)
            charts: list[dict[str, Any]] = []
            session_id = maybe_create_session(sandbox, job.role or "executor")
            try:
                response = exec_in_sandbox(
                    sandbox,
                    job.command,
                    cwd=self._settings.work_dir,
                    timeout=self._settings.command_timeout_seconds,
                    session_id=session_id,
                )
                if response.exit_code != 0:
                    raise RuntimeError(
                        f"verification command exited {response.exit_code}: "
                        f"{response.result.strip()}"
                    )
                if job.code_run:
                    chart_response = sandbox.process.code_run(
                        job.code_run,
                        timeout=self._settings.command_timeout_seconds,
                    )
                    if getattr(chart_response, "exit_code", 0) not in (0, None):
                        raise RuntimeError(
                            f"code_run exited {chart_response.exit_code}: "
                            f"{str(getattr(chart_response, 'result', '')).strip()}"
                        )
                    charts = serialize_code_run_charts(chart_response)
            finally:
                maybe_delete_session(sandbox, session_id)

            payload = sandbox.fs.download_file(remote_evidence_path)
            evidence = attach_charts(
                ExecutionEvidence.from_dict(json.loads(payload)), charts
            )
            self._validate_identity(bundle, spec, evidence)
            if remote_downloads:
                artifacts = {
                    role: sandbox.fs.download_file(remote_path)
                    for role, remote_path in remote_downloads.items()
                }
                persisted = self._artifact_sink(bundle, spec, artifacts)
                evidence = replace(evidence, artifact_paths=persisted)
            return evidence
        finally:
            if sandbox is not None:
                self._client.delete(sandbox, wait=True)

    def list_executions(self) -> list[dict[str, str]]:
        query = ListSandboxesQuery(labels={"project": "ghostdata"})
        keys = (
            "bundle_id",
            "claim_id",
            "verification_id",
            "experiment_type",
            "agent_id",
            "discovery_id",
            "role",
        )
        return [
            {
                "sandbox_id": str(sandbox.id),
                "state": str(sandbox.state),
                **{
                    key: str(getattr(sandbox, "labels", {}).get(key, ""))
                    for key in keys
                },
            }
            for sandbox in self._client.list(query)
        ]

    @staticmethod
    def _validate_identity(
        bundle: AnalysisBundle,
        spec: VerificationSpec,
        evidence: ExecutionEvidence,
    ) -> None:
        expected = (
            bundle.bundle_id,
            spec.verification_id,
            spec.claim_id,
            spec.experiment_type,
        )
        observed = (
            evidence.bundle_id,
            evidence.verification_id,
            evidence.claim_id,
            evidence.experiment_type,
        )
        if observed != expected:
            raise ValueError(
                f"evidence identity {observed!r} does not match verification {expected!r}"
            )

    def _remote_path(self, relative_path: str) -> str:
        return posixpath.join(self._settings.work_dir, _safe_relative_path(relative_path))


class DaytonaProposalRunner:
    """Inspect a table inside an isolated sandbox and return VerificationSpecs."""

    def __init__(
        self,
        settings: DaytonaSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or DaytonaSettings()
        self._client = client or Daytona(DaytonaConfig(use_deprecated_polling=False))

    def propose(
        self,
        bundle: AnalysisBundle,
        files: Mapping[str, bytes],
        label_column: str,
        claim_id: str,
        volume_files: Mapping[str, bytes] | None = None,
    ) -> tuple[list[VerificationSpec], dict[str, Any]]:
        work = self._settings.work_dir
        remote_files: dict[str, bytes] = {
            posixpath.join(work, _safe_relative_path(relative_path)): contents
            for relative_path, contents in files.items()
        }
        sandbox = None
        try:
            _maybe_ensure_snapshot(self._client, self._settings)
            sandbox = self._client.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self._settings.snapshot,
                    language="python",
                    ephemeral=True,
                    labels={
                        "project": "ghostdata",
                        "role": "proposer",
                        "bundle_id": bundle.bundle_id,
                        "claim_id": claim_id,
                    },
                    auto_stop_interval=self._settings.auto_stop_minutes,
                    network_block_all=True,
                    network_allow_list=self._settings.network_allow_list,
                    volumes=_volume_mounts(self._client, self._settings),
                ),
                timeout=self._settings.create_timeout_seconds,
            )
            sandbox.process.exec(f"mkdir -p {shlex.quote(work)}")
            _upload_tree(sandbox, remote_files)
            _seed_volume(sandbox, self._settings, volume_files or {})
            maybe_inspect_table(sandbox, work)
            session_id = maybe_create_session(sandbox, "proposer")
            pythonpath = sandbox_pythonpath(self._settings)
            try:
                response = exec_in_sandbox(
                    sandbox,
                    f"PYTHONPATH={pythonpath} python proposer.py",
                    cwd=work,
                    timeout=self._settings.command_timeout_seconds,
                    session_id=session_id,
                )
                if response.exit_code != 0:
                    raise RuntimeError(
                        f"proposer exited {response.exit_code}: {response.result.strip()}"
                    )
                charts: list[dict[str, Any]] = []
                code_run = getattr(sandbox.process, "code_run", None)
                if callable(code_run):
                    chart = code_run(
                        PROPOSER_CHART_CODE,
                        timeout=self._settings.command_timeout_seconds,
                    )
                    if getattr(chart, "exit_code", 0) not in (0, None):
                        raise RuntimeError(
                            f"proposer code_run exited {chart.exit_code}: "
                            f"{str(getattr(chart, 'result', '')).strip()}"
                        )
                    charts = serialize_code_run_charts(chart)
            finally:
                maybe_delete_session(sandbox, session_id)
            specs = [
                VerificationSpec.from_dict(item)
                for item in json.loads(
                    sandbox.fs.download_file(posixpath.join(work, "specs.json"))
                )
            ]
            if not specs:
                raise RuntimeError("proposer emitted no VerificationSpec objects")
            analysis = json.loads(
                sandbox.fs.download_file(posixpath.join(work, "analysis.json"))
            )
            analysis["proposal_log"] = str(getattr(response, "result", "") or "")
            analysis["inspected_columns"] = list(analysis.get("inspected_columns") or [])
            if charts:
                analysis["daytona_charts"] = charts
            return specs, analysis
        finally:
            if sandbox is not None:
                self._client.delete(sandbox, wait=True)
