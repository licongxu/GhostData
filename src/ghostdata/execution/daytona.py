"""Daytona execution lab for isolated verification experiments."""

from __future__ import annotations

import json
import posixpath
import shlex
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig, ListSandboxesQuery

from ghostdata.bundle import AnalysisBundle
from ghostdata.verification import ExecutionEvidence, VerificationSpec


@dataclass(frozen=True)
class DaytonaSettings:
    snapshot: str = "daytona-small"
    work_dir: str = "/home/daytona/ghostdata"
    create_timeout_seconds: int = 180
    command_timeout_seconds: int = 180
    auto_stop_minutes: int = 10
    network_block_all: bool = True
    volume_name: str | None = None
    volume_mount: str = "/data"


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


JobFactory = Callable[[AnalysisBundle, VerificationSpec], DaytonaJob]
ArtifactSink = Callable[
    [AnalysisBundle, VerificationSpec, Mapping[str, bytes]], Mapping[str, str]
]


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
        VolumeMount(volume_id=volume.id, mount_path=settings.volume_mount)
    ]


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
            for remote_path, contents in remote_files.items():
                parent = posixpath.dirname(remote_path)
                sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
                sandbox.fs.upload_file(contents, remote_path)

            response = sandbox.process.exec(
                job.command,
                cwd=self._settings.work_dir,
                timeout=self._settings.command_timeout_seconds,
            )
            if response.exit_code != 0:
                raise RuntimeError(
                    f"verification command exited {response.exit_code}: "
                    f"{response.result.strip()}"
                )

            payload = sandbox.fs.download_file(remote_evidence_path)
            evidence = ExecutionEvidence.from_dict(json.loads(payload))
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
    """Inspect a table inside an isolated sandbox and return a VerificationSpec."""

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
    ) -> tuple[list[VerificationSpec], dict[str, Any]]:
        work = self._settings.work_dir
        remote_files: dict[str, bytes] = {
            posixpath.join(work, _safe_relative_path(relative_path)): contents
            for relative_path, contents in files.items()
        }
        sandbox = None
        try:
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
                    volumes=_volume_mounts(self._client, self._settings),
                ),
                timeout=self._settings.create_timeout_seconds,
            )
            sandbox.process.exec(f"mkdir -p {shlex.quote(work)}")
            interpreter = getattr(sandbox, "code_interpreter", None)
            if interpreter is not None:
                interpreter.run_code("print('ghostdata-proposer-ready')")
            for remote, contents in remote_files.items():
                parent = posixpath.dirname(remote)
                sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
                sandbox.fs.upload_file(contents, remote)
            create_session = getattr(sandbox.process, "create_session", None)
            if callable(create_session):
                create_session("proposer")
            response = sandbox.process.exec(
                "PYTHONPATH=src python proposer.py",
                cwd=work,
                timeout=self._settings.command_timeout_seconds,
            )
            if response.exit_code != 0:
                raise RuntimeError(
                    f"proposer exited {response.exit_code}: {response.result.strip()}"
                )
            spec = VerificationSpec.from_dict(
                json.loads(sandbox.fs.download_file(posixpath.join(work, "verification.json")))
            )
            analysis = json.loads(
                sandbox.fs.download_file(posixpath.join(work, "analysis.json"))
            )
            delete_session = getattr(sandbox.process, "delete_session", None)
            if callable(delete_session):
                delete_session("proposer")
            return [spec], analysis
        finally:
            if sandbox is not None:
                self._client.delete(sandbox, wait=True)
