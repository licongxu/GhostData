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


@dataclass(frozen=True)
class DaytonaJob:
    command: str
    files: Mapping[str, bytes] = field(default_factory=dict)
    evidence_path: str = "evidence.json"
    download_paths: Mapping[str, str] = field(default_factory=dict)


JobFactory = Callable[[AnalysisBundle, VerificationSpec], DaytonaJob]
ArtifactSink = Callable[
    [AnalysisBundle, VerificationSpec, Mapping[str, bytes]], Mapping[str, str]
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
                "bundle_id": bundle.bundle_id,
                "claim_id": spec.claim_id,
                "verification_id": spec.verification_id,
                "experiment_type": spec.experiment_type,
            }
            for key in ("agent_id", "discovery_id"):
                value = spec.parameters.get(key)
                if isinstance(value, str) and value:
                    labels[key] = value
            sandbox = self._client.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self._settings.snapshot,
                    language="python",
                    ephemeral=True,
                    labels=labels,
                    auto_stop_interval=self._settings.auto_stop_minutes,
                    network_block_all=self._settings.network_block_all,
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
        normalized = posixpath.normpath(relative_path)
        if (
            normalized in {"", ".", ".."}
            or normalized.startswith("/")
            or normalized.startswith("../")
        ):
            raise ValueError(f"job path must stay inside the work directory: {relative_path}")
        return posixpath.join(self._settings.work_dir, normalized)
