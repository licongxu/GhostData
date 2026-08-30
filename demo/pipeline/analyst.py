"""Analyst stage: Codex SDK proposes worlds; Daytona is the fallback computer.

Default (`GHOSTDATA_ANALYST=auto`) uses the local Codex SDK with ChatGPT login
credits. If Codex is missing or fails, the original pandas script still runs
inside a Daytona sandbox. Set `GHOSTDATA_ANALYST=deterministic` for the demo
path only.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from dotenv import load_dotenv

from ghostdata.demo.credit import PROJECT_ROOT
from ghostdata.execution.daytona import (
    exec_in_sandbox,
    maybe_create_session,
    maybe_delete_session,
    maybe_inspect_table,
    serialize_code_run_charts,
    RUNNER_SNAPSHOT,
    DaytonaSettings,
    _seed_volume,
    _volume_mounts,
    ensure_runner_snapshot,
)


EventSink = Callable[[dict[str, Any]], None]
ANALYST_SCRIPT = PROJECT_ROOT / "demo" / "pipeline" / "sandbox_analyst.py"
DAYTONA_ROOT = "/home/daytona/workspace"
ANALYST_CHART_CODE = """
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
path = Path("/data/dataset.csv")
if not path.is_file():
    path = Path("/home/daytona/workspace/data/dataset.csv")
frame = pd.read_csv(path)
column = frame.columns[0]
plt.figure()
frame[column].dropna().hist(bins=16)
plt.title("Same values. Different relationships.")
plt.xlabel(str(column))
plt.ylabel("Rows")
plt.show()
print("ghostdata-chart-ready")
"""


def download_workspace(sandbox: Any, root: str) -> dict[str, str]:
    listing = sandbox.process.exec(f"find {root} -type f")
    output = str(getattr(listing, "result", "") or "")
    files: dict[str, str] = {}
    for line in output.splitlines():
        path = line.strip()
        if not path:
            continue
        try:
            payload = sandbox.fs.download_file(path)
        except Exception:
            continue
        relative = path[len(root) :].lstrip("/") if path.startswith(root) else path
        files[relative] = payload.decode("utf-8", errors="replace")
    return files


def _metrics_payload(sandbox: Any) -> dict[str, object] | None:
    metrics_fn = getattr(sandbox, "get_metrics_latest", None)
    if not callable(metrics_fn):
        return None
    latest = metrics_fn()
    if latest is None:
        return None
    if isinstance(latest, dict):
        return dict(latest)
    return {
        key: getattr(latest, key, None)
        for key in ("cpu", "memory", "disk", "network_rx", "network_tx")
        if getattr(latest, key, None) is not None
    }


def _run_python_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink,
    settings: DaytonaSettings | None = None,
) -> dict[str, Any]:
    import shlex

    from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig

    load_dotenv(PROJECT_ROOT / ".env")
    resolved = settings or DaytonaSettings()
    client = Daytona(DaytonaConfig(use_deprecated_polling=False))
    sandbox = None
    try:
        ensure_runner_snapshot(client, resolved.snapshot)
        sandbox = client.create(
            CreateSandboxFromSnapshotParams(
                snapshot=resolved.snapshot or RUNNER_SNAPSHOT,
                language="python",
                ephemeral=True,
                labels={
                    "project": "ghostdata",
                    "role": "analyst",
                    "filename": filename[:40],
                },
                auto_stop_interval=resolved.auto_stop_minutes,
                network_block_all=True,
                network_allow_list=resolved.network_allow_list,
                volumes=_volume_mounts(client, resolved),
            ),
            timeout=resolved.create_timeout_seconds,
        )
        sandbox_id = str(sandbox.id)
        on_event(
            {
                "kind": "analyst",
                "status": "running",
                "sandbox_id": sandbox_id,
                "text": f"ANALYST Daytona #{sandbox_id[:8]} inspecting {filename}",
            }
        )
        sandbox.process.exec(f"mkdir -p {shlex.quote(DAYTONA_ROOT)}/data {shlex.quote(DAYTONA_ROOT)}/worlds")
        sandbox.fs.upload_file(csv_bytes, f"{DAYTONA_ROOT}/data/dataset.csv")
        sandbox.fs.upload_file(prompt.encode(), f"{DAYTONA_ROOT}/task.md")
        sandbox.fs.upload_file(ANALYST_SCRIPT.read_bytes(), f"{DAYTONA_ROOT}/sandbox_analyst.py")
        _seed_volume(sandbox, resolved, {"dataset.csv": csv_bytes})
        maybe_inspect_table(sandbox, DAYTONA_ROOT)
        charts: list[dict[str, object]] = []
        session_id = maybe_create_session(sandbox, "analyst")
        try:
            response = exec_in_sandbox(
                sandbox,
                "python sandbox_analyst.py",
                cwd=DAYTONA_ROOT,
                timeout=resolved.command_timeout_seconds,
                session_id=session_id,
            )
            log = str(getattr(response, "result", "") or "")
            for line in log.splitlines():
                if line.strip():
                    on_event(
                        {
                            "kind": "analyst",
                            "status": "running",
                            "sandbox_id": sandbox_id,
                            "text": line.strip()[:240],
                        }
                    )
            if response.exit_code != 0:
                raise RuntimeError(f"analyst exited {response.exit_code}: {log.strip()}")
            code_run = getattr(sandbox.process, "code_run", None)
            if callable(code_run):
                chart = code_run(
                    ANALYST_CHART_CODE,
                    timeout=resolved.command_timeout_seconds,
                )
                if getattr(chart, "exit_code", 0) not in (0, None):
                    raise RuntimeError(
                        f"analyst code_run exited {chart.exit_code}: "
                        f"{str(getattr(chart, 'result', '')).strip()}"
                    )
                charts = serialize_code_run_charts(chart)
        finally:
            maybe_delete_session(sandbox, session_id)
        files = download_workspace(sandbox, DAYTONA_ROOT)
        metrics = _metrics_payload(sandbox)
        return {
            "sandbox_id": sandbox_id,
            "files": files,
            "metrics": metrics,
            "daytona_charts": charts,
        }
    finally:
        if sandbox is not None:
            client.delete(sandbox, wait=True)


def _run_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink,
    settings: DaytonaSettings | None = None,
) -> dict[str, Any]:
    from ghostdata.demo.codex_analyst import try_run_codex_analyst

    payload = try_run_codex_analyst(csv_bytes, prompt, filename, on_event)
    if payload is not None:
        return payload
    return _run_python_analyst(csv_bytes, prompt, filename, on_event, settings)


async def run_sandbox_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink,
    settings: DaytonaSettings | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _run_analyst, csv_bytes, prompt, filename, on_event, settings
    )
