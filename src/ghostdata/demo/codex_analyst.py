"""Codex SDK proposer. Uses ChatGPT/Codex login credits, not the OpenAI API bill.

The model inspects the uploaded table and writes failure worlds. Daytona still
measures those worlds. The deterministic pandas script is only a fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

EventSink = Callable[[dict[str, Any]], None]

ANALYST_MODE_ENV = "GHOSTDATA_ANALYST"
VALID_ANALYST_MODES = frozenset({"auto", "codex", "deterministic"})
MAX_COLLECTED_FILE_BYTES = 1_000_000

ANALYST_INSTRUCTIONS = """
You are GhostData's failure-world proposer, not a data-analyst chatbot.

The user uploaded a CSV and described what the data is for.
Inspect the table by running Python in this workspace.

You MUST:
1. Identify the prediction target from the prompt and the columns.
2. Inspect schema, missingness, and feature–label associations by running code.
3. Fit a quick baseline only to learn which columns a model would rely on.
4. Propose 3 DISTINCT executable preprocessing failures.

Write ONLY these files. Do not write ghost_dataset.csv, model_report.json,
or regression_contract.py. Do not claim a world is a Ghost or that AUC dropped.

./analysis.json
{
  "label_column": "...",
  "inspected_columns": ["..."],
  "fragile_assumptions": ["..."],
  "hypotheses": [
    {"world_id":"W001","title":"...","hypothesis":"..."}
  ]
}

For each world:
./worlds/W00N/hypothesis.json
./worlds/W00N/transform.py

transform.py MUST define:

    def transform(dataframe):
        ...
        return dataframe

Rules:
- Deterministic (fixed seed). Do not modify the label column.
- Prefer relationship-breaking permutations that preserve schema, missingness, and marginals.
- Make the three worlds different: one aggressive, one weak, one subtle entity misalignment.
- pandas/numpy/sklearn are already installed. Do not pip install.
- When the files are written, stop. Do not write a long EDA essay as the answer.
""".strip()


def analyst_mode() -> str:
    value = os.environ.get(ANALYST_MODE_ENV, "auto").strip().lower()
    if value not in VALID_ANALYST_MODES:
        return "auto"
    return value


def _emit(on_event: EventSink | None, **event: Any) -> None:
    if on_event is None:
        return
    payload = {"kind": "analyst", **event}
    on_event(payload)


def collect_workspace_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("data/") or relative.startswith("."):
            continue
        if any(part.startswith(".") for part in Path(relative).parts):
            continue
        if path.stat().st_size > MAX_COLLECTED_FILE_BYTES:
            continue
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def _account_ready(info: Any) -> bool:
    if getattr(info, "requires_openai_auth", False) and getattr(info, "account", None) is None:
        return False
    return True


def codex_available() -> bool:
    try:
        from openai_codex import Codex
    except ImportError:
        return False
    try:
        with Codex() as client:
            return _account_ready(client.account())
    except Exception:
        return False


def run_codex_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink | None = None,
) -> dict[str, Any]:
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    import tempfile

    _emit(
        on_event,
        status="running",
        sandbox_id="codex",
        text=f"CODEX inspecting {filename} (ChatGPT credits)",
    )
    with tempfile.TemporaryDirectory(prefix="ghostdata-codex-") as temp:
        work = Path(temp)
        data_dir = work / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "dataset.csv").write_bytes(csv_bytes)
        (work / "task.md").write_text(prompt, encoding="utf-8")
        (work / "AGENTS.md").write_text(ANALYST_INSTRUCTIONS, encoding="utf-8")
        config = CodexConfig(
            cwd=str(work),
            config_overrides=(
                "approval_policy=never",
                "sandbox_mode=workspace-write",
            ),
        )
        user_prompt = (
            f"The uploaded file is {filename}.\n"
            f"User task:\n{prompt}\n\n"
            "Inspect data/dataset.csv by running Python. Follow AGENTS.md. "
            "Write analysis.json and three worlds under worlds/W00N/. "
            "Do not write ghost_dataset.csv, model_report.json, or "
            "regression_contract.py. Stop when those files exist."
        )
        with Codex(config) as client:
            thread = client.thread_start(
                cwd=str(work),
                sandbox=Sandbox.workspace_write,
                approval_mode=ApprovalMode.auto_review,
                developer_instructions=ANALYST_INSTRUCTIONS,
                ephemeral=True,
            )
            result = thread.run(user_prompt)
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        if status_value in {"failed", "error"} or getattr(result, "error", None):
            detail = getattr(getattr(result, "error", None), "message", None)
            raise RuntimeError(detail or f"Codex turn failed: {status_value}")
        files = collect_workspace_files(work)
        if "analysis.json" not in files:
            raise RuntimeError("Codex analyst did not write analysis.json")
        _emit(
            on_event,
            status="running",
            sandbox_id="codex",
            text=(getattr(result, "final_response", None) or "Codex wrote worlds")[:240],
        )
        return {
            "sandbox_id": "codex",
            "files": files,
            "metrics": None,
            "daytona_charts": [],
            "planner": "codex_sdk",
            "final_response": getattr(result, "final_response", None),
        }


def try_run_codex_analyst(
    csv_bytes: bytes,
    prompt: str,
    filename: str,
    on_event: EventSink | None = None,
) -> dict[str, Any] | None:
    mode = analyst_mode()
    if mode == "deterministic":
        return None
    if mode == "auto" and not codex_available():
        return None
    try:
        return run_codex_analyst(csv_bytes, prompt, filename, on_event)
    except Exception as exc:
        if mode == "codex":
            raise
        _emit(
            on_event,
            status="running",
            sandbox_id="",
            text=f"Codex unavailable, falling back ({type(exc).__name__})",
        )
        return None
