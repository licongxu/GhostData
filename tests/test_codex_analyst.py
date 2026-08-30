import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghostdata.demo import codex_analyst


def _world_files() -> dict[str, str]:
    return {
        "analysis.json": json.dumps(
            {
                "label_column": "churned",
                "inspected_columns": ["churned", "tenure"],
                "hypotheses": [{"world_id": "W001", "title": "t", "hypothesis": "h"}],
            }
        ),
        "worlds/W001/transform.py": "def transform(dataframe):\n    return dataframe\n",
        "worlds/W001/hypothesis.json": json.dumps(
            {"world_id": "W001", "title": "t", "hypothesis": "h", "target_feature": "tenure"}
        ),
        "AGENTS.md": "rules",
        "task.md": "Predict churned",
    }


class FakeThread:
    def __init__(self, cwd: Path, files: dict[str, str] | None = None, result: object | None = None) -> None:
        self.cwd = cwd
        self.files = files if files is not None else _world_files()
        self.result = result or SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            error=None,
            final_response="wrote worlds",
        )

    def run(self, prompt: str, **kwargs: object) -> object:
        del kwargs, prompt
        for relative, contents in self.files.items():
            path = self.cwd / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        return self.result


class FakeCodex:
    def __init__(self, config: object = None, thread: FakeThread | None = None) -> None:
        self.config = config
        self._thread = thread
        self.started: dict[str, object] = {}

    def __enter__(self) -> "FakeCodex":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def account(self, refresh_token: bool = False) -> SimpleNamespace:
        del refresh_token
        return SimpleNamespace(account=object(), requires_openai_auth=False)

    def thread_start(self, **kwargs: object) -> FakeThread:
        self.started = kwargs
        cwd = Path(str(kwargs.get("cwd") or getattr(self.config, "cwd", ".")))
        if self._thread is not None:
            self._thread.cwd = cwd
            return self._thread
        return FakeThread(cwd)


def test_analyst_mode_defaults_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GHOSTDATA_ANALYST", raising=False)
    assert codex_analyst.analyst_mode() == "auto"
    monkeypatch.setenv("GHOSTDATA_ANALYST", "codex")
    assert codex_analyst.analyst_mode() == "codex"
    monkeypatch.setenv("GHOSTDATA_ANALYST", "nope")
    assert codex_analyst.analyst_mode() == "auto"


def test_collect_workspace_files_skips_data_dotfiles_and_binaries(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / ".secret").write_text("nope", encoding="utf-8")
    nested = tmp_path / "worlds" / ".cache"
    nested.mkdir(parents=True)
    (nested / "x").write_text("nope", encoding="utf-8")
    (tmp_path / "analysis.json").write_text("{}", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe")
    huge = tmp_path / "huge.txt"
    huge.write_bytes(b"a" * (codex_analyst.MAX_COLLECTED_FILE_BYTES + 1))
    files = codex_analyst.collect_workspace_files(tmp_path)
    assert files == {"analysis.json": "{}"}


def test_codex_available_handles_import_account_and_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import openai_codex

    monkeypatch.setitem(sys.modules, "openai_codex", None)
    assert codex_analyst.codex_available() is False
    monkeypatch.undo()

    class Ready:
        def __enter__(self) -> "Ready":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def account(self) -> SimpleNamespace:
            return SimpleNamespace(account=object(), requires_openai_auth=False)

    class NeedAuth:
        def __enter__(self) -> "NeedAuth":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def account(self) -> SimpleNamespace:
            return SimpleNamespace(account=None, requires_openai_auth=True)

    class Exploding:
        def __enter__(self) -> "Exploding":
            raise RuntimeError("down")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(openai_codex, "Codex", Ready)
    assert codex_analyst.codex_available() is True
    monkeypatch.setattr(openai_codex, "Codex", NeedAuth)
    assert codex_analyst.codex_available() is False
    monkeypatch.setattr(openai_codex, "Codex", Exploding)
    assert codex_analyst.codex_available() is False


def test_run_codex_analyst_writes_worlds(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai_codex

    monkeypatch.setenv("GHOSTDATA_ANALYST", "codex")
    monkeypatch.setattr(openai_codex, "Codex", FakeCodex)
    events: list[dict[str, object]] = []
    payload = codex_analyst.run_codex_analyst(
        b"churned,tenure\n0,1\n1,2\n",
        "Predict churned",
        "churn.csv",
        events.append,
    )
    assert payload["sandbox_id"] == "codex"
    assert payload["planner"] == "codex_sdk"
    assert "analysis.json" in payload["files"]
    assert "worlds/W001/transform.py" in payload["files"]
    assert "data/dataset.csv" not in payload["files"]
    assert any("CODEX inspecting" in str(event.get("text")) for event in events)


def test_run_codex_analyst_requires_analysis_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai_codex

    class Empty(FakeCodex):
        def thread_start(self, **kwargs: object) -> FakeThread:
            cwd = Path(str(kwargs.get("cwd")))
            return FakeThread(cwd, files={"task.md": "x"})

    monkeypatch.setattr(openai_codex, "Codex", Empty)
    with pytest.raises(RuntimeError, match="analysis.json"):
        codex_analyst.run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv")


def test_run_codex_analyst_failed_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai_codex

    class Failed(FakeCodex):
        def thread_start(self, **kwargs: object) -> FakeThread:
            cwd = Path(str(kwargs.get("cwd")))
            return FakeThread(
                cwd,
                files=_world_files(),
                result=SimpleNamespace(
                    status=SimpleNamespace(value="failed"),
                    error=SimpleNamespace(message="model overloaded"),
                    final_response=None,
                ),
            )

    monkeypatch.setattr(openai_codex, "Codex", Failed)
    with pytest.raises(RuntimeError, match="overloaded"):
        codex_analyst.run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv")

    class FailedSilent(FakeCodex):
        def thread_start(self, **kwargs: object) -> FakeThread:
            cwd = Path(str(kwargs.get("cwd")))
            return FakeThread(
                cwd,
                files=_world_files(),
                result=SimpleNamespace(status="failed", error=None, final_response=None),
            )

    monkeypatch.setattr(openai_codex, "Codex", FailedSilent)
    with pytest.raises(RuntimeError, match="Codex turn failed"):
        codex_analyst.run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv")


def test_try_run_codex_analyst_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOSTDATA_ANALYST", "deterministic")
    assert (
        codex_analyst.try_run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv") is None
    )

    monkeypatch.setenv("GHOSTDATA_ANALYST", "auto")
    monkeypatch.setattr(codex_analyst, "codex_available", lambda: False)
    assert (
        codex_analyst.try_run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv") is None
    )

    monkeypatch.setattr(codex_analyst, "codex_available", lambda: True)
    monkeypatch.setattr(
        codex_analyst,
        "run_codex_analyst",
        lambda *args, **kwargs: {"sandbox_id": "codex", "files": {"analysis.json": "{}"}},
    )
    assert codex_analyst.try_run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv")["sandbox_id"] == "codex"

    def boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("nope")

    monkeypatch.setattr(codex_analyst, "run_codex_analyst", boom)
    events: list[dict[str, object]] = []
    assert (
        codex_analyst.try_run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv", events.append)
        is None
    )
    assert any("falling back" in str(event.get("text")) for event in events)

    monkeypatch.setenv("GHOSTDATA_ANALYST", "codex")
    with pytest.raises(RuntimeError, match="nope"):
        codex_analyst.try_run_codex_analyst(b"a,b\n1,2\n", "task", "x.csv")


def test_emit_noop_without_sink() -> None:
    codex_analyst._emit(None, status="running", text="x")
