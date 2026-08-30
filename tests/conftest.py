import pytest


@pytest.fixture(autouse=True)
def _demo_analyst_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOSTDATA_ANALYST", "deterministic")
