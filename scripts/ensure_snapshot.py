"""Build or reuse the ghostdata-runner Daytona snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from daytona import Daytona, DaytonaConfig  # noqa: E402

from ghostdata.execution.daytona import (  # noqa: E402
    RUNNER_SNAPSHOT,
    ensure_runner_snapshot,
)


def main() -> None:
    client = Daytona(DaytonaConfig(use_deprecated_polling=False))
    name = ensure_runner_snapshot(client, RUNNER_SNAPSHOT)
    snapshot = client.snapshot.get(name)
    print(f"snapshot={name} state={getattr(snapshot, 'state', '?')}")


if __name__ == "__main__":
    main()
