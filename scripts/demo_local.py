"""Run the measured credit verification demo on the local execution backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo import run_credit_demo  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_credit_demo("local").to_dict(), indent=2))
