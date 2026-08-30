"""Run the multi-agent Ghost discovery from any working directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo.credit import DEFAULT_DATA_PATH  # noqa: E402
from ghostdata.demo.discovery import (  # noqa: E402
    DEFAULT_AGENT_PROFILES,
    DEFAULT_OUTPUT_ROOT,
    FULL_DATA_PATH,
    run_credit_discovery,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "daytona"), default="local")
    parser.add_argument("--dataset", choices=("full", "debug"), default="full")
    parser.add_argument(
        "--agents", type=int, choices=range(1, len(DEFAULT_AGENT_PROFILES) + 1), default=4
    )
    parser.add_argument("--discovery-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    data_path = FULL_DATA_PATH if args.dataset == "full" else DEFAULT_DATA_PATH
    report = run_credit_discovery(
        backend=args.backend,
        data_path=data_path,
        output_root=args.output_root,
        profiles=DEFAULT_AGENT_PROFILES[: args.agents],
        discovery_id=args.discovery_id,
    )
    report["artifact_directory"] = str(
        args.output_root.resolve() / str(report["discovery_id"])
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
