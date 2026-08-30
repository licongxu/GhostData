"""Run Ghost discovery from any working directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo.credit import DEFAULT_DATA_PATH, TARGET_COLUMN  # noqa: E402
from ghostdata.demo.discovery import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    FULL_DATA_PATH,
    run_table_discovery,
)
from ghostdata.execution.daytona import leftover_sandboxes  # noqa: E402


APPROVAL_PATH = ROOT / "data" / "live" / "credit_approval.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "daytona"), default="local")
    parser.add_argument(
        "--dataset",
        choices=("full", "debug", "approval"),
        default="debug",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label-column")
    parser.add_argument("--max-specs", type=int, default=4)
    parser.add_argument("--discovery-id")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    fixtures = {
        "full": (FULL_DATA_PATH, TARGET_COLUMN),
        "debug": (DEFAULT_DATA_PATH, TARGET_COLUMN),
        "approval": (APPROVAL_PATH, "class"),
    }
    if args.csv is not None:
        if not args.label_column:
            parser.error("--label-column is required with --csv")
        data_path, label_column = args.csv, args.label_column
    else:
        data_path, label_column = fixtures[args.dataset]
        if args.label_column:
            label_column = args.label_column
    report = run_table_discovery(
        data_path,
        label_column,
        backend=args.backend,
        output_root=args.output_root,
        discovery_id=args.discovery_id,
        max_specs=args.max_specs,
    )
    leftover = None
    if args.backend == "daytona":
        from daytona import Daytona, DaytonaConfig
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        leftover = len(
            leftover_sandboxes(Daytona(DaytonaConfig(use_deprecated_polling=False)))
        )
        report["leftover_sandboxes"] = leftover
    report["artifact_directory"] = str(
        args.output_root.resolve() / str(report["discovery_id"])
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
