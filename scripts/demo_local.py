"""Run the measured table verification demo on the local execution backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo.credit import DEFAULT_DATA_PATH, TARGET_COLUMN  # noqa: E402
from ghostdata.demo.table import run_table_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--label-column", default=TARGET_COLUMN)
    args = parser.parse_args()
    report, spec, analysis = run_table_demo(args.csv, args.label_column, "local")
    print(
        json.dumps(
            {
                "verdict": report.verdict,
                "winning_spec": spec.to_dict(),
                "inspected_columns": analysis.get("inspected_columns"),
                "hypotheses": analysis.get("hypotheses"),
                "report": report.to_dict(),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
