"""CSV + prompt red-team on Daytona. Control plane only; sandboxes do the work."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from ghostdata.demo.redteam import get_run, start_run  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    run_id = start_run(args.csv.read_bytes(), args.prompt, args.csv.name, backend="daytona")
    print(json.dumps({"run_id": run_id, "status": "running"}))
    while True:
        payload = get_run(run_id)
        for event in payload.get("events") or []:
            text = event.get("text")
            if text:
                print(text)
        if payload["status"] != "running":
            print(json.dumps(payload.get("report") or payload, default=str)[:4000])
            if payload["status"] == "failed":
                sys.exit(1)
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
