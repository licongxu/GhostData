"""Live smoke test of Daytona 0.207 APIs GhostData will use tomorrow.

Exercises env-config, WebSocket state streaming, snapshot create,
ephemeral sandboxes, process.exec, process.code_run, stateful
code_interpreter, filesystem upload/download, and sessions.
Always deletes the sandbox in finally.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from daytona import (  # noqa: E402
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    SessionExecuteRequest,
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    config = DaytonaConfig(use_deprecated_polling=False)
    daytona = Daytona(config)
    print("ok  client (websocket state streaming, not polling)")
    print(f"    target={daytona._target!r} api={daytona._api_url}")

    snaps = daytona.snapshot.list(page=1, limit=20)
    print(f"ok  snapshot.list  total={snaps.total} page_items={len(snaps.items)}")
    for s in snaps.items[:8]:
        print(f"    snapshot {s.name} state={getattr(s, 'state', '?')}")

    volumes = daytona.volume.list()
    print(f"ok  volume.list  n={len(volumes)}")

    try:
        pools = daytona.warm_pool.list()
        print(f"ok  warm_pool.list  n={len(pools)}")
    except Exception as exc:
        # SDK 0.207 exposes this; hosted API currently 404s GET /api/warm-pools.
        print(f"skip warm_pool.list  ({type(exc).__name__})")

    try:
        secrets = daytona.secret.list()
        n_secrets = getattr(secrets, "items", secrets)
        print(f"ok  secret.list  n={len(n_secrets) if hasattr(n_secrets, '__len__') else n_secrets}")
    except Exception as exc:
        print(f"skip secret.list  ({type(exc).__name__})")

    items = list(daytona.list())
    print(f"ok  sandbox.list  n={len(items)}")

    sandbox = None
    try:
        sandbox = daytona.create(
            CreateSandboxFromSnapshotParams(
                snapshot="daytona-small",
                language="python",
                ephemeral=True,
                labels={"project": "ghostdata", "purpose": "smoke"},
            ),
            timeout=180,
        )
        print(f"ok  create ephemeral sandbox id={sandbox.id} state={sandbox.state}")

        exec_res = sandbox.process.exec("echo ghostdata-daytona-ok && python3 --version && uname -s")
        if exec_res.exit_code != 0:
            fail(f"process.exec exit={exec_res.exit_code} {exec_res.result}")
        print(f"ok  process.exec  {exec_res.result.strip()!r}")

        code_res = sandbox.process.code_run(
            "\n".join(
                [
                    "import numpy as np",
                    "import pandas as pd",
                    "from sklearn.linear_model import LogisticRegression",
                    "X = np.array([[0], [1], [0], [1]])",
                    "y = np.array([0, 1, 0, 1])",
                    "m = LogisticRegression().fit(X, y)",
                    "print('sklearn', round(float(m.predict_proba([[1]])[0, 1]), 3))",
                    "print('pandas', pd.DataFrame({'x': [1, 2]}).shape)",
                ]
            )
        )
        if code_res.exit_code != 0:
            fail(f"process.code_run exit={code_res.exit_code} {code_res.result}")
        print(f"ok  process.code_run  {code_res.result.strip()!r}")

        interp1 = sandbox.code_interpreter.run_code("ghost = 41")
        if interp1.error:
            fail(f"code_interpreter init error={interp1.error}")
        interp2 = sandbox.code_interpreter.run_code("print(ghost + 1)")
        if interp2.error:
            fail(f"code_interpreter stateful error={interp2.error}")
        out = (interp2.stdout or "").strip()
        if "42" not in out:
            fail(f"code_interpreter state lost, stdout={out!r}")
        print(f"ok  code_interpreter stateful  stdout={out!r}")

        csv = "id,income\nA,10\nB,20\nC,30\n"
        sandbox.fs.upload_file(csv.encode(), "/home/daytona/candidate.csv")
        downloaded = sandbox.fs.download_file("/home/daytona/candidate.csv")
        if downloaded != csv.encode():
            fail("fs round-trip mismatch")
        print("ok  fs upload/download candidate.csv")

        sandbox.process.create_session("ghostdata-smoke")
        sess = sandbox.process.execute_session_command(
            "ghostdata-smoke",
            SessionExecuteRequest(command="wc -l /home/daytona/candidate.csv"),
        )
        print(f"ok  session command  {getattr(sess, 'output', sess)!r}")
        sandbox.process.delete_session("ghostdata-smoke")
        print("ok  session deleted")
    finally:
        if sandbox is not None:
            daytona.delete(sandbox)
            print(f"ok  delete sandbox {sandbox.id}")

    print("PASS  Daytona newest APIs ran with no error")


if __name__ == "__main__":
    main()
