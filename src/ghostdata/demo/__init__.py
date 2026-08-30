"""Runnable demonstrations built on the public GhostData contracts."""

from ghostdata.demo.credit import (
    DEFAULT_DATA_PATH,
    PreparedCreditDemo,
    prepare_credit_demo,
    run_credit_demo,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "PreparedCreditDemo",
    "prepare_credit_demo",
    "run_credit_demo",
]
