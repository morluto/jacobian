"""Run the deterministic pre-container gate for every Harbor dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.harbor_suite import (  # noqa: E402
    HarborSuiteError,
    check_suite,
    load_registry,
    report_failures,
    report_ok,
)


def check() -> int:
    failures: list[str] = []
    checked = 0
    for suite in load_registry():
        checked += 1
        failures.extend(check_suite(suite))
    if report_failures(failures, header="Harbor dataset contract failures"):
        return 1
    report_ok(f"Harbor dataset contracts match for {checked} dataset(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify contracts")
    parser.parse_args()
    try:
        return check()
    except HarborSuiteError as exc:
        print(f"harbor dataset check error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
