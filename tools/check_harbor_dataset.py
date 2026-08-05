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
    check_selected_tasks,
    check_suite,
    get_suite,
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


def check_selected(dataset: str, tasks: tuple[str, ...]) -> int:
    """Run the leaf-only topology, digest, and verifier-support gate."""

    suite = get_suite(dataset)
    failures = check_selected_tasks(suite, tasks)
    if report_failures(
        failures,
        header=f"Harbor selected-task contract failures for {suite.id}",
    ):
        return 1
    report_ok(
        f"Harbor selected-task contracts match for {suite.id}: " + ", ".join(tasks)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify contracts")
    parser.add_argument("--dataset", help="select one dataset for the leaf gate")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="select one or more task IDs for the leaf gate",
    )
    args = parser.parse_args()
    if (args.dataset is None) != (args.tasks is None):
        parser.error("--dataset and --tasks must be provided together")
    try:
        if args.dataset is not None and args.tasks is not None:
            return check_selected(args.dataset, tuple(args.tasks))
        return check()
    except HarborSuiteError as exc:
        print(f"harbor dataset check error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
