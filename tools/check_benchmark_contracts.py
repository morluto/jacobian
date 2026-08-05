"""Validate every repository-owned benchmark schema and cross-file contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.benchmark_contracts import validate_all  # noqa: E402
from benchmarks.tooling.errors import HarborSuiteError  # noqa: E402


def main() -> int:
    try:
        failures = validate_all()
    except HarborSuiteError as exc:
        print(f"benchmark contract error: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("Benchmark contract failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Benchmark contracts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
