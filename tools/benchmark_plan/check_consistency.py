"""Check benchmark compiler and validator contract constants."""

from __future__ import annotations

from tools.benchmark_plan import compiler, validation


def main() -> int:
    assert validation.PLAN_VERSION == compiler.PLAN_VERSION
    assert validation.EVENTS == compiler.EVENTS
    assert validation.MODES == compiler.MODES
    print("benchmark plan compiler/validation consistency: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
