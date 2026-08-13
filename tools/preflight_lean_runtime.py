#!/usr/bin/env python3
"""Fail closed when a hosted Lean lane cannot measure its required runtimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jacobian.contracts.capabilities import (  # noqa: E402
    CapabilityProviderAvailability,
)
from jacobian.providers.lean_runtime import (  # noqa: E402
    lean_frontend_provider_runtime,
    lean_provider_runtime,
)


def _report(name: str, runtime: object) -> bool:
    availability = runtime.availability
    diagnostic = getattr(runtime, "diagnostic", None) or ""
    print(f"{name}: {availability.value}")
    if diagnostic:
        print(f"  diagnostic: {diagnostic}")
    configuration = getattr(runtime, "configuration", None)
    if isinstance(configuration, dict):
        semantic = configuration.get("semantic_runtime")
        if isinstance(semantic, dict) and "digest" in semantic:
            print(f"  semantic digest: {semantic['digest']}")
    return availability is CapabilityProviderAvailability.AVAILABLE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required",
        action="store_true",
        help="Exit nonzero when CORE or MATHLIB is unavailable.",
    )
    args = parser.parse_args()
    core = lean_frontend_provider_runtime()
    mathlib = lean_provider_runtime(
        profiles={"mathlib": {"mathlib_commit": "pinned"}},
        checker_ids=(),
    )
    core_ok = _report("CORE", core)
    mathlib_ok = _report("MATHLIB", mathlib)
    if args.required and not (core_ok and mathlib_ok):
        print(
            "hosted Lean preflight failed: required CORE and MATHLIB runtimes "
            "must be AVAILABLE",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
