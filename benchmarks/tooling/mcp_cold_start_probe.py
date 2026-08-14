"""One isolated MCP initialize probe against an already compiled state."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FORBIDDEN_STARTUP_MODULES = (
    "jacobian.catalog.build",
    "jacobian.catalog.compiler",
    "jacobian.provider_inventory",
)
_FORBIDDEN_STARTUP_PREFIXES = ("jacobian.providers.",)
_FORBIDDEN_CALLS = {
    ("jacobian.checker_identity", "batch_checker_manifest_measurement"),
    ("jacobian.checker_identity", "checker_manifest_measurement"),
    ("jacobian.catalog.build", "build_catalog_operations"),
    ("jacobian.catalog.compiler", "compile_operation_catalog"),
}


async def _initialize(state_dir: Path) -> int:
    from mcp import Client

    from jacobian.adapters.mcp.server import create_server

    async with Client(create_server(state_dir), raise_exceptions=True) as client:
        listed = await client.list_tools()
        return len(listed.tools)


def run_probe(state_dir: Path, *, audit: bool = False) -> dict[str, object]:
    forbidden_calls: set[str] = set()

    def profile(
        frame: FrameType, event: str, _arg: object
    ) -> Callable[..., object] | None:
        if event == "call":
            identity = (str(frame.f_globals.get("__name__")), frame.f_code.co_name)
            if identity in _FORBIDDEN_CALLS:
                forbidden_calls.add(":".join(identity))
        return profile

    start = time.perf_counter()
    if audit:
        sys.setprofile(profile)
    try:
        tool_count = asyncio.run(_initialize(state_dir))
    finally:
        if audit:
            sys.setprofile(None)
    elapsed = time.perf_counter() - start
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name in _FORBIDDEN_STARTUP_MODULES
        or name.startswith(_FORBIDDEN_STARTUP_PREFIXES)
    )
    return {
        "elapsed_seconds": elapsed,
        "forbidden_startup_calls": sorted(forbidden_calls),
        "forbidden_startup_modules": forbidden_modules,
        "tool_count": tool_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_probe(args.state_dir, audit=args.audit), sort_keys=True))


if __name__ == "__main__":
    main()
