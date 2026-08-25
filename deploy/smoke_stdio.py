"""Exercise an installed Jacobian MCP entry point over the real stdio protocol.

This driver deliberately imports no Jacobian modules. Release checks run it
from outside the checkout with the command supplied after ``--`` so successful
execution proves the installed artifact, rather than a source tree or editable
install, owns the server under test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters, stdio_client

_STDERR_TAIL_BYTES = 8_192
_OPERATION_ID = "integer.compute.extended_gcd"


class SmokePhaseError(RuntimeError):
    """One named protocol phase failed or exceeded its deadline."""


async def _phase[T](
    name: str,
    timeout_seconds: float,
    call: Callable[[], Awaitable[T]],
) -> T:
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            return await call()
    except TimeoutError as exc:
        elapsed = time.monotonic() - started
        raise SmokePhaseError(
            f"MCP smoke phase {name!r} timed out after {elapsed:.3f}s "
            f"(limit {timeout_seconds:.3f}s)"
        ) from exc
    except Exception as exc:
        elapsed = time.monotonic() - started
        raise SmokePhaseError(
            f"MCP smoke phase {name!r} failed after {elapsed:.3f}s: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _require_structured(name: str, result: Any) -> dict[str, Any]:
    if result.is_error:
        raise RuntimeError(f"{name} returned an MCP tool error")
    if not isinstance(result.structured_content, dict):
        raise RuntimeError(f"{name} did not return structured content")
    return result.structured_content


async def inspect(
    *,
    command: list[str],
    expected_version: str,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    shutdown_timeout_seconds: float,
    cwd: Path,
    stderr: Any,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    parameters = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env=environment,
        cwd=cwd,
    )
    stack = AsyncExitStack()
    client: Client | None = None
    primary_failure: BaseException | None = None
    try:
        client = await _phase(
            "initialization",
            startup_timeout_seconds,
            lambda: stack.enter_async_context(
                Client(
                    stdio_client(parameters, errlog=stderr),
                    raise_exceptions=True,
                )
            ),
        )
        if client.server_info is None:
            raise SmokePhaseError("MCP initialization returned no server info")
        if client.server_info.version != expected_version:
            raise SmokePhaseError(
                "MCP initialization version mismatch: "
                f"expected {expected_version!r}, got {client.server_info.version!r}"
            )

        listed = await _phase(
            "tool discovery",
            request_timeout_seconds,
            client.list_tools,
        )
        tool_names = {tool.name for tool in listed.tools}
        required_tools = {"math.find", "math.run"}
        if tool_names != required_tools:
            raise SmokePhaseError(
                f"MCP tool surface mismatch: expected {sorted(required_tools)!r}, "
                f"got {sorted(tool_names)!r}"
            )

        search = _require_structured(
            "math.find search",
            await _phase(
                "math.find search",
                request_timeout_seconds,
                lambda: client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "search",
                            "query": "integer greatest common divisor Bezout coefficients",
                            "limit": 5,
                        }
                    },
                ),
            ),
        )
        inspection = _require_structured(
            "math.find inspect",
            await _phase(
                "math.find inspect",
                request_timeout_seconds,
                lambda: client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "inspect",
                            "operation_id": _OPERATION_ID,
                        }
                    },
                ),
            ),
        )
        if inspection.get("operation", {}).get("operation_id") != _OPERATION_ID:
            raise SmokePhaseError("math.find inspection returned the wrong operation")

        first_run = _require_structured(
            "math.run",
            await _phase(
                "math.run",
                request_timeout_seconds,
                lambda: client.call_tool(
                    "math.run",
                    {
                        "operation_id": _OPERATION_ID,
                        "payload": {"left": "84", "right": "30"},
                    },
                ),
            ),
        )
        if first_run.get("output", {}).get("gcd") != "6":
            raise SmokePhaseError("math.run did not return gcd 6")

        second_run = _require_structured(
            "second math.run",
            await _phase(
                "second math.run",
                request_timeout_seconds,
                lambda: client.call_tool(
                    "math.run",
                    {
                        "operation_id": _OPERATION_ID,
                        "payload": {"left": "21", "right": "14"},
                    },
                ),
            ),
        )
        if second_run.get("output", {}).get("gcd") != "7":
            raise SmokePhaseError("second math.run did not return gcd 7")

        return {
            "transport": "stdio",
            "command": command,
            "server": {
                "name": client.server_info.name,
                "version": client.server_info.version,
            },
            "tool_names": sorted(tool_names),
            "search_matches": [
                match["operation_id"] for match in search.get("matches", [])
            ],
            "inspected_operation": _OPERATION_ID,
            "runs": [
                {"left": "84", "right": "30", "gcd": "6"},
                {"left": "21", "right": "14", "gcd": "7"},
            ],
        }
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        try:
            await _phase("shutdown", shutdown_timeout_seconds, stack.aclose)
        except BaseException:
            if primary_failure is None:
                raise


def _stderr_tail(stderr: Any) -> str:
    stderr.flush()
    stderr.seek(0, os.SEEK_END)
    size = stderr.tell()
    stderr.seek(max(0, size - _STDERR_TAIL_BYTES))
    return stderr.read()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke an installed Jacobian MCP command over stdio.",
    )
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=60)
    parser.add_argument("--request-timeout-seconds", type=float, default=20)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


async def _main() -> None:
    args = _parser().parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("an installed MCP command is required after --")
    for name in (
        "startup_timeout_seconds",
        "request_timeout_seconds",
        "shutdown_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    with (
        tempfile.TemporaryDirectory(prefix="jacobian-packaged-mcp-") as directory,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr,
    ):
        try:
            report = await inspect(
                command=command,
                expected_version=args.expect_version,
                startup_timeout_seconds=args.startup_timeout_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
                shutdown_timeout_seconds=args.shutdown_timeout_seconds,
                cwd=Path(directory),
                stderr=stderr,
            )
        except Exception as exc:
            tail = _stderr_tail(stderr)
            diagnostic = f"packaged MCP smoke failed: {exc}"
            if tail:
                diagnostic += f"\nserver stderr tail:\n{tail}"
            raise SystemExit(diagnostic) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
