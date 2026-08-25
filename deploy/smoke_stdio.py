"""Read-only MCP smoke journey against an installed stdio server executable."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise an installed Jacobian MCP stdio entry point."
    )
    parser.add_argument("server", type=Path, help="installed jacobian-mcp executable")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    return parser


async def inspect(server: Path, *, timeout_seconds: float) -> None:
    from mcp import Client, StdioServerParameters, stdio_client

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    parameters = StdioServerParameters(
        command=str(server),
        env=environment,
        cwd=server.parent,
    )
    async with Client(stdio_client(parameters), raise_exceptions=True) as client:
        listed = await asyncio.wait_for(client.list_tools(), timeout_seconds)
        if {tool.name for tool in listed.tools} != {"math.find", "math.run"}:
            raise RuntimeError(
                "installed MCP server exposed an unexpected tool surface"
            )
        described = await asyncio.wait_for(
            client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "integer.compute.extended_gcd",
                    }
                },
            ),
            timeout_seconds,
        )
        if not isinstance(described.structured_content, dict):
            raise RuntimeError("math.find inspection was not structured")
        for left, right, expected in (("84", "30", "6"), ("35", "14", "7")):
            result = await asyncio.wait_for(
                client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.extended_gcd",
                        "payload": {"left": left, "right": right},
                    },
                ),
                timeout_seconds,
            )
            if (
                not isinstance(result.structured_content, dict)
                or result.structured_content.get("output", {}).get("gcd") != expected
            ):
                raise RuntimeError(
                    "installed MCP server returned an unexpected gcd result"
                )


def main() -> None:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    asyncio.run(inspect(args.server.resolve(), timeout_seconds=args.timeout_seconds))


if __name__ == "__main__":
    main()
