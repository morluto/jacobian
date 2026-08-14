"""One-shot Lean source smoke for a deployed fixed toolchain."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from jacobian._deployment_smoke import (
    TransientSmokeError,
    exit_for_smoke_failure,
    raise_for_http_error,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check accepted and rejected bounded Lean source snippets through "
            "the deployed one-shot lean.check operation."
        )
    )
    parser.add_argument("url", help="public or localhost MCP URL ending in /mcp")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def _token() -> str | None:
    token = os.environ.get("JACOBIAN_MCP_BEARER_TOKEN")
    token_file = os.environ.get("JACOBIAN_MCP_AUTH_TOKENS_FILE")
    if token is None and token_file:
        from jacobian.adapters.mcp.remote import load_static_token_file

        grants = load_static_token_file(token_file)
        token = next(
            (grant.token for grant in grants if "jacobian:use" in grant.scopes),
            None,
        )
        if token is None:
            raise RuntimeError("smoke token file has no jacobian:use grant")
    return token


async def _run(
    client: Any, operation_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    response = await client.call_tool(
        "math.run",
        {"operation_id": operation_id, "payload": payload},
    )
    if response.is_error or not isinstance(response.structured_content, dict):
        raise RuntimeError(f"{operation_id} did not return a structured result")
    return response.structured_content


def _require_completed(result: dict[str, Any], *, operation: str) -> None:
    status = result.get("execution", {}).get("status")
    if status in {"TIMEOUT", "CANCELLED"}:
        raise TransientSmokeError(f"{operation} ended with transient status {status}")
    if status != "COMPLETED":
        raise RuntimeError(f"{operation} did not complete")


def _require_outcome(
    result: dict[str, Any], *, expected: str, require_diagnostics: bool = False
) -> None:
    _require_completed(result, operation="lean.check")
    output = result.get("output", {})
    value = output.get("result") if isinstance(output, dict) else None
    if not isinstance(value, dict) or value.get("outcome") != expected:
        raise RuntimeError(f"lean.check did not return {expected}")
    diagnostics = value.get("diagnostics")
    if require_diagnostics and (not isinstance(diagnostics, list) or not diagnostics):
        raise RuntimeError("rejected Lean source returned no typed diagnostics")


async def inspect(*, url: str, timeout_seconds: float) -> dict[str, Any]:
    token = _token()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with (
        httpx2.AsyncClient(
            headers=headers,
            event_hooks={"response": [raise_for_http_error]},
            trust_env=False,
            timeout=timeout_seconds,
        ) as http,
        Client(
            streamable_http_client(url, http_client=http),
            raise_exceptions=True,
        ) as client,
    ):
        accepted = await _run(
            client,
            "lean.check",
            {"source": "example : True := by trivial"},
        )
        _require_outcome(accepted, expected="ELABORATED")
        rejected = await _run(
            client,
            "lean.check",
            {"source": "example : 1 = 2 := by rfl"},
        )
        _require_outcome(rejected, expected="REJECTED", require_diagnostics=True)

    return {
        "url": url,
        "checks": {
            "accepted_source": "ELABORATED",
            "rejected_source": "REJECTED",
        },
    }


async def _main() -> None:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    report = await inspect(url=args.url, timeout_seconds=args.timeout_seconds)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except Exception as exc:
        exit_for_smoke_failure("Lean smoke", exc)
