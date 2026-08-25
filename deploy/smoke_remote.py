"""Read-only smoke check for a deployed Jacobian Streamable HTTP endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx2
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation, TextContent, TextResourceContents

from jacobian import __version__
from jacobian.canonical import canonicalize_json
from mcp import Client

from .smoke import (
    TransientSmokeError,
    exit_for_smoke_failure,
    is_transient_transport_failure,
    raise_for_http_error,
)

REQUIRED_TOOLS = {
    "math.find",
    "math.run",
}
DISCOVERY_RESPONSE_BYTE_LIMIT = 16_384
_OPERATION_ID = "integer.compute.extended_gcd"


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
        raise TransientSmokeError(
            f"remote MCP smoke phase {name!r} timed out after {elapsed:.3f}s "
            f"(limit {timeout_seconds:.3f}s)"
        ) from exc
    except Exception as exc:
        elapsed = time.monotonic() - started
        failure_type = (
            TransientSmokeError if is_transient_transport_failure(exc) else RuntimeError
        )
        raise failure_type(
            f"remote MCP smoke phase {name!r} failed after {elapsed:.3f}s: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _require_server_info(server_info: Implementation | None) -> Implementation:
    if server_info is None:
        raise RuntimeError("deployed MCP server did not provide implementation info")
    return server_info


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the deployed MCP handshake, version, tool surface, catalog, "
            "and bounded discovery response without mutating server state."
        )
    )
    parser.add_argument("url", help="public or localhost MCP URL ending in /mcp")
    parser.add_argument(
        "--expect-version",
        default=__version__,
        help="required MCP server version; defaults to this checkout's package version",
    )
    parser.add_argument(
        "--require-operation",
        action="append",
        default=[],
        help="operation ID that must be installed; repeatable",
    )
    parser.add_argument(
        "--query",
        default="exact finite graph invariant",
        help="read-only operation discovery query",
    )
    parser.add_argument(
        "--expect-session-mode",
        choices=("stateless", "stateful"),
        default="stateless",
        help="required Streamable HTTP session mode; defaults to stateless",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120)
    return parser


def _headers() -> dict[str, str] | None:
    token = os.environ.get("JACOBIAN_MCP_BEARER_TOKEN")
    token_file = os.environ.get("JACOBIAN_MCP_AUTH_TOKENS_FILE")
    if token is None and token_file:
        from jacobian.mcp.remote import load_static_token_file

        grants = load_static_token_file(token_file)
        token = next(
            (grant.token for grant in grants if "jacobian:use" in grant.scopes),
            None,
        )
        if token is None:
            raise RuntimeError("smoke token file has no jacobian:use grant")
    return {"Authorization": f"Bearer {token}"} if token else None


def _validate_tool_surface(listed: Any, failures: list[str]) -> set[str]:
    tool_names = {tool.name for tool in listed.tools}
    missing = sorted(REQUIRED_TOOLS - tool_names)
    unexpected = sorted(tool_names - REQUIRED_TOOLS)
    if missing:
        failures.append(
            f"deployed MCP tool surface is missing required tools: {missing!r}"
        )
    if unexpected:
        failures.append(
            f"deployed MCP tool surface has unexpected tools: {unexpected!r}"
        )
    return tool_names


def _validate_server_version(
    actual: str,
    expected: str,
    failures: list[str],
) -> None:
    if actual != expected:
        failures.append(
            f"deployed MCP version mismatch: expected {expected!r}, got {actual!r}"
        )


async def _inspect_catalog(
    client: Client,
    *,
    required_operations: set[str],
    timeout_seconds: float,
    failures: list[str],
) -> dict[str, Any]:
    catalog_result = await _phase(
        "operation catalog",
        timeout_seconds,
        lambda: client.read_resource("operation://catalog"),
    )
    catalog_content = catalog_result.contents[0]
    if not isinstance(catalog_content, TextResourceContents):
        raise RuntimeError("deployed operation catalog is not text")
    catalog_text = catalog_content.text
    catalog = json.loads(catalog_text)
    operation_ids = {operation["operation_id"] for operation in catalog["operations"]}
    missing = sorted(required_operations - operation_ids)
    if missing:
        failures.append(f"deployed catalog is missing required operations: {missing!r}")
    catalog_digest = hashlib.sha256(
        canonicalize_json(
            {
                "operations": catalog["operations"],
            }
        )
    ).hexdigest()
    return {
        "operations": len(operation_ids),
        "catalog_digest": f"sha256:{catalog_digest}",
        "sha256": hashlib.sha256(catalog_text.encode("utf-8")).hexdigest(),
    }


async def _inspect_discovery(
    client: Client,
    *,
    query: str,
    timeout_seconds: float,
    failures: list[str],
) -> dict[str, Any]:
    result = await _phase(
        "math.find search",
        timeout_seconds,
        lambda: client.call_tool(
            "math.find",
            {"request": {"op": "search", "query": query, "limit": 5}},
        ),
    )
    if result.is_error:
        failures.append("deployed operation discovery returned an MCP error")
    content = result.content[0]
    if not isinstance(content, TextContent):
        raise RuntimeError("deployed operation discovery is not text")
    if not isinstance(result.structured_content, dict):
        raise RuntimeError("deployed operation discovery is not structured")
    discovery = result.structured_content
    discovery_bytes = len(
        json.dumps(discovery, ensure_ascii=False, indent=2).encode("utf-8")
    )
    if discovery["response_byte_limit"] != DISCOVERY_RESPONSE_BYTE_LIMIT:
        failures.append("deployed discovery byte limit does not match the contract")
    if discovery_bytes > DISCOVERY_RESPONSE_BYTE_LIMIT:
        failures.append("deployed discovery response exceeds its byte limit")
    return {
        "bytes": discovery_bytes,
        "model_visible_bytes": len(content.text.encode("utf-8")),
        "matches": [match["operation_id"] for match in discovery["matches"]],
    }


async def _inspect_execution(
    client: Client,
    *,
    timeout_seconds: float,
    failures: list[str],
) -> dict[str, str]:
    inspection_result = await _phase(
        "math.find inspect",
        timeout_seconds,
        lambda: client.call_tool(
            "math.find",
            {"request": {"op": "inspect", "operation_id": _OPERATION_ID}},
        ),
    )
    if not isinstance(inspection_result.structured_content, dict):
        raise RuntimeError("deployed operation inspection is not structured")
    if (
        inspection_result.structured_content.get("operation", {}).get("operation_id")
        != _OPERATION_ID
    ):
        failures.append("deployed operation inspection returned the wrong operation")

    async def run(left: str, right: str, phase: str) -> str:
        result = await _phase(
            phase,
            timeout_seconds,
            lambda: client.call_tool(
                "math.run",
                {
                    "operation_id": _OPERATION_ID,
                    "payload": {"left": left, "right": right},
                },
            ),
        )
        if not isinstance(result.structured_content, dict):
            raise RuntimeError(f"deployed {phase} result is not structured")
        return str(result.structured_content.get("output", {}).get("gcd"))

    gcd = await run("84", "30", "math.run")
    second_gcd = await run("21", "14", "second math.run")
    if gcd != "6":
        failures.append("deployed math.run did not return gcd 6")
    if second_gcd != "7":
        failures.append("deployed second math.run did not return gcd 7")
    return {"operation_id": _OPERATION_ID, "gcd": gcd, "second_gcd": second_gcd}


async def inspect(
    *,
    url: str,
    expected_version: str,
    expected_revision: str | None = None,
    required_operations: set[str],
    query: str,
    timeout_seconds: float,
    expected_session_mode: str = "stateless",
) -> dict[str, Any]:
    headers = _headers()
    failures: list[str] = []
    session_ids: set[str] = set()

    async def inspect_http_response(response: httpx2.Response) -> None:
        await raise_for_http_error(response)
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            session_ids.add(session_id)

    stack = AsyncExitStack()
    primary_failure: BaseException | None = None
    try:
        http = await stack.enter_async_context(
            httpx2.AsyncClient(
                headers=headers,
                event_hooks={"response": [inspect_http_response]},
                trust_env=False,
                timeout=timeout_seconds,
            )
        )
        client = await _phase(
            "initialization",
            timeout_seconds,
            lambda: stack.enter_async_context(
                Client(
                    streamable_http_client(url, http_client=http),
                    raise_exceptions=True,
                )
            ),
        )
        server_info = _require_server_info(client.server_info)
        server_version = server_info.version
        _validate_server_version(server_version, expected_version, failures)

        listed = await _phase("tool discovery", timeout_seconds, client.list_tools)
        tool_names = _validate_tool_surface(listed, failures)

        catalog_report = await _inspect_catalog(
            client,
            required_operations=required_operations,
            timeout_seconds=timeout_seconds,
            failures=failures,
        )
        discovery_report = await _inspect_discovery(
            client,
            query=query,
            timeout_seconds=timeout_seconds,
            failures=failures,
        )
        execution_report = await _inspect_execution(
            client,
            timeout_seconds=timeout_seconds,
            failures=failures,
        )

        observed_session_mode = "stateful" if session_ids else "stateless"
        if observed_session_mode != expected_session_mode:
            failures.append(
                "deployed MCP session mode mismatch: "
                f"expected {expected_session_mode}, observed {observed_session_mode}"
            )

        report = {
            "url": url,
            "session_mode": observed_session_mode,
            "server": {
                "name": server_info.name,
                "version": server_version,
            },
            "tool_names": sorted(tool_names),
            "catalog": catalog_report,
            "discovery": discovery_report,
            "execution": execution_report,
        }
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        try:
            await _phase("shutdown", timeout_seconds, stack.aclose)
        except BaseException:
            if primary_failure is None:
                raise
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


async def _main() -> None:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    report = await inspect(
        url=args.url,
        expected_version=args.expect_version,
        required_operations=set(args.require_operation),
        query=args.query,
        timeout_seconds=args.timeout_seconds,
        expected_session_mode=args.expect_session_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except Exception as exc:
        exit_for_smoke_failure("smoke", exc)
