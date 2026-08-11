"""Read-only smoke check for a deployed Jacobian Streamable HTTP endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import TextContent, TextResourceContents

from jacobian import __version__
from jacobian.canonical import canonicalize_json

REQUIRED_TOOLS = {
    "math.find",
    "math.run",
}
DISCOVERY_RESPONSE_BYTE_LIMIT = 16_384


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
        "--expect-policy-profile",
        default="DEFAULT",
        help="required catalog policy profile",
    )
    parser.add_argument(
        "--require-capability",
        action="append",
        default=[],
        help="capability ID that must be installed; repeatable",
    )
    parser.add_argument(
        "--query",
        default="exact finite graph invariant",
        help="read-only capability discovery query",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120)
    return parser


async def inspect(
    *,
    url: str,
    expected_version: str,
    expected_policy_profile: str,
    required_capabilities: set[str],
    query: str,
    timeout_seconds: float,
) -> dict[str, Any]:
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
    headers = {"Authorization": f"Bearer {token}"} if token else None
    failures: list[str] = []
    async with (
        httpx2.AsyncClient(
            headers=headers,
            trust_env=False,
            timeout=timeout_seconds,
        ) as http,
        Client(
            streamable_http_client(url, http_client=http),
            raise_exceptions=True,
        ) as client,
    ):
        server_version = client.server_info.version
        if server_version != expected_version:
            failures.append(
                "deployed MCP version mismatch: "
                f"expected {expected_version!r}, got {server_version!r}"
            )

        listed = await client.list_tools()
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

        catalog_result = await client.read_resource("capability://catalog")
        catalog_content = catalog_result.contents[0]
        if not isinstance(catalog_content, TextResourceContents):
            raise RuntimeError("deployed capability catalog is not text")
        catalog_text = catalog_content.text
        catalog = json.loads(catalog_text)
        catalog_digest = (
            "sha256:"
            + hashlib.sha256(
                canonicalize_json(
                    {
                        "catalog_version": catalog["catalog_version"],
                        "capabilities": catalog["capabilities"],
                    }
                )
            ).hexdigest()
        )
        capability_ids = {
            capability["capability_id"] for capability in catalog["capabilities"]
        }
        missing = sorted(required_capabilities - capability_ids)
        if missing:
            failures.append(
                f"deployed catalog is missing required capabilities: {missing!r}"
            )
        policy_profile = catalog["policy_profile"]
        if policy_profile != expected_policy_profile:
            failures.append(
                "deployed capability policy mismatch: "
                f"expected {expected_policy_profile!r}, got {policy_profile!r}"
            )

        discovery_result = await client.call_tool(
            "math.find",
            {"query": query, "limit": 5},
        )
        if discovery_result.is_error:
            failures.append("deployed capability discovery returned an MCP error")
        discovery_content = discovery_result.content[0]
        if not isinstance(discovery_content, TextContent):
            raise RuntimeError("deployed capability discovery is not text")
        discovery_text = discovery_content.text
        if not isinstance(discovery_result.structured_content, dict):
            raise RuntimeError("deployed capability discovery is not structured")
        discovery = discovery_result.structured_content
        discovery_bytes = len(
            json.dumps(discovery, ensure_ascii=False, indent=2).encode("utf-8")
        )
        discovery_model_visible_bytes = len(discovery_text.encode("utf-8"))
        if discovery["response_byte_limit"] != DISCOVERY_RESPONSE_BYTE_LIMIT:
            failures.append("deployed discovery byte limit does not match the contract")
        if discovery_bytes > DISCOVERY_RESPONSE_BYTE_LIMIT:
            failures.append("deployed discovery response exceeds its byte limit")

        report = {
            "url": url,
            "server": {
                "name": client.server_info.name,
                "version": server_version,
            },
            "tool_names": sorted(tool_names),
            "catalog": {
                "catalog_version": catalog["catalog_version"],
                "capabilities": len(capability_ids),
                "policy_profile": policy_profile,
                "catalog_digest": catalog_digest,
                "policy_digest": catalog["policy_digest"],
                "sha256": hashlib.sha256(catalog_text.encode("utf-8")).hexdigest(),
            },
            "discovery": {
                "bytes": discovery_bytes,
                "model_visible_bytes": discovery_model_visible_bytes,
                "matches": [match["capability_id"] for match in discovery["matches"]],
            },
        }
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
        expected_policy_profile=args.expect_policy_profile,
        required_capabilities=set(args.require_capability),
        query=args.query,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except RuntimeError as exc:
        raise SystemExit(f"smoke failed: {exc}") from None
