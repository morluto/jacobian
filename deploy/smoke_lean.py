"""Operation and behavior smoke for a deployed pinned Lean portfolio."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check deployed CORE/MATHLIB verification and accepted/rejected "
            "Lean tactic transitions. This smoke writes disposable artifacts."
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


def _require_verified(result: dict[str, Any], *, environment: str) -> None:
    if result.get("execution", {}).get("status") != "COMPLETED":
        raise RuntimeError(f"{environment} lean.check did not complete")
    if result.get("output", {}).get("conclusion") != "TRUE":
        raise RuntimeError(f"{environment} lean.check did not accept True")
    if not result.get("verification_record_uri"):
        raise RuntimeError(f"{environment} lean.check was not independently verified")


def _require_transition(
    result: dict[str, Any], *, accepted: bool, completed: bool
) -> None:
    if result.get("execution", {}).get("status") != "COMPLETED":
        raise RuntimeError("Lean tactic transition did not complete operationally")
    output = result.get("output", {})
    if (
        output.get("accepted") is not accepted
        or output.get("completed") is not completed
    ):
        raise RuntimeError(
            "Lean tactic transition returned an unexpected candidate verdict"
        )
    successors = output.get("successor_states")
    if accepted != (isinstance(successors, list) and len(successors) == 1):
        raise RuntimeError("Lean tactic transition successor binding is inconsistent")
    if not accepted and not output.get("diagnostics"):
        raise RuntimeError("rejected Lean tactic returned no actionable diagnostics")


def _require_mathlib_declaration(result: dict[str, Any]) -> None:
    if result.get("execution", {}).get("status") != "COMPLETED":
        raise RuntimeError("MATHLIB declaration search did not complete")
    output = result.get("output", {})
    native_result = output.get("result") if isinstance(output, dict) else None
    declarations = (
        native_result.get("declarations")
        if isinstance(native_result, dict)
        else None
    )
    if not isinstance(declarations, list) or not declarations:
        raise RuntimeError("MATHLIB declaration search returned no exact match")
    if declarations[0].get("name") != "irrational_sqrt_two":
        raise RuntimeError("MATHLIB declaration search returned an unexpected match")


async def inspect(*, url: str, timeout_seconds: float) -> dict[str, Any]:
    token = _token()
    headers = {"Authorization": f"Bearer {token}"} if token else None
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
        core = await _run(
            client,
            "lean.check",
            {"statement": "True", "proof": "by trivial", "environment": "CORE"},
        )
        _require_verified(core, environment="CORE")
        mathlib = await _run(
            client,
            "lean.check",
            {
                "statement": "True",
                "proof": "by trivial",
                "environment": "MATHLIB",
            },
        )
        _require_verified(mathlib, environment="MATHLIB")
        declaration = await _run(
            client,
            "lean.declaration.search",
            {
                "environment": "MATHLIB",
                "name_contains": "irrational_sqrt_two",
                "result_limit": 1,
            },
        )
        _require_mathlib_declaration(declaration)

        accepted = await _run(
            client,
            "lean.proof_state.apply_tactic",
            {"statement": "True", "tactic": "trivial", "environment": "CORE"},
        )
        _require_transition(accepted, accepted=True, completed=True)
        rejected = await _run(
            client,
            "lean.proof_state.apply_tactic",
            {"statement": "1 = 2", "tactic": "rfl", "environment": "CORE"},
        )
        _require_transition(rejected, accepted=False, completed=False)

    return {
        "url": url,
        "checks": {
            "core_verification": "VERIFIED",
            "mathlib_verification": "VERIFIED",
            "mathlib_declaration_search": "COMPLETED",
            "accepted_tactic": "COMPLETED",
            "rejected_tactic": "DIAGNOSTIC",
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
    except RuntimeError as exc:
        raise SystemExit(f"Lean smoke failed: {exc}") from None
