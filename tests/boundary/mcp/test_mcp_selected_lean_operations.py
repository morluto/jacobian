from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server
from jacobian.registry import CheckerRegistry


def test_lean_operations_do_not_assemble_the_portfolio(
    mcp_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_installation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selected Lean operations must not install the portfolio")

    monkeypatch.setattr(CheckerRegistry, "authorize", reject_installation)

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(mcp_state), raise_exceptions=True) as client:
            for operation_id in (
                "lean.check",
                "lean.declaration.dependencies",
                "lean.declaration.inspect",
                "lean.declaration.search",
                "lean.proof.axioms.inspect",
                "lean.proof_edit.validate",
                "lean.proof_state.apply_tactic",
                "lean.proof_state.inspect",
                "lean.proof_state.metavariable_fields",
                "lean.retrieve.premises",
                "lean.statement.compare",
                "lean.statement.propose",
                "lean.term.apply",
            ):
                result = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {}},
                )
                assert isinstance(result.structured_content, dict), operation_id
                assert result.structured_content["operation_id"] == operation_id

    asyncio.run(scenario())
