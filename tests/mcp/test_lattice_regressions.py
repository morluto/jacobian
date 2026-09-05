"""Lattice corrections retain their exact values across live MCP."""

import asyncio

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.mcp.server import create_server
from mcp import Client


def test_lattice_corrections_through_mcp() -> None:
    cases = (
        ("lattice.dual.compute", [[1, 1], [0, 2]]),
        ("lattice.dual.compute", [[2, 2]]),
        ("lattice.rank_gram.compute", [[1, 0], [0, 1]]),
        ("lattice.rank_gram.compute", [[1, 1]]),
        ("lattice.saturation.compute", [[2, 2]]),
    )

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=True) as client:
            for operation_id, rows in cases:
                lattice = {
                    "ambient_dimension": len(rows[0]),
                    "basis": {"entries": [[str(v) for v in row] for row in rows]},
                }
                payload = (
                    lattice
                    if operation_id == "lattice.saturation.compute"
                    else {"lattice": lattice}
                )
                expected = invoke_operation(operation_id, payload, Catalog.open())
                result = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert result.structured_content["output"] == expected.output

    asyncio.run(scenario())
