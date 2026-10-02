"""Grouped square-free output closure and resource refusal through live MCP."""

import asyncio
import json
from itertools import product
from math import prod
from typing import Any

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.math.polynomials._tools import TOOLS
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client

OPERATION = "polynomial.compute.square_free_decomposition"


def _source(n: int) -> dict[str, Any]:
    local = {0: 1, 1: -1, n: -1, n + 1: 1}
    terms = {
        tuple(e for e, c in row): prod(c for e, c in row)
        for row in product(local.items(), repeat=3)
    }
    return {
        "domain": "QQ",
        "variables": ["x", "y", "z"],
        "polynomial": {
            "terms": [
                {"exponents": list(e), "coefficient": {"num": str(c), "den": "1"}}
                for e, c in sorted(terms.items(), reverse=True)
            ]
        },
    }


def test_grouped_factor_handoff_refusal_and_recovery() -> None:
    async def scenario() -> None:
        server = _build_server(state=AppState(operation_catalog=Catalog(TOOLS)))
        async with Client(server, raise_exceptions=False) as client:

            async def run(source: dict[str, Any]) -> Any:
                return await client.call_tool(
                    "math.run",
                    {"operation_id": OPERATION, "payload": {"polynomial": source}},
                )

            found = await client.call_tool("math.find", {"operation_id": OPERATION})
            assert "4096" in json.dumps(found.structured_content)
            accepted = await run(_source(11))
            assert not accepted.is_error
            output = accepted.structured_content["output"]
            assert [row["multiplicity"] for row in output["factors"]] == [1, 2]
            factor = output["factors"][0]["factor"]
            assert len(factor["polynomial"]["terms"]) == 1331
            assert output["reconstructed"] == _source(11)
            consumed = await run(factor)
            assert not consumed.is_error
            assert (
                consumed.structured_content["output"]["factors"][0]["factor"] == factor
            )
            refused = await run(_source(17))
            assert refused.is_error
            block = refused.content[0]
            assert isinstance(block, TextContent)
            diagnostic = json.loads(
                block.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            assert (
                diagnostic["errors"][0]["code"]
                == "polynomial.square_free_grouped_support"
            )
            recovered = await run(_source(10))
            assert not recovered.is_error
            assert (
                len(
                    recovered.structured_content["output"]["factors"][0]["factor"][
                        "polynomial"
                    ]["terms"]
                )
                == 1000
            )

    asyncio.run(scenario())
