"""Larger LP certificates compose through discovery and live MCP checking."""

import asyncio
import json
from itertools import combinations
from typing import Any

import pytest
from mcp.types import TextContent
from tests.support.rationals import rational_payload as q

from jacobian.catalog.catalog import Catalog
from jacobian.math.optimization._general_models import GeneralFormRationalLinearProgram
from jacobian.math.optimization._optimality import RationalLinearOptimalityResult
from jacobian.math.optimization._tools import TOOLS
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client

CHECK = "optimization.linear.rational_optimality.check"
SOLVE = "optimization.linear.rational_general_optimum.compute"


def _cover(m: int = 246) -> dict[str, Any]:
    n = 15
    cyclic = [tuple(sorted((i + j) % n for j in range(4))) for i in range(n)]
    supports = (
        cyclic + [s for s in combinations(range(n), 4) if s not in cyclic][: m - n]
    )
    return {
        "program": {
            "variables": [{"name": f"x{i}", "lower_bound": q(0)} for i in range(n)],
            "objective": {"sense": "MINIMIZE", "coefficients": [q(1)] * n},
            "constraints": [
                {
                    "label": f"row{i}",
                    "coefficients": [q(int(j in s)) for j in range(n)],
                    "relation": "GE",
                    "rhs": q(1),
                }
                for i, s in enumerate(supports)
            ],
        },
        "primal_candidate": [q(1, 4)] * n,
        "constraint_dual": [q(1, 4)] * n + [q(0)] * (m - n),
        "lower_bound_dual": [q(0)] * n,
        "upper_bound_dual": [q(0)] * n,
    }


@pytest.mark.parametrize("direct", [False, True])
def test_large_certificate_resource_classification_and_recovery(direct: bool) -> None:
    async def scenario() -> None:
        catalog = Catalog(TOOLS)
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=direct_operation_tools(catalog) if direct else (),
        )
        async with Client(server, raise_exceptions=False) as client:

            async def invoke(operation: str, payload: dict[str, Any]) -> Any:
                return await client.call_tool(
                    operation if direct else "math.run",
                    payload
                    if direct
                    else {"operation_id": operation, "payload": payload},
                )

            found = await client.call_tool("math.find", {"operation_id": CHECK})
            discovery = json.dumps(found.structured_content)
            assert "1,024 rows" in discovery
            assert "100,000 scalar updates" in discovery
            accepted = await invoke(CHECK, _cover())
            assert not accepted.is_error
            output = (
                accepted.structured_content
                if direct
                else accepted.structured_content["output"]
            )
            result = RationalLinearOptimalityResult.model_validate_json(
                json.dumps(output)
            )
            assert result.is_optimal
            assert (
                result.primal_objective.num == 15 and result.primal_objective.den == 4
            )
            assert isinstance(
                result.candidate.program, GeneralFormRationalLinearProgram
            )
            assert len(result.candidate.program.constraints) == 246
            forged = _cover()
            forged["program"]["constraints"][-1]["rhs"] = q(2)
            rejected_claim = await invoke(CHECK, forged)
            assert not rejected_claim.is_error
            claim_output = (
                rejected_claim.structured_content
                if direct
                else rejected_claim.structured_content["output"]
            )
            assert claim_output["is_optimal"] is False
            assert claim_output["objectives_equal"] is True
            refused = await invoke(SOLVE, {"program": _cover()["program"]})
            assert refused.is_error
            block = refused.content[0]
            assert isinstance(block, TextContent)
            diagnostic = json.loads(
                block.text.removeprefix(
                    f"Error executing tool {SOLVE if direct else 'math.run'}: "
                )
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                diagnostic["errors"][0]["code"]
                == "optimization.linear.solver_shape_bound"
            )
            overwork = await invoke(CHECK, _cover(1024))
            assert overwork.is_error
            block = overwork.content[0]
            assert isinstance(block, TextContent)
            diagnostic = json.loads(
                block.text.removeprefix(
                    f"Error executing tool {CHECK if direct else 'math.run'}: "
                )
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                diagnostic["errors"][0]["code"]
                == "optimization.linear.optimality_check_bound"
            )
            recovered = await invoke(CHECK, _cover())
            assert not recovered.is_error

    asyncio.run(scenario())
