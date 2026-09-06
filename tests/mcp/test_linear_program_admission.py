"""LP-owned MCP inspection and derived admission recovery (#3191)."""

import asyncio
import json
from itertools import combinations

import pytest
from mcp.shared.exceptions import MCPError
from tests.support.rationals import rational_payload as q

from jacobian.catalog.catalog import Catalog
from jacobian.math.optimization._general_models import (
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._tools import TOOLS
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client

OPERATION = "optimization.linear.rational_general_optimum.compute"


@pytest.mark.parametrize(
    ("n", "m", "relation", "code", "expect_rejection"),
    [
        (28, 24, "GE", "normalized_columns", True),
        (2, 64, "EQ", "normalized_rows", True),
        (18, 6, "EQ", "work_bound", False),
        (24, 12, "EQ", "basis_bound", False),
    ],
)
def test_lp_inspection_explains_derived_admission(
    n: int, m: int, relation: str, code: str, expect_rejection: bool
) -> None:
    async def scenario() -> None:
        server = _build_server(state=AppState(operation_catalog=Catalog(TOOLS)))
        variables = [{"name": f"private_x{i}", "lower_bound": q(0)} for i in range(n)]
        if code == "normalized_rows":
            for variable in variables:
                variable["upper_bound"] = q(1)
        rows = [[int(i == j % m) for j in range(n)] for i in range(m)]
        if code == "normalized_columns":
            pairs = list(combinations(range(8), 2))
            triples = [
                t
                for t in combinations(range(8), 3)
                if not any(
                    i // 4 == j // 4 and j - i == 1 for i, j in combinations(t, 2)
                )
            ]
            rows = [[int(i in t and j in t) for i, j in pairs] for t in triples]
        payload = {
            "program": {
                "variables": variables,
                "objective": {"sense": "MINIMIZE", "coefficients": [q(1)] * n},
                "constraints": [
                    {
                        "label": f"private_row{i}",
                        "coefficients": [q(v) for v in row],
                        "relation": relation,
                        "rhs": q(1),
                    }
                    for i, row in enumerate(rows)
                ],
            }
        }
        async with Client(server, raise_exceptions=True) as client:
            inspection = await client.call_tool(
                "math.find", {"request": {"op": "inspect", "operation_id": OPERATION}}
            )
            text = json.dumps(inspection.structured_content)
            assert "Normalized limits are 32 columns and 64 rows" in text
            assert "C(n+1,r)" in text and "50000000" in text
            if expect_rejection:
                with pytest.raises(MCPError) as caught:
                    await client.call_tool(
                        "math.run", {"operation_id": OPERATION, "payload": payload}
                    )
            else:
                result = await client.call_tool(
                    "math.run", {"operation_id": OPERATION, "payload": payload}
                )
                assert result.structured_content is not None
                output = result.structured_content["output"]
                parsed = GeneralRationalLinearProgramResult.model_validate(output)
                assert parsed.status == "OPTIMAL"
                assert parsed.primal_objective is not None
                assert parsed.primal_objective.as_fraction() == m
                return
        diagnostic = caught.value.data
        assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
        assert diagnostic["stage"] == "resource_admission"
        assert diagnostic["errors"][0]["code"] == f"optimization.linear.{code}"
        message = diagnostic["errors"][0]["message"]
        assert "normalized_columns=" in message and "normalized_rows=" in message
        assert "private_" not in message and "input_value" not in message
        assert "correct the fields" not in diagnostic["hint"]
        if code == "normalized_columns":
            assert "normalized_columns=52" in message
        if code == "normalized_rows":
            assert "normalized_rows=66" in message

    asyncio.run(scenario())
