"""LP-owned MCP inspection and derived admission recovery (#3191)."""

import asyncio
import json
from itertools import combinations
from typing import Any

import pytest
from mcp.types import ContentBlock, TextContent
from tests.support.rationals import rational_payload as q

from jacobian.catalog.catalog import Catalog
from jacobian.math.optimization._general_models import (
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._tools import TOOLS
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


def _content_text(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text


OPERATION = "optimization.linear.rational_general_optimum.compute"


def _cover_payload(rows: list[list[int]]) -> dict[str, object]:
    return {
        "program": {
            "variables": [
                {"name": f"x{i}", "lower_bound": q(0)} for i in range(len(rows[0]))
            ],
            "objective": {
                "sense": "MINIMIZE",
                "coefficients": [q(1)] * len(rows[0]),
            },
            "constraints": [
                {
                    "label": f"row{i}",
                    "coefficients": [q(value) for value in row],
                    "relation": "GE",
                    "rhs": q(1),
                }
                for i, row in enumerate(rows)
            ],
        }
    }


async def _invoke_lp(payload: dict[str, object], *, direct: bool) -> Any:
    catalog = Catalog(TOOLS)
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    async with Client(server, raise_exceptions=False) as client:
        return await client.call_tool(
            OPERATION if direct else "math.run",
            payload if direct else {"operation_id": OPERATION, "payload": payload},
        )


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
        async with Client(server, raise_exceptions=False) as client:
            inspection = await client.call_tool(
                "math.find", {"operation_id": OPERATION}
            )
            text = json.dumps(inspection.structured_content)
            assert "Normalized limits are 32 columns and 64 rows" in text
            assert "C(n+1,r)" in text and "50000000" in text
            if expect_rejection:
                caught = await client.call_tool(
                    "math.run", {"operation_id": OPERATION, "payload": payload}
                )
            else:
                result = await client.call_tool(
                    "math.run", {"operation_id": OPERATION, "payload": payload}
                )
                assert result.structured_content is not None
                output = result.structured_content["output"]
                parsed = GeneralRationalLinearProgramResult.model_validate_json(
                    json.dumps(output)
                )
                assert parsed.status == "OPTIMAL"
                assert parsed.primal_objective is not None
                assert parsed.primal_objective.as_fraction() == m
                return
        diagnostic = json.loads(
            _content_text(caught.content[0]).removeprefix(
                "Error executing tool math.run: "
            )
        )
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


@pytest.mark.parametrize("direct", [False, True])
def test_lp_entry_points_distinguish_short_certificates_from_work_exhaustion(
    direct: bool,
) -> None:
    supports = [
        (0, 1, 3),
        (0, 1, 6),
        (0, 2, 3),
        (0, 2, 6),
        (1, 2, 4),
        (1, 2, 7),
        (3, 4, 6),
        (3, 5, 6),
        (4, 5, 7),
    ]
    grid_rows = [[int(j in support) for j in range(9)] for support in supports]
    success = asyncio.run(_invoke_lp(_cover_payload(grid_rows), direct=direct))
    assert not success.is_error
    assert success.structured_content is not None
    output = (
        success.structured_content if direct else success.structured_content["output"]
    )
    parsed = GeneralRationalLinearProgramResult.model_validate_json(json.dumps(output))
    assert parsed.status == "OPTIMAL"
    assert parsed.primal_objective is not None
    assert parsed.primal_objective.as_fraction().as_integer_ratio() == (8, 3)

    n, m = 18, 6
    exhaustion_rows = [[(j + 1) ** i for j in range(n)] for i in range(m)]
    payload = {
        "program": {
            "variables": [f"x{i}" for i in range(n)],
            "objective": [q(1)] * n,
            "coefficients": [[q(value) for value in row] for row in exhaustion_rows],
            "rhs": [q(-1)] * m,
        }
    }
    standard_operation = "optimization.linear.rational_optimum.compute"

    async def invoke_exhaustion() -> Any:
        catalog = Catalog(TOOLS)
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=direct_operation_tools(catalog) if direct else (),
        )
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                standard_operation if direct else "math.run",
                payload
                if direct
                else {"operation_id": standard_operation, "payload": payload},
            )

    failed = asyncio.run(invoke_exhaustion())
    assert failed.is_error
    assert failed.structured_content is None
    diagnostic = json.loads(
        _content_text(failed.content[0])[_content_text(failed.content[0]).index("{") :]
    )
    assert diagnostic["code"] == "RESOURCE_EXHAUSTED"
    assert diagnostic["resource"] == "work"
    assert diagnostic["operation_id"] == standard_operation
    assert "fixed scalar-update allowance" in diagnostic["hint"]
