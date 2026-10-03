"""Actual MCP rank/nullspace errors expose caller paths and truthful counts."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import parse_operation_input
from jacobian.math.matrices._operation_models import MatrixRankRequest
from jacobian.math.matrices._tools import TOOLS
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client

_OPERATIONS = ("matrix.rank.compute", "matrix.nullspace.compute")


def _invalid_matrices() -> list[tuple[dict[str, Any], int]]:
    scalar = {"num": "1", "den": "1"}
    return [
        ({"row_count": 2, "column_count": 2, "entries": [[scalar]]}, 1),
        ({"entries": [[{"num": "1", "den": "bad"}]]}, 1),
        (
            {
                "row_count": 1,
                "column_count": 1,
                "entries": [{"row": -1, "value": scalar}],
            },
            2,
        ),
        (
            {
                "row_count": 1,
                "column_count": 1,
                "entries": [{"row": 1, "column": 0, "value": scalar}],
            },
            1,
        ),
        (
            {
                "row_count": 1,
                "column_count": 1,
                "entries": [[scalar], {"row": 0, "column": 0, "value": scalar}],
            },
            2,
        ),
        ({"row_count": -1, "entries": []}, 3),
        ({"entries": [[{"num": "1", "den": "bad"}] * 9 for _ in range(9)]}, 81),
    ]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("operation_id", _OPERATIONS)
def test_rank_nullspace_mcp_paths_counts_and_recovery(
    direct: bool, operation_id: str
) -> None:
    async def scenario() -> None:
        catalog = Catalog(
            tuple(tool for tool in TOOLS if tool.operation_id in _OPERATIONS)
        )
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=direct_operation_tools(catalog) if direct else (),
        )
        tool_name = operation_id if direct else "math.run"
        async with Client(server, raise_exceptions=False) as client:
            for matrix, count in _invalid_matrices():
                payload = {"matrix": matrix}
                with pytest.raises(ValidationError) as native:
                    parse_operation_input(MatrixRankRequest, payload)
                assert native.value.error_count() == count
                records = native.value.errors(include_url=False)
                response = await client.call_tool(
                    tool_name,
                    payload
                    if direct
                    else {"operation_id": operation_id, "payload": payload},
                )
                assert response.is_error
                assert response.structured_content is None
                assert len(response.content) == 1
                content = response.content[0]
                assert isinstance(content, TextContent)
                diagnostic = json.loads(
                    content.text.removeprefix(f"Error executing tool {tool_name}: ")
                )
                assert diagnostic["code"] == "INVALID_REQUEST"
                assert diagnostic["stage"] == "operation_validation"
                assert diagnostic["operation_id"] == operation_id
                assert diagnostic["errors"] == [
                    {
                        "location": list(error["loc"]),
                        "code": error["type"],
                        "message": error["msg"],
                    }
                    for error in records[:64]
                ]
                assert all(
                    not any(
                        isinstance(part, str)
                        and ("function-after[" in part or "json-or-python[" in part)
                        for part in error["location"]
                    )
                    for error in diagnostic["errors"]
                )
                assert len(diagnostic["errors"]) == min(count, 64)
                assert diagnostic["omitted_error_count"] == max(0, count - 64)
                assert (
                    len(diagnostic["errors"]) + diagnostic["omitted_error_count"]
                    == count
                )
            payload = {"matrix": {"entries": [[{"num": "2", "den": "-4"}]]}}
            recovered = await client.call_tool(
                tool_name,
                payload
                if direct
                else {"operation_id": operation_id, "payload": payload},
            )
            assert not recovered.is_error
            assert recovered.structured_content is not None
            output = (
                recovered.structured_content
                if direct
                else recovered.structured_content["output"]
            )
            assert output["rank"] == 1
            assert output["matrix"]["entries"] == [[{"num": "-1", "den": "2"}]]
            operation = catalog.operation(operation_id)
            assert operation is not None
            operation.result_type.model_validate_json(json.dumps(output), strict=True)
            output["matrix"]["entries"] = [[{"num": "2", "den": "-4"}]]
            with pytest.raises(ValidationError):
                operation.result_type.model_validate_json(
                    json.dumps(output), strict=True
                )

    asyncio.run(scenario())
