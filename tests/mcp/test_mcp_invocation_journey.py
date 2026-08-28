"""Owned MCP smoke journey: live SDK find → run without complete-runtime fixtures.

``create_server()`` serves the immutable inline library. Keep this module
small; do not grow ordinary projection or operation matrices here.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import ContentBlock, TextContent

from jacobian.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def _text_content(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text


def test_mcp_runs_independent_sync_operations_concurrently() -> None:
    """One slow kernel cannot block an independent MCP request."""

    from jacobian._models import StrictModel
    from jacobian.catalog.builtins import BUILTIN_TOOLS
    from jacobian.catalog.catalog import Catalog
    from jacobian.catalog.models import MathTool
    from jacobian.mcp.runtime import AppState
    from jacobian.mcp.server import _build_server

    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int
        simultaneous_calls: int

    active_calls = 0
    active_lock = threading.Lock()
    second_call_entered = threading.Event()

    def concurrent_kernel(request: Request) -> Result:
        nonlocal active_calls
        with active_lock:
            active_calls += 1
            simultaneous_calls = active_calls
            if active_calls == 2:
                second_call_entered.set()
        try:
            second_call_entered.wait(timeout=0.25)
            return Result(value=request.value, simultaneous_calls=simultaneous_calls)
        finally:
            with active_lock:
                active_calls -= 1

    tool = MathTool(
        operation_id="test.concurrent.kernel",
        title="Concurrent execution sentinel",
        description="Reports simultaneous kernel calls.",
        request_type=Request,
        result_type=Result,
        run=concurrent_kernel,
    )
    server = _build_server(
        state=AppState(operation_catalog=Catalog((*BUILTIN_TOOLS, tool)))
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            results = await asyncio.gather(
                *(
                    client.call_tool(
                        "math.run",
                        {
                            "operation_id": "test.concurrent.kernel",
                            "payload": {"value": value},
                        },
                    )
                    for value in range(2)
                )
            )
        assert any(
            result.structured_content["output"]["simultaneous_calls"] == 2
            for result in results
        )

    asyncio.run(scenario())


def test_mcp_describes_and_invokes_operations(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            described = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "integer.compute.extended_gcd",
                    }
                },
            )
            assert isinstance(described.structured_content, dict)
            contract = described.structured_content
            assert (
                contract["operation"]["operation_id"] == "integer.compute.extended_gcd"
            )
            assert "output_schema" in contract["operation"]

            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.extended_gcd",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            response = json.loads(_text_content(result.content[0]))
            assert response["runtime_ms"] >= 0
            assert isinstance(result.structured_content, dict)
            assert "mcp_projection" not in result.structured_content
            assert result.structured_content["output"] == response["output"]
            assert result.structured_content["output"] == {
                "gcd": "6",
                "left_coefficient": "-1",
                "right_coefficient": "3",
            }
            assert "provider" not in result.structured_content
            assert "provider_digest" not in result.structured_content

            cnf_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "sat.cnf.canonicalize",
                    "payload": {
                        "variable_names": ["b", "a"],
                        "clauses": [[1, -2], [2]],
                    },
                },
            )
            assert isinstance(cnf_call.structured_content, dict)
            cnf_result = cnf_call.structured_content["output"]["cnf"]
            assert cnf_result == {
                "variables": ["a", "b"],
                "clauses": [[-1, 2], [1]],
            }

            assignment_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "sat.assignment.check",
                    "payload": {"cnf": cnf_result, "assignment": [True, True]},
                },
            )
            assert isinstance(assignment_call.structured_content, dict)
            assert assignment_call.structured_content["output"] == {
                "satisfies": True,
                "first_unsatisfied_clause": None,
            }

            with pytest.raises(MCPError) as invalid_error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.extended_gcd",
                        "payload": {
                            "left": "84",
                            "right": "30",
                            "private": "reject-this-private-value",
                        },
                    },
                )
            assert invalid_error.value.code == -32602
            assert invalid_error.value.message == "operation payload failed validation"
            assert invalid_error.value.data == {
                "code": "INVALID_REQUEST",
                "stage": "operation_validation",
                "operation_id": "integer.compute.extended_gcd",
                "errors": [
                    {
                        "location": ["private"],
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    }
                ],
                "hint": (
                    "Inspect the operation with math.find and correct the fields at "
                    "the reported locations before retrying."
                ),
            }
            assert "reject-this-private-value" not in invalid_error.value.message

            with pytest.raises(MCPError) as noncanonical_error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.extended_gcd",
                        "payload": {"left": 1.5, "right": "30"},
                    },
                )
            assert noncanonical_error.value.code == -32602
            assert noncanonical_error.value.data["errors"] == [
                {
                    "location": [],
                    "code": "canonicalization_error",
                    "message": "JSON floating-point numbers are not allowed",
                }
            ]

            with pytest.raises(MCPError) as semantic_error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "universal_algebra.term.evaluate.compute",
                        "payload": {
                            "algebra": {
                                "carrier": ["0", "1"],
                                "operations": [{"operation_id": "and", "arity": 2}],
                                "tables": [[0, 0, 0, 1]],
                            },
                            "term": {
                                "nodes": [
                                    {"kind": "variable", "variable_id": 0},
                                    {"kind": "variable", "variable_id": 1},
                                    {
                                        "kind": "application",
                                        "operation": 0,
                                        "children": [0, 1],
                                    },
                                ],
                                "root": 2,
                            },
                            "assignment": [0],
                        },
                    },
                )
            assert semantic_error.value.code == -32602
            assert semantic_error.value.data["errors"] == [
                {
                    "location": ["assignment"],
                    "code": "universal_algebra.assignment_coverage",
                    "message": "assignment must cover exactly the referenced variables",
                }
            ]

            with pytest.raises(MCPError) as oversized_error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "universal_algebra.term.evaluate.compute",
                        "payload": {
                            "term": {
                                "nodes": [{"kind": "x" * 4_096}],
                                "root": 0,
                            }
                        },
                    },
                )
            assert oversized_error.value.code == -32602
            assert all(
                len(issue["message"]) <= 1_024
                for issue in oversized_error.value.data["errors"]
            )

            oversized_fields = {
                f"{'x' * 4_096}{index}": "y" * 2_000 for index in range(64)
            }
            with pytest.raises(MCPError) as bounded_locations_error:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.extended_gcd",
                        "payload": {
                            "left": "84",
                            "right": "30",
                            **oversized_fields,
                        },
                    },
                )
            bounded_data = bounded_locations_error.value.data
            assert all(
                len(component) <= 128
                for issue in bounded_data["errors"]
                for component in issue["location"]
                if isinstance(component, str)
            )

            with pytest.raises(MCPError) as multiple_errors:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.extended_gcd",
                        "payload": {"left": "01", "right": "not-an-integer"},
                    },
                )
            assert [
                error["location"] for error in multiple_errors.value.data["errors"]
            ] == [["left"], ["right"]]

            matching_description = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": ("graph.invariant.maximum_matching.compute"),
                    }
                },
            )
            assert isinstance(matching_description.structured_content, dict)
            matching_contract = matching_description.structured_content
            assert "version" not in matching_contract["operation"]
            assert matching_contract["operation"]["examples"], (
                "operation must publish at least one invocation example"
            )

            unknown = await client.call_tool(
                "math.run",
                {
                    "operation_id": "missing.operation",
                    "payload": {},
                },
            )
            assert unknown.is_error is True
            unknown_text = _text_content(unknown.content[0])
            assert "unknown operation" in unknown_text
            assert len(unknown_text.encode("utf-8")) < 2_048
            assert unknown.structured_content is None

    asyncio.run(scenario())


@pytest.mark.requires_backend("flint")
def test_mcp_composes_public_finite_field_values_with_native_projections(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from jacobian.math.finite_fields import (
            Axis,
            AxisBoundMatrix,
            FiberPartition,
            FiniteDimensionalSubspace,
            FiniteMapTable,
            ProjectiveLine,
            direction_rank_ledger,
            element,
            fiber_partition,
            finite_field,
            finite_polynomial,
            finite_polynomial_map,
        )
        from mcp import Client

        presentation = finite_field(2, (1, 1, 1))
        zero = element(presentation, (0, 0))
        one = element(presentation, (1, 0))
        polynomial_map = finite_polynomial_map(
            finite_polynomial(presentation, (zero, zero, zero, one))
        )

        async with Client(
            create_server(),
            raise_exceptions=True,
        ) as client:
            table_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.polynomial_map.table.compute",
                    "payload": {
                        "polynomial_map": polynomial_map.model_dump(mode="json")
                    },
                },
            )
            assert isinstance(table_call.structured_content, dict)
            assert table_call.structured_content["runtime_ms"] >= 0
            table_output = table_call.structured_content["output"]
            assert "value_refs" not in table_output
            table_value = table_output

            table = FiniteMapTable.model_validate(table_value)
            fibers_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.polynomial_map.fibers.compute",
                    "payload": {"table": table_value},
                },
            )
            assert isinstance(fibers_call.structured_content, dict)
            fibers = FiberPartition.model_validate(
                fibers_call.structured_content["output"]
            )
            assert fibers == fiber_partition(table)
            assert fibers.table == table
            assert sorted(len(sources) for _image, sources in fibers.fibers) == [1, 3]

            rows = Axis(name="b", labels=("b1", "b2"))
            columns = Axis(name="image", labels=("y1",))
            basis_axis = Axis(name="basis", labels=("B1",))
            subspace = FiniteDimensionalSubspace(
                presentation=presentation,
                basis_axis=basis_axis,
                basis=(
                    AxisBoundMatrix(
                        presentation=presentation,
                        row_axis=rows,
                        column_axis=columns,
                        entries=((one,), (zero,)),
                    ),
                ),
            )
            directions_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.projective_line.enumerate",
                    "payload": {
                        "presentation": presentation.model_dump(mode="json"),
                        "axis": rows.model_dump(mode="json"),
                    },
                },
            )
            assert isinstance(directions_call.structured_content, dict)
            assert directions_call.structured_content["runtime_ms"] >= 0
            directions_output = directions_call.structured_content["output"]
            assert "value_refs" not in directions_output
            directions_value = directions_output

            directions = ProjectiveLine.model_validate(directions_value)
            ledger = direction_rank_ledger(subspace, directions)
            assert len(ledger.entries) == 5

    asyncio.run(scenario())
