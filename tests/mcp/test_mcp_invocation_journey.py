"""Owned MCP smoke journey: live SDK find → run without complete-runtime fixtures.

``create_server()`` serves the immutable inline library. Keep this module
small; do not grow ordinary projection or operation matrices here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jacobian.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


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
            response = json.loads(result.content[0].text)
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

            invalid = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.extended_gcd",
                    "payload": {"left": "84", "unexpected": "30"},
                },
            )
            assert invalid.is_error is True
            assert invalid.structured_content is None
            assert "Extra inputs are not permitted" in invalid.content[0].text

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
            assert matching_contract["operation"]["version"] == "3"
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
            assert "unknown operation" in unknown.content[0].text
            assert len(unknown.content[0].text.encode("utf-8")) < 2_048
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
            excluded = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": ("finite_field.polynomial_map.fibers.compute"),
                    }
                },
            )
            assert excluded.is_error is False
            assert excluded.structured_content == {
                "kind": "error",
                "error": {
                    "code": "UNKNOWN_OPERATION",
                    "stage": "operation_resolution",
                    "message": (
                        "Unknown operation: finite_field.polynomial_map.fibers.compute"
                    ),
                    "hint": (
                        "Call math.find with a mathematical query to search "
                        "installed operations."
                    ),
                },
            }

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
            fibers = fiber_partition(table)
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
