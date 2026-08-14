"""Owned MCP smoke journey: live SDK find → run without complete-runtime fixtures.

``create_server()`` serves the immutable inline library. Keep this module
small; do not grow ordinary projection or operation matrices here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server

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
                        "operation_id": "integer.compute.gcd",
                    }
                },
            )
            assert isinstance(described.structured_content, dict)
            contract = described.structured_content
            assert contract["operation"]["operation_id"] == "integer.compute.gcd"
            assert contract["operation"]["provider"] == "built-in"
            assert "provider_runtime" not in contract["operation"]
            assert "output_schema" in contract["operation"]

            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.gcd",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            response = json.loads(result.content[0].text)
            assert response["execution"]["status"] == "COMPLETED"
            assert isinstance(result.structured_content, dict)
            assert "mcp_projection" not in result.structured_content
            assert result.structured_content["output"] == response["output"]
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
            cnf_result = cnf_call.structured_content["output"]["result"]["cnf"]
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
            assert assignment_call.structured_content["output"]["result"] == {
                "satisfies": True,
                "first_unsatisfied_clause": None,
            }

            invalid = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.gcd",
                    "payload": {"left": "84", "unexpected": "30"},
                },
            )
            assert isinstance(invalid.structured_content, dict)
            invalid_result = invalid.structured_content
            assert invalid_result["execution"]["status"] == "ERROR"
            assert (
                invalid_result["output"]["error"]["code"]
                == "INVALID_NUMBER_THEORY_REQUEST"
            )

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
            assert matching_contract["operation"]["examples"][0]["name"] == (
                "triangle_with_tail"
            )

            unknown = await client.call_tool(
                "math.run",
                {
                    "operation_id": "missing.operation",
                    "payload": {},
                },
            )
            unknown_result = json.loads(unknown.content[0].text)
            assert unknown.is_error is False
            assert unknown_result["execution"]["status"] == "ERROR"
            assert unknown_result["output"]["error"]["code"] == "UNKNOWN_OPERATION"
            assert "available_operation_ids" not in unknown_result["output"]
            assert len(unknown.content[0].text.encode("utf-8")) < 2_048
            assert isinstance(unknown.structured_content, dict)
            output = unknown.structured_content["output"]
            assert "available_operation_ids" not in output
            assert len(output["nearby_operation_ids"]) <= 5
            assert output["available_recovery_paths"][-1] == {
                "action": "inspect_catalog",
                "resource_uri": "operation://catalog",
            }

    asyncio.run(scenario())


@pytest.mark.requires_provider("flint")
def test_mcp_composes_finite_field_values_by_inline_typed_payload(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        from jacobian.math.finite_fields import (
            Axis,
            AxisBoundMatrix,
            FiniteDimensionalSubspace,
            element,
            finite_field,
            finite_polynomial,
            finite_polynomial_map,
        )

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
            inspected = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": ("finite_field.polynomial_map.fibers.compute"),
                    }
                },
            )
            assert isinstance(inspected.structured_content, dict)

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
            assert table_call.structured_content["execution"]["status"] == "COMPLETED"
            table_output = table_call.structured_content["output"]
            assert "value_refs" not in table_output
            table_value = table_output["result"]

            fibers_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.polynomial_map.fibers.compute",
                    "payload": {"table": table_value},
                },
            )
            assert isinstance(fibers_call.structured_content, dict)
            fibers_output = fibers_call.structured_content["output"]
            assert fibers_output["result"]["table"] == table_value
            assert sorted(
                len(sources) for _image, sources in fibers_output["result"]["fibers"]
            ) == [1, 3]

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
            assert directions_call.structured_content["execution"]["status"] == (
                "COMPLETED"
            )
            directions_output = directions_call.structured_content["output"]
            assert "value_refs" not in directions_output
            directions_value = directions_output["result"]

            incomplete_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.direction_rank_ledger.compute",
                    "payload": {"directions": directions_value},
                },
            )
            assert isinstance(incomplete_call.structured_content, dict)
            assert incomplete_call.structured_content["execution"]["status"] == "ERROR"

            ledger_call = await client.call_tool(
                "math.run",
                {
                    "operation_id": "finite_field.direction_rank_ledger.compute",
                    "payload": {
                        "subspace": subspace.model_dump(mode="json"),
                        "directions": directions_value,
                    },
                },
            )
            assert isinstance(ledger_call.structured_content, dict)
            ledger_output = ledger_call.structured_content["output"]
            assert ledger_call.structured_content["execution"]["status"] == "COMPLETED"
            assert len(ledger_output["result"]["entries"]) == 5

    asyncio.run(scenario())
