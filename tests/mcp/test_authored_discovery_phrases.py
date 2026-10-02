"""Ordinary MCP discovery retains complete authored phrases and usable results."""

import asyncio
from fractions import Fraction

from jacobian.mcp.server import create_server
from mcp import Client


def test_live_authored_phrase_discovery_and_selected_example_execution() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            queries = (
                ("pp-defined relation", "pp_formula.evaluate_relation.compute"),
                ("H to V polyhedron conversion", "polytope.rational.h_to_v.compute"),
                (
                    "characteristic p syzygies",
                    "finite_field.jacobian_syzygy.generators.compute",
                ),
                (
                    "characteristic p differentiation",
                    "finite_field.polynomial.jacobian.compute",
                ),
                (
                    "characteristic p Jacobian syzygy verification",
                    "finite_field.jacobian_syzygy.check",
                ),
                ("GL n p", "finite_matrix_group.general_linear.construct"),
                (
                    "GL n q extension field",
                    "finite_matrix_group.extension.general_linear.construct",
                ),
                ("solve Ax=b", "linear.rational_solution.compute"),
            )
            for query, owner in queries:
                # Omit limit and search_mode: the ordinary ten-result precise query page.
                found = await client.call_tool("math.find", {"query": query})
                assert not found.is_error
                assert found.structured_content is not None
                assert found.structured_content["matches"][0]["operation_id"] == owner
                inspected = await client.call_tool("math.find", {"operation_id": owner})
                assert not inspected.is_error
                assert inspected.structured_content is not None
                declaration = inspected.structured_content["operation"]
                assert query in declaration["discovery_terms"]
            assert owner == "linear.rational_solution.compute"
            result = await client.call_tool(
                "math.run",
                {"operation_id": owner, "payload": declaration["examples"][0]["input"]},
            )
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            assert output["status"] == "SOLUTION"
            solution = [
                Fraction(int(value["num"]), int(value["den"]))
                for value in output["values"]
            ]
            system = output["system"]
            for row, rhs in enumerate(system["rhs"]):
                actual = sum(
                    Fraction(int(entry["value"]["num"]), int(entry["value"]["den"]))
                    * solution[entry["column"]]
                    for entry in system["coefficients"]["entries"]
                    if entry["row"] == row
                )
                assert actual == Fraction(int(rhs["num"]), int(rhs["den"]))
            unsupported = await client.call_tool(
                "math.find", {"query": "check polynomial identity over integers"}
            )
            assert not unsupported.is_error
            assert unsupported.structured_content is not None
            assert unsupported.structured_content["total_matches"] == 0

    asyncio.run(scenario())
