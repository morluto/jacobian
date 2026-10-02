"""Live graph polynomial inspection, resource diagnostics, and correction."""

from __future__ import annotations

import asyncio
import json
from itertools import combinations
from math import comb
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.types import TextContent

from jacobian.math.graphs.polynomials._models import GraphPolynomialResult
from jacobian.mcp.server import create_server
from mcp import Client


def _path(order: int) -> dict[str, Any]:
    return {
        "vertex_count": order,
        "edges": [[i, i + 1] for i in range(order - 1)],
    }


def _edge_boundary() -> dict[str, Any]:
    # K7 with three pendant edges and two isolated vertices: 12 vertices,
    # 24 edges. The bridges give a zero flow polynomial and an x^3 Tutte factor.
    return {
        "vertex_count": 12,
        "edges": [list(edge) for edge in combinations(range(7), 2)]
        + [[0, 7], [0, 8], [0, 9]],
    }


def _coefficients(output: dict[str, Any]) -> dict[tuple[int, ...], int]:
    # Decode the very result delivered through MCP, including retained source.
    result = GraphPolynomialResult.model_validate_json(json.dumps(output))
    return {
        term.exponents: term.coefficient.as_fraction().numerator
        for term in result.polynomial.polynomial.terms
    }


@pytest.mark.parametrize("kind", ("chromatic", "tutte", "flow"))
def test_graph_polynomial_inspection_rejection_and_boundary_recovery(kind: str) -> None:
    async def scenario() -> None:
        operation_id = f"graph.polynomial.{kind}.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            contract = inspected.structured_content["operation"]
            schema = contract["input_schema"]
            graph_schema = schema["properties"]["graph"]
            input_definition = schema["$defs"][
                graph_schema["$ref"].removeprefix("#/$defs/")
            ]
            assert input_definition["title"] == "IndexedSimpleUndirectedGraphInput"
            assert (
                "either endpoint order"
                in input_definition["properties"]["edges"]["description"]
            )
            assert graph_schema["properties"]["vertex_count"]["maximum"] == 12
            assert graph_schema["properties"]["edges"]["maxItems"] == 24
            for description in (contract["description"], graph_schema["description"]):
                assert "12 vertices" in description
                assert "24 edges" in description
                assert "computation" in description
            # These limits belong to this request, not to the reusable graph value.
            canonical = input_definition["properties"]
            assert canonical["vertex_count"]["maximum"] == 1024
            assert canonical["edges"]["maxItems"] == 65_536
            validator = Draft202012Validator(schema)
            validator.validate({"graph": _path(12)})
            validator.validate({"graph": _edge_boundary()})

            beyond_edges = _edge_boundary()
            beyond_edges["edges"].append([7, 8])
            for graph, dimension, limit, count in (
                ({"vertex_count": 13, "edges": []}, "vertex_count", 12, 13),
                (_path(13), "vertex_count", 12, 13),
                (beyond_edges, "edges", 24, 25),
            ):
                assert not validator.is_valid({"graph": graph})
                rejected = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {"graph": graph}},
                )
                assert rejected.is_error
                content = rejected.content[0]
                assert isinstance(content, TextContent)
                diagnostic = json.loads(
                    content.text.removeprefix("Error executing tool math.run: ")
                )
                assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
                assert diagnostic["stage"] == "resource_admission"
                error = diagnostic["errors"][0]
                assert error["location"] == ["graph", dimension]
                assert error["code"] == (
                    "graph.polynomial.vertex_count_limit"
                    if dimension == "vertex_count"
                    else "graph.polynomial.edge_count_limit"
                )
                assert f"at most {limit}" in error["message"]
                assert f"received {count}" in error["message"]
                assert "Submit a graph" in error["message"]
                assert "original graph" in error["message"]

            # Follow the correction with an actual request at the vertex cap.
            # Independently: chi(P12)=x(x-1)^11, T(P12)=x^11, F(P12)=0.
            recovery_graphs: tuple[dict[str, Any], ...] = (
                _path(12),
                {"vertex_count": 12, "edges": []},
                _path(0),
            )
            for recovery_graph in recovery_graphs:
                # Preserve the upstream JSON endpoint normalization together
                # with the owner-specific computation limits and canonical output.
                unoriented = {
                    **recovery_graph,
                    "edges": [list(reversed(edge)) for edge in recovery_graph["edges"]],
                }
                validator.validate({"graph": unoriented})
                accepted = await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {"graph": unoriented}},
                )
                assert not accepted.is_error
                output = accepted.structured_content["output"]
                assert output["graph"] == recovery_graph
                expected: dict[tuple[int, ...], int]
                if recovery_graph["edges"]:
                    expected = (
                        {
                            (degree + 1,): (-1) ** (11 - degree) * comb(11, degree)
                            for degree in range(12)
                        }
                        if kind == "chromatic"
                        else {(11, 0): 1}
                        if kind == "tutte"
                        else {}
                    )
                else:
                    vertex_count = recovery_graph["vertex_count"]
                    assert isinstance(vertex_count, int)
                    expected = {
                        (
                            (vertex_count,)
                            if kind == "chromatic"
                            else (0, 0)
                            if kind == "tutte"
                            else (0,)
                        ): 1
                    }
                assert _coefficients(output) == expected

    asyncio.run(scenario())


@pytest.mark.scale
@pytest.mark.parametrize("kind", ("tutte", "flow"))
def test_graph_polynomial_mcp_24_edge_boundary(kind: str) -> None:
    async def scenario() -> None:
        graph = _edge_boundary()
        async with Client(create_server(), raise_exceptions=False) as client:
            accepted = await client.call_tool(
                "math.run",
                {
                    "operation_id": f"graph.polynomial.{kind}.compute",
                    "payload": {"graph": graph},
                },
            )
            assert not accepted.is_error
            output = accepted.structured_content["output"]
            assert output["graph"] == graph
            terms = _coefficients(output)
            if kind == "flow":
                assert terms == {}  # A nowhere-zero flow cannot cross a bridge.
            else:
                # Defining evaluations independently count spanning forests,
                # all edge subsets, and acyclic orientations of the K7 core.
                assert sum(terms.values()) == 7**5  # Cayley's spanning-tree count.
                assert sum(c * 2 ** (i + j) for (i, j), c in terms.items()) == 2**24
                assert (
                    sum(c * 2**i for (i, j), c in terms.items() if j == 0)
                    == 5040 * 2**3
                )
                assert sum(c for (i, j), c in terms.items() if j == 0) == 720
                assert min(i for i, _ in terms) == 3  # Three bridges.
                assert terms[(9, 0)] == 1

    asyncio.run(scenario())
