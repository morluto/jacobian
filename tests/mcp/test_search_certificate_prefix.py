"""Search prefixes may establish witnesses; exhaustion establishes no decision."""

from __future__ import annotations

import asyncio
import importlib
import json
from itertools import combinations
from typing import Any

import jsonschema
import pytest
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


def _complete_graph(prefix: str, order: int) -> dict[str, object]:
    vertices = [f"{prefix}{index:02d}" for index in range(order)]
    return {
        "vertices": vertices,
        "edges": [list(edge) for edge in combinations(vertices, 2)],
    }


def _positive_cases() -> list[tuple[str, dict[str, object], str, object]]:
    regular_vertices = [f"v{index:02d}" for index in range(18)]
    regular_edges = [["v00", "v01"], ["v00", "v02"], ["v01", "v02"]]
    regular_edges.extend(
        [regular_vertices[index], regular_vertices[index + 1]] for index in range(3, 17)
    )
    pattern = _complete_graph("p", 6)
    pattern_edges = pattern["edges"]
    assert isinstance(pattern_edges, list)
    host = _complete_graph("h", 64)
    host_edges = host["edges"]
    assert isinstance(host_edges, list)
    colored_pattern = {
        "graph": pattern,
        "edge_colors": ["red"] * len(pattern_edges),
    }
    colored_host = {
        "graph": host,
        "edge_colors": ["red"] * len(host_edges),
    }
    hypergraph_vertices = [f"v{index:02d}" for index in range(22)]
    return [
        (
            "graph.k_regular_subgraph.find",
            {
                "graph": {"vertices": regular_vertices, "edges": regular_edges},
                "k": 2,
            },
            "found",
            True,
        ),
        (
            "graph.cycle.fixed_length.decide",
            {"graph": host, "length": 4},
            "decision",
            "EXISTS",
        ),
        (
            "graph.subgraph_pattern.find",
            {"pattern": pattern, "host": host},
            "decision",
            "EXISTS",
        ),
        (
            "graph.edge_colored_subgraph_pattern.find",
            {"pattern": colored_pattern, "host": colored_host},
            "decision",
            "EXISTS",
        ),
        (
            "hypergraph.nonmonochromatic_vertex_coloring.q_decide",
            {
                "hypergraph": {
                    "vertices": hypergraph_vertices,
                    "edges": [["edge", hypergraph_vertices]],
                },
                "palette_size": 2,
            },
            "outcome",
            "COLORABLE",
        ),
    ]


async def _invoke(
    operation_id: str, payload: dict[str, object], *, direct: bool
) -> Any:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    async with Client(server, raise_exceptions=False) as client:
        return await client.call_tool(
            operation_id if direct else "math.run",
            payload if direct else {"operation_id": operation_id, "payload": payload},
        )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("operation_id", "payload", "field", "expected"), _positive_cases()
)
def test_early_witnesses_reach_both_mcp_entry_points(
    direct: bool,
    operation_id: str,
    payload: dict[str, object],
    field: str,
    expected: object,
) -> None:
    result = asyncio.run(_invoke(operation_id, payload, direct=direct))
    assert not result.is_error
    assert result.structured_content is not None
    output = (
        result.structured_content if direct else result.structured_content["output"]
    )
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    jsonschema.validate(
        output, operation.result_type.model_json_schema(mode="serialization")
    )
    assert output[field] == expected


_EXHAUSTION_CASES = (
    (
        "graph.k_regular_subgraph.find",
        "jacobian.math.graphs.regular_subgraph.operations",
        "MAX_REGULAR_SUBGRAPH_WORK",
        3,
        {
            "graph": {
                "vertices": ["a", "b", "c", "d", "e"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]],
            },
            "k": 2,
        },
    ),
    (
        "graph.cycle.fixed_length.decide",
        "jacobian.math.graphs.morphisms.operations",
        "MAX_CYCLE_SEARCH_PATHS",
        1,
        {
            "graph": {
                "vertices": ["a", "b", "c"],
                "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
            },
            "length": 3,
        },
    ),
    (
        "graph.subgraph_pattern.find",
        "jacobian.math.graphs.morphisms.operations",
        "MAX_SUBGRAPH_CANDIDATE_CHECKS",
        1,
        {
            "pattern": {"vertices": ["p", "q"], "edges": [["p", "q"]]},
            "host": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        },
    ),
    (
        "graph.edge_colored_subgraph_pattern.find",
        "jacobian.math.graphs.morphisms.edge_colored_pattern.operations",
        "MAX_ASSIGNMENTS",
        1,
        {
            "pattern": {
                "graph": {"vertices": ["p", "q"], "edges": [["p", "q"]]},
                "edge_colors": ["red"],
            },
            "host": {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["b", "c"]],
                },
                "edge_colors": ["red"],
            },
        },
    ),
    (
        "hypergraph.nonmonochromatic_vertex_coloring.q_decide",
        "jacobian.math.combinatorics.finite_structures.hypergraph_coloring._models",
        "MAX_COLORING_WORK",
        1,
        {
            "hypergraph": {
                "vertices": ["a", "b", "c"],
                "edges": [
                    ["ab", ["a", "b"]],
                    ["ac", ["a", "c"]],
                    ["bc", ["b", "c"]],
                ],
            },
            "palette_size": 2,
        },
    ),
)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("operation_id", "module", "limit", "limit_value", "payload"),
    _EXHAUSTION_CASES,
)
def test_exhausted_prefixes_are_mcp_errors_without_mathematical_output(
    monkeypatch: pytest.MonkeyPatch,
    direct: bool,
    operation_id: str,
    module: str,
    limit: str,
    limit_value: int,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(importlib.import_module(module), limit, limit_value)
    result = asyncio.run(_invoke(operation_id, payload, direct=direct))
    assert result.is_error
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    diagnostic = json.loads(result.content[0].text[result.content[0].text.index("{") :])
    assert diagnostic["code"] == "RESOURCE_EXHAUSTED"
    assert diagnostic["operation_id"] == operation_id
    assert diagnostic["resource"] == "work"
    assert "does not establish a negative result" in diagnostic["hint"]
