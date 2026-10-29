"""Exact repair values decode structurally and preserve their actual consumers."""

import json
from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.coloring import verify_vertex_coloring
from jacobian.math.graphs.coloring._models import (
    MAX_EDGE_COLORING_EDGES,
    PrecoloringEdgeRepairResult,
)


@pytest.mark.parametrize(
    ("vertices", "edges", "colors", "fixed"),
    [
        (0, (), 3, ()),
        (4, (), 3, ()),
        (2, ((0, 1),), 1, ((0, 0),)),
        (4, tuple(combinations(range(4), 2)), 3, ((0, 0), (1, 0))),
        (4, tuple(combinations(range(4), 2)), 4, ()),
    ],
)
def test_public_repair_result_has_independent_optimum_and_unchanged_coloring_consumer(
    vertices: int,
    edges: tuple[tuple[int, int], ...],
    colors: int,
    fixed: tuple[tuple[int, int], ...],
) -> None:
    payload = {
        "graph": {"vertex_count": vertices, "edges": [list(edge) for edge in edges]},
        "colors": colors,
        "fixed_colors": [list(pair) for pair in fixed],
    }
    reply = invoke_operation(
        "graph.coloring.precoloring_edge_repair.compute", payload, Catalog.open()
    )
    result = PrecoloringEdgeRepairResult.model_validate_json(json.dumps(reply.output))
    assignment = result.coloring.coloring
    repaired = tuple(
        index
        for index, (left, right) in enumerate(edges)
        if assignment[left] == assignment[right]
    )
    optimum = min(
        sum(candidate[left] == candidate[right] for left, right in edges)
        for candidate in product(range(colors), repeat=vertices)
        if all(candidate[vertex] == color for vertex, color in fixed)
    )
    assert result.graph.vertex_count == vertices
    assert result.graph.edges == edges
    assert result.fixed_colors == fixed
    assert result.repaired_edge_indices == repaired
    assert result.repaired_edge_count == optimum == len(repaired)
    # The existing native consumer checks properness of the canonical assignment,
    # not the optimum of an independently supplied repair-result claim.
    assert verify_vertex_coloring(result.coloring) is (optimum == 0)


def test_full_source_edge_axis_remains_admitted_and_shape_bounds_are_visible() -> None:
    edges = tuple(combinations(range(64), 2))
    assert len(edges) == MAX_EDGE_COLORING_EDGES
    output = invoke_operation(
        "graph.coloring.precoloring_edge_repair.compute",
        {
            "graph": {"vertex_count": 64, "edges": [list(edge) for edge in edges]},
            "colors": 1,
        },
        Catalog.open(),
    ).output
    result = PrecoloringEdgeRepairResult.model_validate_json(json.dumps(output))
    # A one-color palette forces every edge to be monochromatic; no search
    # or replay is needed to derive the independent optimum.
    assert result.repaired_edge_indices == tuple(range(MAX_EDGE_COLORING_EDGES))
    assert result.repaired_edge_count == MAX_EDGE_COLORING_EDGES
    assert result.coloring.coloring == (0,) * 64
    schema = PrecoloringEdgeRepairResult.model_json_schema(mode="serialization")
    assert (
        schema["properties"]["repaired_edge_indices"]["maxItems"]
        == MAX_EDGE_COLORING_EDGES
    )
    assert (
        schema["properties"]["repaired_edge_count"]["maximum"]
        == MAX_EDGE_COLORING_EDGES
    )
    assert schema["properties"]["repaired_edge_count"]["minimum"] == 0
    for count, indices, code in (
        (-1, [], "greater_than_equal"),
        (MAX_EDGE_COLORING_EDGES + 1, [], "less_than_equal"),
        (1, [0] * (MAX_EDGE_COLORING_EDGES + 1), "too_long"),
    ):
        with pytest.raises(ValidationError) as error:
            PrecoloringEdgeRepairResult.model_validate_json(
                json.dumps(
                    dict(
                        output, repaired_edge_count=count, repaired_edge_indices=indices
                    )
                )
            )
        assert error.value.errors()[0]["type"] == code
