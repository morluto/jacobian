from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.coloring.hypergraph_coloring._models import (
    HypergraphColoringRequest,
)
from jacobian.math.graphs.coloring.hypergraph_coloring.operations import (
    decide_hypergraph_coloring,
)


def _hypergraph(
    vertices: Sequence[str], edges: Sequence[tuple[str, Sequence[str]]]
) -> FiniteHypergraph:
    return FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((eid, tuple(m)) for eid, m in edges),
    )


def test_empty_hypergraph() -> None:
    hg = _hypergraph(["a"], [])
    result = decide_hypergraph_coloring(hg, 2)
    assert result.colorable


def test_simple_2_colorable() -> None:
    hg = _hypergraph(
        ["a", "b", "c"],
        [("e0", ["a", "b"]), ("e1", ["b", "c"])],
    )
    result = decide_hypergraph_coloring(hg, 2)
    assert result.colorable


def test_3_edge_2_colorable() -> None:
    """3-element edge with 2 colours: still colourable (not monochromatic)."""
    hg = _hypergraph(["a", "b", "c"], [("e0", ["a", "b", "c"])])
    result = decide_hypergraph_coloring(hg, 2)
    assert result.colorable


def test_single_edge_1_color() -> None:
    """With palette size 1, every edge is monochromatic."""
    hg = _hypergraph(["a", "b"], [("e0", ["a", "b"])])
    result = decide_hypergraph_coloring(hg, 1)
    assert not result.colorable


def test_result_preserves_source() -> None:
    hg = _hypergraph(["a", "b"], [("e0", ["a", "b"])])
    result = decide_hypergraph_coloring(hg, 2)
    assert result.hypergraph == hg
    assert result.palette_size == 2


def test_dense_graph_coloring_search_is_rejected_before_backtracking() -> None:
    vertices = [f"v{i:02d}" for i in range(20)]
    edges = [
        (f"e{edge_index:03d}", [left, right])
        for edge_index, (left, right) in enumerate(
            (
                pair
                for i, left in enumerate(vertices)
                for pair in [(left, right) for right in vertices[i + 1 :]]
            )
        )
    ]
    hypergraph = _hypergraph(vertices, edges)

    with pytest.raises(OperationDomainValidationError, match="backtracking-state"):
        decide_hypergraph_coloring(hypergraph, 19)
    with pytest.raises(ValidationError, match="backtracking-state"):
        HypergraphColoringRequest(hypergraph=hypergraph, palette_size=19)


def test_distinct_color_fast_path_handles_large_palette() -> None:
    hypergraph = _hypergraph(["a", "b"], [("e0", ["a", "b"])])
    result = decide_hypergraph_coloring(hypergraph, 2)
    assert result.coloring == (0, 1)


def test_singleton_edge_is_uncolorable_with_any_palette() -> None:
    hypergraph = _hypergraph(["a", "b"], [("e0", ["a"])])
    result = decide_hypergraph_coloring(hypergraph, 2)
    assert not result.colorable
    assert result.coloring is None
