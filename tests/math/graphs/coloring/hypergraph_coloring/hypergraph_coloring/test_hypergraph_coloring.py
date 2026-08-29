from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.coloring.hypergraph_coloring.operations import (
    decide_hypergraph_coloring,
)


def _hypergraph(vertices, edges):
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
