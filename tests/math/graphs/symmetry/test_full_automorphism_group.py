"""Full color-preserving graph automorphism group tests."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.symmetry.operations import full_graph_automorphism_group
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def test_cycle_four_has_dihedral_automorphism_group() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3"),
            edges=(("0", "1"), ("0", "3"), ("1", "2"), ("2", "3")),
        )
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 8
    assert result.generated_group_order == 8
    assert len(result.generators) == 2


def test_vertex_colors_restrict_the_full_group() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
        ),
        vertex_colors=("left", "middle", "right"),
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 1
    assert result.generators == ()


def test_dense_graph_edge_scan_work_is_rejected_before_enumeration() -> None:
    vertices = tuple(f"v{i:03}" for i in range(256))
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        ),
        vertex_colors=("free",) * 8 + tuple(f"fixed-{i}" for i in range(248)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        full_graph_automorphism_group(graph)
