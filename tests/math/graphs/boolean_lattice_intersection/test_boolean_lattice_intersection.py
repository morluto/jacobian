from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.boolean_lattice_intersection._models import (
    BooleanLatticeIntersectionRequest,
)
from jacobian.math.graphs.boolean_lattice_intersection.operations import (
    construct_boolean_lattice_intersection_graph,
)

# Lexicographic order of subset labels: {0,1} < {0} < {1} < {}


def test_n0() -> None:
    """Empty ground set has one vertex (empty set), no edges."""
    result = construct_boolean_lattice_intersection_graph(0, 0, "INTERSECTION_EQ")
    assert len(result.graph.vertices) == 1
    assert len(result.graph.edges) == 0


def test_n1() -> None:
    """Ground set of size 1 has 2 vertices: {} and {0}."""
    result = construct_boolean_lattice_intersection_graph(1, 0, "INTERSECTION_EQ")
    assert len(result.graph.vertices) == 2
    assert len(result.graph.edges) == 1


def test_n2_vertex_count() -> None:
    """Ground set of size 2 has 4 vertices."""
    result = construct_boolean_lattice_intersection_graph(2, 0, "INTERSECTION_EQ")
    assert len(result.graph.vertices) == 4


def test_eq_threshold_r0() -> None:
    """EQ threshold 0 on n=2: edges between disjoint subsets (intersection == 0)."""
    result = construct_boolean_lattice_intersection_graph(2, 0, "INTERSECTION_EQ")
    assert len(result.graph.edges) == 4


def test_lt_threshold() -> None:
    """LT threshold: edges when intersection < threshold."""
    result = construct_boolean_lattice_intersection_graph(2, 1, "INTERSECTION_LT")
    # All pairs with intersection < 1, i.e., intersection = 0
    assert len(result.graph.edges) == 4


def test_gt_threshold() -> None:
    """GT threshold: edges when intersection > threshold."""
    result = construct_boolean_lattice_intersection_graph(2, 0, "INTERSECTION_GT")
    # All pairs with intersection > 0: {0}∩{0,1}={0}(1), {1}∩{0,1}={1}(1), {0}∩{1}={} (0)
    assert len(result.graph.edges) == 2


def test_no_loops_or_duplicates() -> None:
    """No self-loops or duplicate edges."""
    result = construct_boolean_lattice_intersection_graph(3, 1, "INTERSECTION_EQ")
    for a, b in result.graph.edges:
        assert a != b
    edges = result.graph.edges
    assert len(edges) == len(set(edges))


def test_vertex_labels_are_subsets() -> None:
    """Vertex labels are canonical subset representations."""
    result = construct_boolean_lattice_intersection_graph(2, 0, "INTERSECTION_EQ")
    assert set(result.graph.vertices) == {"{}", "{0}", "{1}", "{0,1}"}


def test_threshold_above_ground_set_is_defined() -> None:
    request = BooleanLatticeIntersectionRequest(
        ground_set_size=2, threshold=3, relation="INTERSECTION_LT"
    )
    result = construct_boolean_lattice_intersection_graph(
        request.ground_set_size, request.threshold, request.relation
    )
    assert len(result.graph.edges) == 6


def test_result_preserves_metadata() -> None:
    """Result retains the source parameters."""
    result = construct_boolean_lattice_intersection_graph(2, 1, "INTERSECTION_EQ")
    assert result.ground_set_size == 2
    assert result.threshold == 1
    assert result.relation == "INTERSECTION_EQ"


def test_n8_is_the_carrier_boundary() -> None:
    result = construct_boolean_lattice_intersection_graph(8, 0, "INTERSECTION_EQ")
    assert len(result.graph.vertices) == 256
    assert len(result.graph.edges) == 3_280


def test_native_rejects_negative_ground_set_size() -> None:
    with pytest.raises(OperationDomainValidationError, match="between 0"):
        construct_boolean_lattice_intersection_graph(-1, 0, "INTERSECTION_EQ")


def test_native_rejects_invalid_relation_before_enumeration() -> None:
    with pytest.raises(OperationDomainValidationError, match="relation"):
        construct_boolean_lattice_intersection_graph(8, 0, "INVALID")  # type: ignore[arg-type]
