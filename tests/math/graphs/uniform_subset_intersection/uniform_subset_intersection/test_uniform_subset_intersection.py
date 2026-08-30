from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.uniform_subset_intersection.operations import (
    construct_uniform_subset_intersection_graph,
)


def test_n3_k2_lt1() -> None:
    """All pairs of 2-subsets of [3] have intersection of size 1 (not < 1)."""
    result = construct_uniform_subset_intersection_graph(
        3, 2, 1, "INTERSECTION_LT_THRESHOLD"
    )
    # No pairs with intersection < 1 (all share exactly 1 element)
    assert len(result.graph.edges) == 0


def test_n5_k2_eq0() -> None:
    """Kneser graph KG(5,2): disjoint pairs of 2-subsets."""
    result = construct_uniform_subset_intersection_graph(
        5, 2, 0, "INTERSECTION_EQ_THRESHOLD"
    )
    # C(5,2)=10 vertices; disjoint pairs = edges of Petersen graph
    assert len(result.graph.vertices) == 10
    assert len(result.graph.edges) == 15  # Petersen graph has 15 edges


def test_n4_k2_gt0() -> None:
    """All pairs sharing at least one element."""
    result = construct_uniform_subset_intersection_graph(
        4, 2, 0, "INTERSECTION_GT_THRESHOLD"
    )
    # C(4,2)=6 vertices; pairs sharing at least one element
    assert len(result.graph.vertices) == 6


def test_n3_k1_lt1() -> None:
    """Singletons of [3] are all pairwise disjoint."""
    result = construct_uniform_subset_intersection_graph(
        3, 1, 1, "INTERSECTION_LT_THRESHOLD"
    )
    # C(3,1)=3 vertices, all pairs have intersection 0 < 1
    assert len(result.graph.edges) == 3


def test_result_preserves_params() -> None:
    result = construct_uniform_subset_intersection_graph(
        4, 2, 1, "INTERSECTION_EQ_THRESHOLD"
    )
    assert result.n == 4
    assert result.k == 2
    assert result.threshold == 1


def test_negative_uniform_cardinality_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="0 <= k <= n"):
        construct_uniform_subset_intersection_graph(
            3, -1, 0, "INTERSECTION_EQ_THRESHOLD"
        )


def test_uniform_family_must_fit_the_graph_carrier() -> None:
    with pytest.raises(OperationDomainValidationError, match="256-vertex"):
        construct_uniform_subset_intersection_graph(
            257, 1, 2, "INTERSECTION_EQ_THRESHOLD"
        )


def test_single_huge_subset_is_rejected_before_materialization() -> None:
    with pytest.raises(OperationDomainValidationError, match="materialization work"):
        construct_uniform_subset_intersection_graph(
            10_000_000,
            10_000_000,
            0,
            "INTERSECTION_EQ_THRESHOLD",
        )


def test_huge_central_binomial_is_rejected_without_materializing_it() -> None:
    with pytest.raises(OperationDomainValidationError, match="256-vertex"):
        construct_uniform_subset_intersection_graph(
            1_000_000_000_000,
            500_000_000_000,
            0,
            "INTERSECTION_EQ_THRESHOLD",
        )


def test_multidigit_labels_have_canonical_edge_endpoints() -> None:
    result = construct_uniform_subset_intersection_graph(
        11, 1, 1, "INTERSECTION_LT_THRESHOLD"
    )
    assert ("L1_10", "L1_2") in result.graph.edges
