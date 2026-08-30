from __future__ import annotations

from itertools import combinations
from typing import Literal

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chip_firing._models import LaplacianRequest
from jacobian.math.graphs.chip_firing.operations import laplacian
from jacobian.math.graphs.uniform_subset_intersection._models import (
    UniformSubsetIntersectionRequest,
    UniformSubsetIntersectionResult,
)
from jacobian.math.graphs.uniform_subset_intersection._tools import (
    compute_uniform_subset_intersection_graph,
)
from jacobian.math.graphs.uniform_subset_intersection.operations import (
    construct_uniform_subset_intersection_graph,
)


def _subset_label(subset: tuple[int, ...]) -> str:
    return "{" + ",".join(str(x) for x in subset) + "}"


def _construct(
    request: UniformSubsetIntersectionRequest,
) -> UniformSubsetIntersectionResult:
    return construct_uniform_subset_intersection_graph(
        request.ground_set_size,
        request.subset_cardinality,
        request.threshold,
        request.relation,
    )


def test_kneser_kg42() -> None:
    """KG(4,2): 2-subsets of [4] with intersection < 1."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = _construct(req)
    graph = result.graph
    assert len(graph.vertices) == 6  # C(4,2) = 6
    # KG(4,2) is the Petersen graph minus... actually KG(4,2) has 6 vertices
    # and is 3 disjoint edges (perfect matching)
    assert len(graph.edges) == 3


def test_johnson_eq_threshold() -> None:
    """Johnson-scheme equality: edges when intersection == threshold."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=5,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_EQ_THRESHOLD",
    )
    result = _construct(req)
    graph = result.graph
    assert len(graph.vertices) == 10  # C(5,2) = 10
    # Two 2-subsets of [5] share exactly 1 element when they are not disjoint
    # Pairs sharing exactly 1 element: total - disjoint pairs
    # Disjoint pairs: for each pair {a,b}, {c,d} with no overlap, that's C(5,2)*C(3,2)/2 = 15
    # Total pairs: C(10,2) = 45
    # So 45 - 15 = 30 edges with intersection == 1
    assert len(graph.edges) == 30


def test_empty_k_zero() -> None:
    """k=0 means one empty subset vertex, no edges."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=0,
        threshold=0,
        relation="INTERSECTION_EQ_THRESHOLD",
    )
    result = _construct(req)
    graph = result.graph
    assert graph.vertices == ("{}",)
    assert graph.edges == ()


def test_empty_subset_does_not_materialize_the_ground_axis() -> None:
    result = construct_uniform_subset_intersection_graph(
        (1 << 53) - 1,
        0,
        0,
        "INTERSECTION_EQ_THRESHOLD",
    )
    assert result.graph.vertices == ("{}",)
    assert result.graph.edges == ()


def test_single_subset() -> None:
    """k=n yields one subset, no edges."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=3,
        subset_cardinality=3,
        threshold=0,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = _construct(req)
    graph = result.graph
    assert len(graph.vertices) == 1
    assert len(graph.edges) == 0


def test_vertex_labels_retain_source() -> None:
    """Each vertex label is the canonical subset representation."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = _construct(req)
    expected_labels = {_subset_label(s) for s in combinations(range(4), 2)}
    assert set(result.graph.vertices) == expected_labels


def test_exhaustive_small_comparison() -> None:
    """Compare against independent enumeration for small n,k,t."""
    for n in range(2, 6):
        for k in range(1, n + 1):
            subsets = list(combinations(range(n), k))
            for t in range(k + 1):
                relations: tuple[
                    Literal["INTERSECTION_LT_THRESHOLD", "INTERSECTION_EQ_THRESHOLD"],
                    ...,
                ] = (
                    "INTERSECTION_LT_THRESHOLD",
                    "INTERSECTION_EQ_THRESHOLD",
                )
                for relation in relations:
                    req = UniformSubsetIntersectionRequest(
                        ground_set_size=n,
                        subset_cardinality=k,
                        threshold=t,
                        relation=relation,
                    )
                    result = _construct(req)
                    edges = set()
                    for i, a in enumerate(subsets):
                        for j, b in enumerate(subsets):
                            if i >= j:
                                continue
                            isect = len(set(a) & set(b))
                            if relation == "INTERSECTION_LT_THRESHOLD":
                                adj = isect < t
                            else:
                                adj = isect == t
                            if adj:
                                la, lb = _subset_label(a), _subset_label(b)
                                edges.add((min(la, lb), max(la, lb)))
                    assert set(result.graph.edges) == edges


def test_no_loops_or_duplicates() -> None:
    """No self-loops or duplicate edges."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=5,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = _construct(req)
    for a, b in result.graph.edges:
        assert a != b, "self-loop found"
    edges = result.graph.edges
    assert len(edges) == len(set(edges)), "duplicate edges found"


def test_rejects_k_exceeds_n() -> None:
    request = UniformSubsetIntersectionRequest(
        ground_set_size=3,
        subset_cardinality=5,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    with pytest.raises(OperationDomainValidationError):
        _construct(request)


def test_rejects_threshold_exceeds_k() -> None:
    request = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=2,
        threshold=3,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    with pytest.raises(OperationDomainValidationError):
        _construct(request)


def test_rejects_family_beyond_graph_vertex_bound_before_enumeration() -> None:
    request = UniformSubsetIntersectionRequest(
        ground_set_size=11,
        subset_cardinality=5,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    with pytest.raises(OperationDomainValidationError, match="256-vertex graph bound"):
        _construct(request)


def test_rejects_huge_binomial_family_without_constructing_it() -> None:
    request = UniformSubsetIntersectionRequest(
        ground_set_size=(1 << 53) - 1,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    with pytest.raises(OperationDomainValidationError, match="256-vertex graph bound"):
        _construct(request)


def test_catalog_adapter_uses_the_native_admission() -> None:
    request = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = compute_uniform_subset_intersection_graph(request)
    assert len(result.graph.vertices) == 6


def test_native_api_accepts_domain_arguments() -> None:
    result = construct_uniform_subset_intersection_graph(
        4, 2, 1, "INTERSECTION_LT_THRESHOLD"
    )
    assert len(result.graph.vertices) == 6
    assert len(result.graph.edges) == 3


def test_graph_serializes_unchanged_into_graph_consumer() -> None:
    result = construct_uniform_subset_intersection_graph(
        3, 2, 1, "INTERSECTION_EQ_THRESHOLD"
    )
    consumer_request = LaplacianRequest.model_validate(
        {"graph": result.graph.model_dump(mode="json")}
    )
    consumed = laplacian(consumer_request.graph)
    assert consumed.vertices == result.graph.vertices
    assert consumed.degrees == (2, 2, 2)


def test_result_retains_metadata() -> None:
    """Result retains the source parameters."""
    req = UniformSubsetIntersectionRequest(
        ground_set_size=4,
        subset_cardinality=2,
        threshold=1,
        relation="INTERSECTION_LT_THRESHOLD",
    )
    result = _construct(req)
    assert result.ground_set_size == 4
    assert result.subset_cardinality == 2
    assert result.threshold == 1
    assert result.relation == "INTERSECTION_LT_THRESHOLD"
