"""Tests for graph morphism operations."""

from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    HomomorphismCheckRequest,
    HomomorphismFindRequest,
    RetractionCheckRequest,
    SimpleGraph,
)
from jacobian.math.graphs.morphisms._operations import (
    compute_core_check,
    compute_homomorphism_check,
    compute_homomorphism_find,
    compute_retraction_check,
)
from jacobian.math.graphs.morphisms._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "graph.core.check",
        "graph.homomorphism.check",
        "graph.homomorphism.find",
        "graph.retraction.check",
    }


def test_homomorphism_check_identity() -> None:
    request = HomomorphismCheckRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        vertex_map=(0, 1),
    )
    result = compute_homomorphism_check(request)
    assert result.is_homomorphism is True


def test_homomorphism_check_non_homomorphism() -> None:
    request = HomomorphismCheckRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=()),
        vertex_map=(0, 0),
    )
    result = compute_homomorphism_check(request)
    assert result.is_homomorphism is False


def test_homomorphism_find_k2_to_k2() -> None:
    request = HomomorphismFindRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
    )
    result = compute_homomorphism_find(request)
    assert result.found is True
    assert len(result.vertex_map) == 2


def test_homomorphism_find_no_homomorphism() -> None:
    request = HomomorphismFindRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=1, edges=()),
    )
    result = compute_homomorphism_find(request)
    assert result.found is False


def test_core_check_k2_is_core() -> None:
    request = CoreCheckRequest(graph=SimpleGraph(vertex_count=2, edges=((0, 1),)))
    result = compute_core_check(request)
    assert result.is_core is True


def test_core_check_independent_set_is_not_core() -> None:
    request = CoreCheckRequest(graph=SimpleGraph(vertex_count=3, edges=()))
    result = compute_core_check(request)
    assert result.is_core is False


def test_retraction_check_k3_to_edge() -> None:
    request = RetractionCheckRequest(
        graph=SimpleGraph(vertex_count=3, edges=((0, 1), (1, 2), (0, 2))),
        subgraph_vertices=(0, 1),
    )
    result = compute_retraction_check(request)
    assert result.is_retraction is False
