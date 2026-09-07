"""Admission follows certificate references, pair visits and retained output."""

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.clique_partition._models import (
    EdgeCliquePartitionRequest,
    EdgeCliquePartitionResult,
)
from jacobian.math.graphs.clique_partition.operations import check_edge_clique_partition
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _complete(order: int) -> SimpleUndirectedGraph:
    vertices = tuple(sorted(map(str, range(order))))
    return SimpleUndirectedGraph(
        vertices=vertices, edges=tuple(combinations(vertices, 2))
    )


@pytest.mark.parametrize("order", [92, 134, 256])
def test_all_edge_certificates_fit_the_graph_carrier(order: int) -> None:
    graph = _complete(order)
    request = EdgeCliquePartitionRequest(graph=graph, parts=graph.edges)
    result = check_edge_clique_partition(request.graph, request.parts)
    assert result.is_partition
    assert len(result.parts) == order * (order - 1) // 2
    assert (
        EdgeCliquePartitionResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_mixed_certificate_above_old_cap_preserves_invalid_diagnostics() -> None:
    graph = _complete(100)
    large_parts = (graph.vertices[:3], graph.vertices[3:7])
    covered = {edge for part in large_parts for edge in combinations(part, 2)}
    parts = large_parts + tuple(edge for edge in graph.edges if edge not in covered)
    assert len(parts) > 4096
    assert check_edge_clique_partition(graph, parts).is_partition
    assert check_edge_clique_partition(graph, parts[:-1]).uncovered_edge == parts[-1]
    assert (
        check_edge_clique_partition(graph, (*parts, parts[-1])).overcovered_edge
        == parts[-1]
    )


@pytest.mark.parametrize("parts", [33, 256])
def test_native_admission_rejects_excess_pair_or_reference_work(parts: int) -> None:
    graph = _complete(256)
    with pytest.raises(OperationResourceAdmissionError):
        check_edge_clique_partition(graph, (graph.vertices,) * parts)
