"""Admission follows certificate references, pair visits and retained output."""

import json
from collections.abc import Iterator
from itertools import combinations
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.clique_partition._models import (
    MAX_PARTITION_PARTS,
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


@pytest.mark.parametrize(
    "parts",
    ["ab", ("ab",), (("a", 1),), (("a",),), (("a", "a"),), (("a", "z"),)],
)
def test_native_malformed_parts_cannot_produce_unserializable_results(
    parts: object,
) -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    with pytest.raises(OperationDomainValidationError):
        check_edge_clique_partition(graph, cast(tuple[tuple[str, ...], ...], parts))


@pytest.mark.parametrize("wire_json", [False, True])
def test_raw_aggregate_gate_precedes_nested_label_validation(wire_json: bool) -> None:
    payload = {
        "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
        "parts": [[None] * 256] * 256,
    }
    with pytest.raises(ValidationError) as error:
        if wire_json:
            EdgeCliquePartitionRequest.model_validate_json(json.dumps(payload))
        else:
            EdgeCliquePartitionRequest.model_validate(payload)
    assert [entry["type"] for entry in error.value.errors()] == [
        "graph.clique_partition.vertex_reference_bound"
    ]


def test_native_aggregate_gate_precedes_member_validation() -> None:
    graph = _complete(256)
    parts = cast(tuple[tuple[str, ...], ...], ((None,) * 256,) * 256)
    with pytest.raises(OperationResourceAdmissionError):
        check_edge_clique_partition(graph, parts)


def test_native_part_count_rejection_does_not_iterate_the_parts() -> None:
    class UnreadableParts(tuple[tuple[str, ...], ...]):
        def __iter__(self) -> Iterator[tuple[str, ...]]:
            raise AssertionError("over-budget outer tuples must not be traversed")

    graph = _complete(2)
    parts = UnreadableParts(((),) * (MAX_PARTITION_PARTS + 1))
    with pytest.raises(OperationResourceAdmissionError, match="parts"):
        check_edge_clique_partition(graph, parts)
