"""Tests for edge-deletion diameter profile."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_diameter_profile import (
    EdgeDeletionDiameterProfileResult,
    edge_deletion_diameter_profile,
)
from jacobian.math.graphs.edge_deletion_diameter_profile.operations import (
    MAX_EDGE_DELETION_DIAMETER_WORK,
    MAX_RETAINED_LABEL_CHARACTERS,
    _diameter_profile_work,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def _independent_diameter(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> int | None:
    """Return eccentricity-max diameter, or None when the graph is disconnected."""

    adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertices}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    eccentricities: list[int] = []
    for start in vertices:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)
        if len(distances) != len(vertices):
            return None
        eccentricities.append(max(distances.values()))
    return max(eccentricities)


def _assert_defining_invariant(graph: SimpleUndirectedGraph) -> None:
    result = edge_deletion_diameter_profile(graph)
    assert result.source_diameter == _independent_diameter(graph.vertices, graph.edges)
    assert len(result.entries) == len(graph.edges)
    for index, entry in enumerate(result.entries):
        remaining = tuple(
            edge for edge_index, edge in enumerate(graph.edges) if edge_index != index
        )
        independent = _independent_diameter(graph.vertices, remaining)
        if independent is None:
            assert entry.result == "DISCONNECTED"
            assert entry.diameter is None
        else:
            assert entry.result == "DIAMETER"
            assert entry.diameter == independent
            assert independent >= result.source_diameter


def test_path_three_vertices_disconnected() -> None:
    g = _graph(["0", "1", "2"], [["0", "1"], ["1", "2"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 2
    assert len(result.entries) == 2
    for entry in result.entries:
        assert entry.result == "DISCONNECTED"
        assert entry.diameter is None
    _assert_defining_invariant(g)


def test_triangle_becomes_path() -> None:
    g = _graph(["0", "1", "2"], [["0", "1"], ["0", "2"], ["1", "2"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 1
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        assert entry.diameter == 2
    _assert_defining_invariant(g)


def test_square_cycle() -> None:
    g = _graph(["0", "1", "2", "3"], [["0", "1"], ["0", "3"], ["1", "2"], ["2", "3"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 2
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        assert entry.diameter == 3
    _assert_defining_invariant(g)


def test_disconnected_rejected() -> None:
    g = _graph(["0", "1", "2"], [["0", "1"]])
    with pytest.raises(OperationDomainValidationError, match="connected"):
        edge_deletion_diameter_profile(g)


def test_empty_graph_rejected() -> None:
    g = SimpleUndirectedGraph(vertices=(), edges=())
    with pytest.raises(OperationDomainValidationError, match=r"nonempty|connected"):
        edge_deletion_diameter_profile(g)


def test_json_round_trip() -> None:
    g = _graph(["0", "1", "2"], [["0", "1"], ["1", "2"], ["0", "2"]])
    result = edge_deletion_diameter_profile(g)
    json_val = result.model_dump_json()
    replay = type(result).model_validate_json(json_val, strict=True)
    assert replay == result


def test_complete_graph_remains_connected() -> None:
    g = _graph(
        ["0", "1", "2", "3"],
        [["0", "1"], ["0", "2"], ["0", "3"], ["1", "2"], ["1", "3"], ["2", "3"]],
    )
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 1
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        assert entry.diameter == 2
    _assert_defining_invariant(g)


def test_singleton_has_zero_diameter_and_round_trips() -> None:
    """A connected singleton has eccentricity 0; the public schema must retain it."""

    result = edge_deletion_diameter_profile(_graph(["v"], []))
    assert isinstance(result, EdgeDeletionDiameterProfileResult)
    assert result.source_diameter == 0
    assert result.entries == ()
    replay = type(result).model_validate_json(result.model_dump_json(), strict=True)
    assert replay == result
    assert replay.source_diameter == 0


def test_native_owner_package_exports_the_operation() -> None:
    from jacobian.math.graphs.edge_deletion_diameter_profile import (
        edge_deletion_diameter_profile as exported,
    )
    from jacobian.math.graphs.edge_deletion_diameter_profile.operations import (
        edge_deletion_diameter_profile as kernel,
    )

    assert exported is kernel
    result = exported(_graph(["0", "1"], [["0", "1"]]))
    assert result.source_diameter == 1
    assert result.entries[0].result == "DISCONNECTED"


def test_sparse_path_beyond_the_old_vertex_ceiling_is_admitted() -> None:
    """A 65-vertex path is cheaper than the former 64-vertex dense envelope."""

    vertex_count = 65
    vertices = [str(index) for index in range(vertex_count)]
    edges = [sorted((str(index), str(index + 1))) for index in range(vertex_count - 1)]
    assert (
        _diameter_profile_work(vertex_count, vertex_count - 1)
        <= MAX_EDGE_DELETION_DIAMETER_WORK
    )
    result = edge_deletion_diameter_profile(_graph(vertices, edges))
    assert result.source_diameter == vertex_count - 1
    assert len(result.entries) == vertex_count - 1
    assert all(entry.result == "DISCONNECTED" for entry in result.entries)


def test_work_bound_charges_per_vertex_traversals() -> None:
    """Forgetting the n eccentricity factor would admit this complete graph."""

    vertex_count = 50
    edge_count = vertex_count * (vertex_count - 1) // 2
    work = _diameter_profile_work(vertex_count, edge_count)
    assert work > MAX_EDGE_DELETION_DIAMETER_WORK
    assert edge_count * (vertex_count + edge_count) <= MAX_EDGE_DELETION_DIAMETER_WORK
    vertices = [f"{index:02d}" for index in range(vertex_count)]
    edges = [
        [vertices[left], vertices[right]]
        for left in range(vertex_count)
        for right in range(left + 1, vertex_count)
    ]
    with pytest.raises(OperationDomainValidationError, match="work bound"):
        edge_deletion_diameter_profile(_graph(vertices, edges))


def test_native_result_accepts_retained_labels_at_character_bound() -> None:
    graph = _graph(["x" * MAX_RETAINED_LABEL_CHARACTERS], [])
    result = edge_deletion_diameter_profile(graph)
    assert result.source_diameter == 0
    assert result.graph == graph


def test_native_result_rejects_labels_above_retained_character_bound() -> None:
    graph = _graph(["x" * (MAX_RETAINED_LABEL_CHARACTERS + 1)], [])
    with pytest.raises(OperationDomainValidationError, match="retained-character"):
        edge_deletion_diameter_profile(graph)


def test_entry_projections_count_toward_the_retained_label_bound() -> None:
    label = "x" * (MAX_RETAINED_LABEL_CHARACTERS // 3 + 1)
    graph = _graph([f"a{label}", f"b{label}"], [[f"a{label}", f"b{label}"]])
    with pytest.raises(OperationDomainValidationError, match="retained-character"):
        edge_deletion_diameter_profile(graph)


def test_cancellation_is_observed_during_the_deletion_loop() -> None:
    class _CancelAfter:
        def __init__(self, remaining: int) -> None:
            self.remaining = remaining

        def is_set(self) -> bool:
            if self.remaining <= 0:
                return True
            self.remaining -= 1
            return False

    graph = _graph(["0", "1", "2"], [["0", "1"], ["0", "2"], ["1", "2"]])
    with (
        request_execution(time.monotonic()),
        request_cancellation(_CancelAfter(2)),
        pytest.raises(
            OperationExecutionCancelledError,
            match="during edge-deletion diameter profile",
        ),
    ):
        edge_deletion_diameter_profile(graph)


def test_expired_owner_deadline_is_observed_before_admission() -> None:
    graph = _graph(["0", "1"], [["0", "1"]])
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(
            OperationExecutionTimeoutError,
            match="before edge-deletion diameter admission",
        ):
            edge_deletion_diameter_profile(graph)
