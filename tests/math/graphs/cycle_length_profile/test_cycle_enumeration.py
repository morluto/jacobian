"""Complete fixed-length cycle enumeration tests."""

import json
from threading import Event

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.cycle_length_profile import operations as cycle_operations
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleFamilyKind,
    FixedLengthCycleEnumerationResult,
    is_dihedral_canonical_cycle,
)
from jacobian.math.graphs.cycle_length_profile._tools import TOOLS
from jacobian.math.graphs.cycle_length_profile.operations import (
    enumerate_chordless_fixed_length_cycles,
    enumerate_fixed_length_cycles,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _square_with_diagonal() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3"),
        edges=(("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("2", "3")),
    )


def test_simple_and_chordless_cycle_families_are_distinct() -> None:
    graph = _square_with_diagonal()
    simple = enumerate_fixed_length_cycles(graph, 4)
    chordless = enumerate_fixed_length_cycles(graph, 4, chordless=True)

    assert simple.cycles == (("0", "1", "2", "3"),)
    assert chordless.cycles == ()
    assert simple.family_kind is CycleFamilyKind.SIMPLE
    assert chordless.family_kind is CycleFamilyKind.CHORDLESS
    assert (
        FixedLengthCycleEnumerationResult.model_validate_json(
            simple.model_dump_json()
        ).family_kind
        is CycleFamilyKind.SIMPLE
    )
    assert (
        FixedLengthCycleEnumerationResult.model_validate_json(
            chordless.model_dump_json()
        ).family_kind
        is CycleFamilyKind.CHORDLESS
    )
    assert simple.vertex_incidence[0].cycle_indices == (0,)
    assert simple.edge_incidence[1].source == ("0", "2")
    assert simple.edge_incidence[1].cycle_indices == ()


def test_complete_graph_triangle_enumeration_matches_binomial_count() -> None:
    vertices = ("0", "1", "2", "3", "4")
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right) for left in vertices for right in vertices if left < right
        ),
    )
    result = enumerate_fixed_length_cycles(graph, 3, chordless=True)
    assert len(result.cycles) == 10
    assert all(
        cycle == min(cycle, (cycle[0], cycle[2], cycle[1])) for cycle in result.cycles
    )


def test_cycle_and_chordless_families_cover_requested_fixtures() -> None:
    vertices = tuple("abcdef")
    edges = (
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("a", "d"),
        ("b", "e"),
        ("e", "f"),
        ("b", "f"),
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

    simple = enumerate_fixed_length_cycles(graph, 4)
    assert simple.cycles == (("a", "b", "c", "d"),)
    assert simple.cycle_count == 1
    assert simple.edge_incidence[0].cycle_indices == (0,)

    chordless = enumerate_chordless_fixed_length_cycles(graph, 4)
    assert chordless.cycles == simple.cycles

    shared_edge = enumerate_fixed_length_cycles(
        SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(
                ("a", "b"),
                ("a", "c"),
                ("b", "c"),
                ("a", "d"),
                ("b", "d"),
            ),
        ),
        3,
    )
    assert shared_edge.cycle_count == 2
    assert shared_edge.edge_incidence[0].cycle_indices == (0, 1)


def test_tree_bridges_and_length_above_order_are_exact_empty_presolves() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    assert enumerate_fixed_length_cycles(graph, 3).cycles == ()
    assert enumerate_fixed_length_cycles(graph, 8).cycles == ()
    assert (
        enumerate_fixed_length_cycles(
            SimpleUndirectedGraph(vertices=("a", "b"), edges=()), 3
        ).cycles
        == ()
    )


def test_complete_family_output_bound_is_admitted_before_search() -> None:
    vertices = tuple(f"v{i:02}" for i in range(51))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right)
            for index, left in enumerate(vertices)
            for right in vertices[index + 1 :]
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="result envelope"):
        enumerate_fixed_length_cycles(graph, 3)


def test_sparse_ring_long_cycle_is_admitted_by_topology_bounds() -> None:
    vertices = tuple(str(index) for index in range(9))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[index], vertices[(index + 1) % 9]) if index < 8 else ("0", "8")
            for index in range(9)
        ),
    )
    expected = (tuple(str(index) for index in range(9)),)

    simple = enumerate_fixed_length_cycles(graph, 9)
    assert simple.cycles == expected
    assert simple.cycle_count == 1

    chordless = enumerate_chordless_fixed_length_cycles(graph, 9)
    assert chordless.cycles == expected


def test_direct_native_noncanonical_labels_use_typed_domain_errors() -> None:
    decomposed = SimpleUndirectedGraph.model_construct(
        vertices=("é", "a"),
        edges=(("a", "é"),),
    )
    with pytest.raises(OperationDomainValidationError, match="Unicode NFC"):
        enumerate_fixed_length_cycles(decomposed, 3)

    surrogate = SimpleUndirectedGraph.model_construct(
        vertices=("\ud800", "a"),
        edges=(("a", "\ud800"),),
    )
    with pytest.raises(OperationDomainValidationError, match="scalar values"):
        enumerate_fixed_length_cycles(surrogate, 3)


def test_direct_native_invalid_inputs_use_typed_domain_errors() -> None:
    with pytest.raises(OperationDomainValidationError, match="canonical simple"):
        enumerate_fixed_length_cycles(object(), 3)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError, match="integer"):
        enumerate_fixed_length_cycles(
            SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=()),
            True,
        )


def test_serialized_family_checks_axes_and_incidence_without_replaying_edges() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )
    result = enumerate_fixed_length_cycles(graph, 3)
    restored = FixedLengthCycleEnumerationResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    chordless = enumerate_chordless_fixed_length_cycles(graph, 3)
    restored_chordless = FixedLengthCycleEnumerationResult.model_validate_json(
        chordless.model_dump_json()
    )
    assert restored_chordless == chordless
    assert restored_chordless.family_kind is CycleFamilyKind.CHORDLESS

    payload = result.model_dump(mode="json")
    payload["edge_incidence"][0]["cycle_indices"] = []
    with pytest.raises(ValueError, match="edge incidence"):
        FixedLengthCycleEnumerationResult.model_validate(
            json.loads(json.dumps(payload))
        )

    oversized = result.model_dump(mode="json")
    oversized["vertex_incidence"][0]["cycle_indices"] = list(range(20_001))
    with pytest.raises(ValidationError):
        FixedLengthCycleEnumerationResult.model_validate(oversized)

    square = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")),
    )
    square_result = enumerate_fixed_length_cycles(square, 4)
    forged_cycle = square_result.model_dump(mode="json")
    forged_cycle["cycles"] = [["a", "b", "d", "c"]]
    forged_cycle["cycle_count"] = 1
    forged_cycle["vertex_incidence"] = [
        {"source": [vertex], "cycle_indices": [0]} for vertex in square.vertices
    ]
    forged_cycle["edge_incidence"] = [
        {
            "source": list(edge),
            "cycle_indices": [0] if edge in (("a", "b"), ("c", "d")) else [],
        }
        for edge in square.edges
    ]
    with pytest.raises(ValueError, match="close through declared graph edges"):
        FixedLengthCycleEnumerationResult.model_validate(
            json.loads(json.dumps(forged_cycle))
        )


def test_large_serialized_family_round_trips_with_linear_incidence_checks() -> None:
    vertices = tuple(f"v{i:02}" for i in range(50))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right)
            for index, left in enumerate(vertices)
            for right in vertices[index + 1 :]
        ),
    )
    result = enumerate_fixed_length_cycles(graph, 3)

    restored = FixedLengthCycleEnumerationResult.model_validate_json(
        result.model_dump_json()
    )

    assert restored.cycle_count == 19_600
    assert restored.cycles == result.cycles


def test_complete_graph_chordless_four_cycles_are_empty_without_simple_bound() -> None:
    vertices = tuple(str(index) for index in range(22))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right) for left in vertices for right in vertices if left < right
        ),
    )

    result = enumerate_chordless_fixed_length_cycles(graph, 4)

    assert result.cycles == ()
    assert result.cycle_count == 0
    assert result.family_kind is CycleFamilyKind.CHORDLESS
    assert len(result.vertex_incidence) == 22
    assert len(result.edge_incidence) == 231


def test_complete_bipartite_chordless_four_cycles_use_the_exact_count() -> None:
    left = tuple(f"a{index:02}" for index in range(11))
    right = tuple(f"b{index:02}" for index in range(11))
    vertices = left + right
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((a, b) for a in left for b in right),
    )

    result = enumerate_chordless_fixed_length_cycles(graph, 4)

    assert result.cycle_count == 55 * 55
    assert result.family_kind is CycleFamilyKind.CHORDLESS
    assert len(result.cycles) == 3_025


def test_disjoint_complete_bipartite_chordless_four_cycles_sum_component_bounds() -> (
    None
):
    def bipartite(prefix: str) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        left = tuple(f"{prefix}a{index:02}" for index in range(11))
        right = tuple(f"{prefix}b{index:02}" for index in range(11))
        return left + right, tuple((a, b) for a in left for b in right)

    first_vertices, first_edges = bipartite("p")
    second_vertices, second_edges = bipartite("q")
    graph = SimpleUndirectedGraph(
        vertices=first_vertices + second_vertices,
        edges=first_edges + second_edges,
    )

    result = enumerate_chordless_fixed_length_cycles(graph, 4)

    assert result.cycle_count == 2 * 55 * 55
    assert len(result.cycles) == 6_050


def test_bridged_complete_bipartite_chordless_four_cycles_sum_block_bounds() -> None:
    def bipartite(
        prefix: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
        left = tuple(f"{prefix}a{index:02}" for index in range(11))
        right = tuple(f"{prefix}b{index:02}" for index in range(11))
        return left, right, tuple((a, b) for a in left for b in right)

    left_p, _right_p, first_edges = bipartite("p")
    left_q, _right_q, second_edges = bipartite("q")
    first_vertices = left_p + _right_p
    second_vertices = left_q + _right_q
    bridge = (left_p[0], left_q[0]) if left_p[0] < left_q[0] else (left_q[0], left_p[0])
    graph = SimpleUndirectedGraph(
        vertices=first_vertices + second_vertices,
        edges=first_edges + second_edges + (bridge,),
    )

    result = enumerate_chordless_fixed_length_cycles(graph, 4)

    assert result.cycle_count == 2 * 55 * 55
    assert len(result.cycles) == 6_050


def test_oversized_forged_graph_rejects_before_label_validation() -> None:
    vertices = tuple(f"v{index}" for index in range(257))
    graph = SimpleUndirectedGraph.model_construct(vertices=vertices, edges=())
    with pytest.raises(OperationDomainValidationError, match="at most 256"):
        enumerate_fixed_length_cycles(graph, 3)


def test_empty_clique_family_honors_cancellation_during_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = tuple(f"v{index:03}" for index in range(64))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right) for left in vertices for right in vertices if left < right
        ),
    )
    cancelled = Event()

    def checkpoint(stage: str) -> None:
        request_checkpoint(stage)
        if stage == "during empty fixed-length cycle incidence assembly":
            cancelled.set()

    monkeypatch.setattr(cycle_operations, "request_checkpoint", checkpoint)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError),
    ):
        enumerate_chordless_fixed_length_cycles(graph, 4)


def test_complete_bipartite_chordless_six_cycles_are_empty() -> None:
    left = tuple(f"a{index}" for index in range(8))
    right = tuple(f"b{index}" for index in range(8))
    vertices = left + right
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((a, b) for a in left for b in right),
    )

    result = enumerate_chordless_fixed_length_cycles(graph, 6)

    assert result.cycles == ()
    assert result.cycle_count == 0
    assert result.family_kind is CycleFamilyKind.CHORDLESS


def test_cycle_canonicality_uses_the_minimum_vertex_and_two_orientations() -> None:
    assert not is_dihedral_canonical_cycle(("c", "a", "b"))
    assert is_dihedral_canonical_cycle(("a", "b", "c"))
    assert not is_dihedral_canonical_cycle(("a", "c", "b"))


def test_incidence_assembly_honors_cancellation_after_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = tuple(str(index) for index in range(8))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (left, right) for left in vertices for right in vertices if left < right
        ),
    )
    cancelled = Event()

    def checkpoint(stage: str) -> None:
        request_checkpoint(stage)
        if stage == "during fixed-length cycle incidence assembly":
            cancelled.set()

    monkeypatch.setattr(cycle_operations, "request_checkpoint", checkpoint)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError),
    ):
        enumerate_fixed_length_cycles(graph, 3)


def test_exact_catalog_ids_are_published() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "graph.cycle.fixed_length.enumerate" in ids
    assert "graph.cycle.chordless_fixed_length.enumerate" in ids
