"""Complete fixed-length cycle enumeration tests."""

import json
import time
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
from jacobian.math.graphs.symmetry._edges import canonical_edge
from jacobian.math.graphs.values import MAX_SIMPLE_GRAPH_VERTICES, SimpleUndirectedGraph


def _square_with_diagonal() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3"),
        edges=(("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("2", "3")),
    )


def test_simple_and_chordless_cycle_families_are_distinct() -> None:
    graph = _square_with_diagonal()
    simple = enumerate_fixed_length_cycles(graph, 4)
    chordless = enumerate_chordless_fixed_length_cycles(graph, 4)

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
    result = enumerate_chordless_fixed_length_cycles(graph, 3)
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


def test_chordless_family_rejects_cycles_that_retain_a_chord() -> None:
    graph = _square_with_diagonal()
    simple = enumerate_fixed_length_cycles(graph, 4)
    payload = simple.model_dump(mode="json")
    payload["family_kind"] = "CHORDLESS"
    with pytest.raises(ValueError, match="nonconsecutive"):
        FixedLengthCycleEnumerationResult.model_validate(payload)


def _ring_edges(vertices: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (left, right) if left < right else (right, left)
        for left, right in zip(vertices, vertices[1:] + vertices[:1], strict=True)
    )


def test_bridged_rings_enumerate_per_biconnected_block() -> None:
    left = tuple(f"a{index:02d}" for index in range(15))
    right = tuple(f"b{index:02d}" for index in range(15))
    bridge = (left[0], right[0]) if left[0] < right[0] else (right[0], left[0])
    graph = SimpleUndirectedGraph(
        vertices=(*left, *right),
        edges=(*_ring_edges(left), *_ring_edges(right), bridge),
    )
    result = enumerate_fixed_length_cycles(graph, 15)
    assert result.cycle_count == 2
    families = {frozenset(cycle) for cycle in result.cycles}
    assert families == {frozenset(left), frozenset(right)}


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


def _complete_bipartite(left: int, right: int) -> SimpleUndirectedGraph:
    lefts = tuple(f"a{index}" for index in range(left))
    rights = tuple(f"b{index}" for index in range(right))
    return SimpleUndirectedGraph(
        vertices=lefts + rights,
        edges=tuple((a, b) for a in lefts for b in rights),
    )


def test_bipartite_odd_cycle_family_is_presolved_empty() -> None:
    """K26,25 has no triangle; the empty family must not pay the generic bound."""
    result = enumerate_fixed_length_cycles(_complete_bipartite(26, 25), 3)
    assert result.cycles == ()
    assert result.cycle_count == 0


def test_bipartite_infeasible_even_length_is_presolved_empty() -> None:
    """K2,9 has no 6-cycle: a part with two vertices cannot supply three."""
    result = enumerate_fixed_length_cycles(_complete_bipartite(2, 9), 6)
    assert result.cycles == ()
    assert result.cycle_count == 0


def test_mixed_blocks_keep_exact_multipartite_bound() -> None:
    """A disjoint K11,11 plus a 5-cycle keeps the exact chordless 4-cycle count."""
    lefts = tuple(f"a{index}" for index in range(11))
    rights = tuple(f"b{index}" for index in range(11))
    pentagon = tuple(f"c{index}" for index in range(5))
    edges = tuple((a, b) for a in lefts for b in rights) + tuple(
        tuple(sorted((pentagon[index], pentagon[(index + 1) % 5])))
        for index in range(5)
    )
    graph = SimpleUndirectedGraph(vertices=lefts + rights + pentagon, edges=edges)
    result = enumerate_chordless_fixed_length_cycles(graph, 4)
    assert result.cycle_count == 3025


def test_incidence_rows_are_bounded_before_decoding() -> None:
    """A forged result with excess incidence rows fails at field level."""
    graph = _complete_bipartite(2, 2)
    payload = {
        "graph": graph.model_dump(),
        "cycle_length": 4,
        "family_kind": "SIMPLE",
        "cycle_count": 1,
        "cycles": [["a0", "b0", "a1", "b1"]],
        "vertex_incidence": [
            {"source": [vertex], "cycle_indices": []} for vertex in graph.vertices
        ]
        * 100,
        "edge_incidence": [
            {"source": list(edge), "cycle_indices": []} for edge in graph.edges
        ],
    }
    with pytest.raises(ValidationError) as error:
        FixedLengthCycleEnumerationResult.model_validate(payload)
    # The outer incidence arrays are rejected at field level before any
    # structural validator walks the forged rows.
    assert error.value.errors()[0]["loc"] == ("vertex_incidence",)


def test_bipartite_higher_odd_length_is_presolved_empty() -> None:
    """K10,10 has no 5-cycle: every cycle in a bipartite graph is even."""
    result = enumerate_fixed_length_cycles(_complete_bipartite(10, 10), 5)
    assert result.cycles == ()
    assert result.cycle_count == 0


def test_exact_chordless_four_cycle_block_charges_traversal_work() -> None:
    """A near-complete multipartite block is refused before its dense DFS.

    The parts ``(2, 2, 1, ..., 1)`` span 256 vertices, so a DFS over the block
    would visit billions of terminal paths. Its traversal work must be charged
    even though the exact induced four-cycle count is one.
    """
    left_pair = ("p0a", "p0b")
    right_pair = ("p1a", "p1b")
    singletons = [(f"s{index}",) for index in range(252)]
    parts = [left_pair, right_pair, *singletons]
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        tuple(sorted((first, second)))
        for index, first in enumerate(vertices)
        for second in vertices[index + 1 :]
        if part_of[first] != part_of[second]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        enumerate_chordless_fixed_length_cycles(graph, 4)


def test_dominant_multipartite_part_makes_long_cycles_empty() -> None:
    """K_{9,1,1} has no cycle longer than four: the small parts separate two."""
    parts = [tuple(f"p{index}" for index in range(9)), ("p9",), ("p10",)]
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        tuple(sorted((first, second)))
        for index, first in enumerate(vertices)
        for second in vertices[index + 1 :]
        if part_of[first] != part_of[second]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    result = enumerate_fixed_length_cycles(graph, 6)
    assert result.cycles == ()
    assert result.cycle_count == 0


def test_oversized_cycle_row_is_bounded_before_decoding() -> None:
    """The inner cycle tuple carries a field-level maximum length."""
    graph = _complete_bipartite(2, 2)
    payload = {
        "graph": graph.model_dump(),
        "cycle_length": 4,
        "family_kind": "SIMPLE",
        "cycle_count": 1,
        "cycles": [["x"] * (MAX_SIMPLE_GRAPH_VERTICES + 1)],
        "vertex_incidence": [],
        "edge_incidence": [],
    }
    with pytest.raises(ValidationError) as error:
        FixedLengthCycleEnumerationResult.model_validate(payload)
    assert error.value.errors()[0]["loc"] == ("cycles", 0)


def _wheel(rim_count: int) -> SimpleUndirectedGraph:
    rim = tuple(f"r{index}" for index in range(rim_count))
    hub = "h"
    edges = tuple(canonical_edge(hub, vertex) for vertex in rim) + tuple(
        canonical_edge(rim[index], rim[(index + 1) % rim_count])
        for index in range(rim_count)
    )
    return SimpleUndirectedGraph(vertices=(hub, *rim), edges=edges)


def test_sparse_wheel_blocks_use_an_exact_cycle_bound() -> None:
    """A 22-vertex wheel has 21 four-cycles and no induced four-cycle."""
    graph = _wheel(21)
    simple = enumerate_fixed_length_cycles(graph, 4)
    assert simple.cycle_count == 21
    chordless = enumerate_chordless_fixed_length_cycles(graph, 4)
    assert chordless.cycle_count == 0


def test_five_vertex_wheel_keeps_the_induced_rim_cycle() -> None:
    """The four rim vertices of W_5 form an induced 4-cycle."""
    graph = _wheel(4)
    chordless = enumerate_chordless_fixed_length_cycles(graph, 4)
    assert chordless.cycle_count == 1


def test_wheel_hamiltonian_count_covers_every_rim_edge() -> None:
    """W_6 has six Hamiltonian cycles with cycle_length = rim_count + 1."""
    graph = _wheel(5)
    result = enumerate_fixed_length_cycles(graph, 6)
    assert result.cycle_count == 5


def test_wheel_block_enumerates_directly_without_dfs() -> None:
    """A 125-vertex wheel with a last-ordered hub returns 124 Hamiltonians fast."""
    rim = tuple(f"r{index:03}" for index in range(124))
    hub = "zzz"
    edges = tuple(canonical_edge(hub, vertex) for vertex in rim) + tuple(
        canonical_edge(rim[index], rim[(index + 1) % 124]) for index in range(124)
    )
    graph = SimpleUndirectedGraph(vertices=(*rim, hub), edges=edges)
    started = time.monotonic()
    result = enumerate_fixed_length_cycles(graph, 125)
    assert result.cycle_count == 124
    assert time.monotonic() - started < 2.0


def test_forged_incidence_rows_reject_before_sorting() -> None:
    """A mismatch is caught by comparison, not by sorting every forged row."""
    graph = _complete_bipartite(2, 2)
    result = enumerate_fixed_length_cycles(graph, 4)
    payload = result.model_dump(mode="json")
    # Replace every incidence row with an in-range sorted sequence that does
    # not match the real incidence; the mismatch must be rejected.
    payload["vertex_incidence"] = [
        {"source": [vertex], "cycle_indices": [0, 1]} for vertex in graph.vertices
    ]
    with pytest.raises(ValidationError):
        FixedLengthCycleEnumerationResult.model_validate(payload)


def test_bipartite_complete_four_cycles_use_the_exact_count() -> None:
    """K_{2,20} has C(2,2)*C(20,2) = 190 four-cycles."""
    result = enumerate_fixed_length_cycles(_complete_bipartite(2, 20), 4)
    assert result.cycle_count == 190


def test_multipartite_complete_four_cycles_use_the_exact_count() -> None:
    """K_{3,3,3} four-cycles match the induced multipartite count."""
    parts = [tuple(f"p{part}{index}" for index in range(3)) for part in range(3)]
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
        if part_of[left] != part_of[right]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    result = enumerate_fixed_length_cycles(graph, 4)
    assert result.cycle_count == 108


def test_multipartite_complete_triangles_use_the_exact_count() -> None:
    """K_{1,1,50} has exactly 50 simple and chordless triangles."""
    parts = (("a",), ("b",), tuple(f"c{index}" for index in range(50)))
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
        if part_of[left] != part_of[right]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    assert enumerate_fixed_length_cycles(graph, 3).cycle_count == 50
    assert enumerate_chordless_fixed_length_cycles(graph, 3).cycle_count == 50


def test_complete_graph_four_cycle_count_admits_below_the_ceiling() -> None:
    """K22 has C(22,4)*3 = 21,945 four-cycles and must be refused at admission.

    Four distinct singleton parts induce K4, which supports three four-cycles,
    so the exact count exceeds the 20,000-cycle result bound and the request is
    rejected rather than returning a result that cannot round-trip.
    """
    vertices = tuple(f"v{index:02d}" for index in range(22))
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    with pytest.raises(OperationResourceAdmissionError):
        enumerate_fixed_length_cycles(graph, 4)


def test_complete_graph_six_vertices_four_cycle_count() -> None:
    """K6 has C(6,4)*3 = 45 simple four-cycles and none induced."""
    vertices = tuple(f"v{index}" for index in range(6))
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    assert enumerate_fixed_length_cycles(graph, 4).cycle_count == 45
    assert enumerate_chordless_fixed_length_cycles(graph, 4).cycle_count == 0


def _fan(path_count: int) -> SimpleUndirectedGraph:
    path = tuple(f"r{index}" for index in range(path_count))
    hub = "h"
    edges = tuple(canonical_edge(hub, vertex) for vertex in path) + tuple(
        canonical_edge(path[index], path[index + 1]) for index in range(path_count - 1)
    )
    return SimpleUndirectedGraph(vertices=(*path, hub), edges=edges)


def test_sparse_fan_blocks_use_an_exact_cycle_bound() -> None:
    """A 22-vertex fan has 19 four-cycles and no induced four-cycle."""
    graph = _fan(21)
    assert enumerate_fixed_length_cycles(graph, 4).cycle_count == 19
    assert enumerate_chordless_fixed_length_cycles(graph, 4).cycle_count == 0
    assert enumerate_fixed_length_cycles(graph, 3).cycle_count == 20
    assert enumerate_chordless_fixed_length_cycles(graph, 3).cycle_count == 20


def test_oversized_label_rejects_before_full_string_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-ceiling label is refused without encoding or NFC-scanning it."""
    import unicodedata

    scanned: list[int] = []
    real_is_normalized = unicodedata.is_normalized

    def spy(value: str) -> bool:
        scanned.append(len(value))
        return real_is_normalized(value)

    monkeypatch.setattr(unicodedata, "is_normalized", spy)
    oversized = "x" * (100_000_000 // 2 + 1)
    forged = SimpleUndirectedGraph.model_construct(vertices=(oversized,), edges=())
    with pytest.raises(OperationResourceAdmissionError, match="retained"):
        enumerate_fixed_length_cycles(forged, 3)
    assert all(length <= 100_000_000 // 2 for length in scanned)


def test_complete_graph_four_cycle_admission_is_polynomial_time() -> None:
    """K256 is refused without enumerating C(256, 4) four-part subsets."""
    vertices = tuple(f"v{index:03d}" for index in range(256))
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
    started = time.monotonic()
    with pytest.raises(OperationResourceAdmissionError):
        enumerate_fixed_length_cycles(graph, 4)
    assert time.monotonic() - started < 2.0


def test_request_schema_publishes_the_vertex_envelope() -> None:
    """The request schema states the 256-vertex operation envelope."""
    from jacobian.math.graphs.cycle_length_profile._models import (
        FixedLengthCycleEnumerationRequest,
    )

    schema = FixedLengthCycleEnumerationRequest.model_json_schema()
    text = schema["properties"]["graph"].get("description", "")
    assert "256" in text
