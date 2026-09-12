"""Complete fixed-length cycle enumeration tests."""

import json

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.cycle_length_profile._models import (
    FixedLengthCycleEnumerationResult,
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

    payload = result.model_dump(mode="json")
    payload["edge_incidence"][0]["cycle_indices"] = []
    with pytest.raises(ValueError, match="edge incidence"):
        FixedLengthCycleEnumerationResult.model_validate(
            json.loads(json.dumps(payload))
        )

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


def test_exact_catalog_ids_are_published() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "graph.cycle.fixed_length.enumerate" in ids
    assert "graph.cycle.chordless_fixed_length.enumerate" in ids
