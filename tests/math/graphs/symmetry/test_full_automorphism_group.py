"""Full color-preserving graph automorphism group tests."""

import pytest

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.symmetry import operations
from jacobian.math.graphs.symmetry._models import FullGraphAutomorphismResult
from jacobian.math.graphs.symmetry.operations import (
    full_graph_automorphism_group,
    graph_symmetry_orbits,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph
from jacobian.math.groups.operations import group_order


def test_cycle_four_has_dihedral_automorphism_group() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3"),
            edges=(("0", "1"), ("0", "3"), ("1", "2"), ("2", "3")),
        )
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 8
    assert result.generated_group_order == 8
    assert len(result.generators) == 2


def test_empty_graph_returns_the_composable_degree_zero_identity() -> None:
    graph = ColoredUndirectedGraph(graph=SimpleUndirectedGraph(vertices=(), edges=()))

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 1
    assert result.group.degree == 0
    assert result.group.generators == ((),)
    assert result.vertex_orbits == ()
    assert result.edge_orbits == ()


def test_native_operation_rejects_unparsed_graph_values() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        full_graph_automorphism_group({})  # type: ignore[arg-type]

    assert error.value.errors()[0]["type"] == "graph.automorphism.graph_type"


def test_vertex_colors_restrict_the_full_group() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
        ),
        vertex_colors=("left", "middle", "right"),
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 1
    assert result.generators == ()


def test_complete_graph_aligns_declared_colors_with_sorted_action_axis() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("b", "a", "c"),
            edges=(("a", "b"), ("a", "c"), ("b", "c")),
        ),
        vertex_colors=("left", "right", "left"),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 2
    assert result.generators[0].mapping == (
        ("b", "c"),
        ("a", "a"),
        ("c", "b"),
    )
    assert tuple(orbit.members for orbit in result.vertex_orbits) == (
        ("a",),
        ("b", "c"),
    )


@pytest.mark.parametrize("complete", [False, True])
def test_complete_and_empty_graphs_use_compact_symmetric_generators(
    complete: bool,
) -> None:
    vertices = ("a", "b", "c", "d")
    edges = (
        tuple(
            (left, right)
            for index, left in enumerate(vertices)
            for right in vertices[index + 1 :]
        )
        if complete
        else ()
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 24
    assert group_order(result.group) == 24
    assert len(result.generators) <= 2


def test_edge_colors_can_break_an_uncolored_cycle_symmetry() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")),
        ),
        edge_colors=("ab", "ad", "bc", "cd"),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 1
    assert result.generators == ()


def test_vf2_accepts_and_uses_refined_edge_signatures() -> None:
    vertices = tuple("v" + str(index) for index in range(10))
    edges = tuple(
        sorted((vertices[index], vertices[(index + 1) % len(vertices)]))
        for index in range(len(vertices))
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=("same",) * len(vertices),
        edge_colors=tuple(f"edge-{index}" for index in range(len(edges))),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 1
    assert result.generators == ()


def test_group_and_source_generators_compose_after_result_serialization() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")),
        )
    )
    result = full_graph_automorphism_group(graph)
    reconstructed = FullGraphAutomorphismResult.model_validate_json(
        result.model_dump_json()
    )

    replay = graph_symmetry_orbits(reconstructed.graph, reconstructed.generators)
    assert group_order(reconstructed.group) == reconstructed.automorphism_count
    assert replay.vertex_orbits == reconstructed.vertex_orbits
    assert replay.edge_orbits == reconstructed.edge_orbits
    assert "completeness" not in reconstructed.model_dump()


def test_repeated_three_cliques_use_a_compact_complete_presentation() -> None:
    vertices = tuple(
        f"v{component}{position}" for component in range(3) for position in range(3)
    )
    edges = tuple(
        (left, right)
        for component in range(3)
        for left in vertices[component * 3 : component * 3 + 3]
        for right in vertices[component * 3 : component * 3 + 3]
        if left < right
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=("edge",) * len(edges),
    )

    result = full_graph_automorphism_group(graph)

    # S_3 wr S_3: each clique can be permuted and the three components can
    # be exchanged, without enumerating all 1,296 automorphisms.
    assert result.automorphism_count == 1_296
    assert result.generated_group_order == 1_296
    assert result.group.degree == 9
    assert len(result.generators) < result.automorphism_count


def test_backend_checkpoint_errors_are_not_reclassified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"),),
        )
    )

    def timeout(_: str) -> None:
        raise OperationExecutionTimeoutError("deadline")

    monkeypatch.setattr(operations, "request_checkpoint", timeout)
    with pytest.raises(OperationExecutionTimeoutError):
        full_graph_automorphism_group(graph)


def test_dense_graph_edge_scan_work_is_rejected_before_enumeration() -> None:
    vertices = tuple(f"v{i:03}" for i in range(256))
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        ),
        vertex_colors=("free",) * 8 + tuple(f"fixed-{i}" for i in range(248)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        full_graph_automorphism_group(graph)
