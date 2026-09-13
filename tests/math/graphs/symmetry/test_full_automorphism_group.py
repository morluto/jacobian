"""Full color-preserving graph automorphism group tests."""

from itertools import pairwise
from math import factorial
from threading import Event

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.symmetry import operations
from jacobian.math.graphs.symmetry._edges import canonical_edge
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


def test_empty_source_generators_require_the_nested_identity() -> None:
    result = full_graph_automorphism_group(
        ColoredUndirectedGraph(
            graph=SimpleUndirectedGraph(
                vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
            ),
            vertex_colors=("left", "middle", "right"),
        )
    )
    payload = result.model_dump()
    payload["group"]["generators"] = [(1, 0, 2)]
    with pytest.raises(Exception, match="identity"):
        FullGraphAutomorphismResult.model_validate(payload)


def test_empty_source_generators_must_report_order_one() -> None:
    result = full_graph_automorphism_group(
        ColoredUndirectedGraph(
            graph=SimpleUndirectedGraph(
                vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
            ),
            vertex_colors=("left", "middle", "right"),
        )
    )
    payload = result.model_dump()
    payload["automorphism_count"] = 2
    payload["generated_group_order"] = 2
    with pytest.raises(Exception, match="order 1"):
        FullGraphAutomorphismResult.model_validate(payload)


def test_identity_source_generators_are_rejected() -> None:
    result = full_graph_automorphism_group(
        ColoredUndirectedGraph(
            graph=SimpleUndirectedGraph(
                vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
            ),
            vertex_colors=("left", "middle", "right"),
        )
    )
    payload = result.model_dump()
    vertices = payload["graph"]["graph"]["vertices"]
    identity = [(vertex, vertex) for vertex in vertices]
    payload["generators"] = [{"generator_id": "g0", "mapping": identity}]
    payload["group"]["generators"] = [list(range(len(vertices)))]
    with pytest.raises(Exception, match="nonidentity"):
        FullGraphAutomorphismResult.model_validate(payload)


def test_full_group_orbits_require_sorted_representatives() -> None:
    result = full_graph_automorphism_group(
        ColoredUndirectedGraph(
            graph=SimpleUndirectedGraph(
                vertices=("a", "b", "c"),
                edges=(("a", "b"), ("a", "c")),
            )
        )
    )
    payload = result.model_dump()
    payload["vertex_orbits"] = list(reversed(payload["vertex_orbits"]))
    for index, orbit in enumerate(payload["vertex_orbits"]):
        orbit["orbit_index"] = index
    with pytest.raises(Exception, match="canonical partition"):
        FullGraphAutomorphismResult.model_validate(payload)


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


@pytest.mark.parametrize("family", ["path", "cycle"])
@pytest.mark.parametrize("coloring", ["vertex", "edge"])
def test_uniform_explicit_path_and_cycle_colors_keep_compact_presentations(
    family: str,
    coloring: str,
) -> None:
    vertices = tuple(f"v{index:02}" for index in range(12))
    edges = tuple(pairwise(vertices))
    if family == "cycle":
        edges += ((vertices[0], vertices[-1]),)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=("same",) * len(vertices) if coloring == "vertex" else (),
        edge_colors=("same",) * len(edges) if coloring == "edge" else (),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == (2 if family == "path" else 24)
    assert len(result.generators) == (1 if family == "path" else 2)


def test_palindromic_colored_path_filters_bounded_reversal() -> None:
    vertices = tuple(f"v{index:02}" for index in range(15))
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(pairwise(vertices)),
        ),
        vertex_colors=tuple("A" if index % 2 == 0 else "B" for index in range(15)),
    )

    result = full_graph_automorphism_group(graph)

    # Only the identity and the endpoint reversal preserve the palindromic
    # alternating colors; generic refinement would reject this request.
    assert result.automorphism_count == 2
    assert result.generated_group_order == 2
    assert len(result.generators) == 1


def test_periodic_colored_cycle_filters_dihedral_maps() -> None:
    vertices = tuple(f"v{index}" for index in range(6))
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                sorted((vertices[index], vertices[(index + 1) % 6]))
                for index in range(6)
            ),
        ),
        vertex_colors=("A", "B") * 3,
    )

    result = full_graph_automorphism_group(graph)

    # Period-two colors keep the even rotations and one reflection coset.
    assert result.automorphism_count == 6
    assert result.generated_group_order == 6
    assert len(result.generators) == 2


def test_componentwise_colored_cliques_keep_compact_presentation() -> None:
    vertices = tuple(
        f"v{component}{position}" for component in range(3) for position in range(5)
    )
    edges = tuple(
        (left, right)
        for component in range(3)
        for left in vertices[component * 5 : component * 5 + 5]
        for right in vertices[component * 5 : component * 5 + 5]
        if left < right
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(
            f"color-{component}" for component in range(3) for _ in range(5)
        ),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 120**3
    assert result.generated_group_order == 120**3
    assert len(result.generators) == 6


def test_patterned_clique_colors_keep_compact_presentation() -> None:
    vertices = tuple(
        f"v{component}{position}" for component in range(3) for position in range(5)
    )
    edges = tuple(
        (left, right)
        for component in range(3)
        for left in vertices[component * 5 : component * 5 + 5]
        for right in vertices[component * 5 : component * 5 + 5]
        if left < right
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(
            "red" if position < 3 else "blue"
            for _component in range(3)
            for position in range(5)
        ),
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == (factorial(3) * factorial(2)) ** 3 * factorial(
        3
    )
    assert result.generated_group_order == 10_368
    assert len(result.generators) < result.automorphism_count


def test_class_pair_colored_clique_union_keeps_compact_presentation() -> None:
    vertices = tuple(
        f"v{component}{position}" for component in range(3) for position in range(5)
    )
    vertex_color = {
        vertex: "red" if int(vertex[-1]) < 3 else "blue" for vertex in vertices
    }
    edges = tuple(
        canonical_edge(left, right)
        for component in range(3)
        for left in vertices[component * 5 : component * 5 + 5]
        for right in vertices[component * 5 : component * 5 + 5]
        if left < right
    )
    edge_colors = tuple(
        "red-red"
        if vertex_color[left] == vertex_color[right] == "red"
        else "blue-blue"
        if vertex_color[left] == vertex_color[right] == "blue"
        else "mixed"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(vertex_color[vertex] for vertex in vertices),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == (factorial(3) * factorial(2)) ** 3 * factorial(
        3
    )
    assert result.generated_group_order == 10_368
    assert len(result.generators) < result.automorphism_count


def test_mixed_size_clique_union_keeps_compact_presentation() -> None:
    sizes = (5, 6, 7)
    vertices = tuple(
        f"v{component}{position}"
        for component, size in enumerate(sizes)
        for position in range(size)
    )
    starts = (0, 5, 11)
    edges = tuple(
        (left, right)
        for start, size in zip(starts, sizes, strict=True)
        for left in vertices[start : start + size]
        for right in vertices[start : start + size]
        if left < right
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(5) * factorial(6) * factorial(7)
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 6


def test_disconnected_path_degree_sequence_does_not_take_the_path_shortcut() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("c0", "c1", "c2", "k0", "k1"),
            edges=(("c0", "c1"), ("c0", "c2"), ("c1", "c2"), ("k0", "k1")),
        )
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(3) * factorial(2)
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 3


def test_clique_with_an_isolated_vertex_keeps_compact_presentation() -> None:
    clique = tuple(f"k{index}" for index in range(9))
    vertices = (*clique, "iso")
    edges = tuple((left, right) for left in clique for right in clique if left < right)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(9)
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 2


def test_complete_graph_with_class_pair_edge_colors_stays_compact() -> None:
    red = tuple(f"r{index:02d}" for index in range(10))
    blue = tuple(f"b{index:02d}" for index in range(10))
    vertices = (*red, *blue)
    vertex_color = dict.fromkeys(red, "red")
    vertex_color.update(dict.fromkeys(blue, "blue"))
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "red-red"
        if vertex_color[left] == vertex_color[right] == "red"
        else "blue-blue"
        if vertex_color[left] == vertex_color[right] == "blue"
        else "mixed"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(vertex_color[vertex] for vertex in vertices),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(10) * factorial(10)
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 4


def test_complete_graph_infers_edge_induced_classes_without_vertex_colors() -> None:
    left = tuple(f"a{index:02d}" for index in range(10))
    right = tuple(f"b{index:02d}" for index in range(10))
    vertices = (*left, *right)
    block = dict.fromkeys(left, 0)
    block.update(dict.fromkeys(right, 1))
    edges = tuple(
        canonical_edge(first, second)
        for index, first in enumerate(vertices)
        for second in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "inside" if block[first] == block[second] else "across"
        for first, second in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == (factorial(10) ** 2) * 2
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 5


def test_complete_graph_respects_vertex_colors_inside_inferred_parts() -> None:
    pairs = tuple((f"r{index}", f"b{index}") for index in range(4))
    vertices = tuple(vertex for pair in pairs for vertex in pair)
    vertex_color = {
        vertex: "red" if vertex.startswith("r") else "blue" for vertex in vertices
    }
    pair_of = {vertex: index for index, pair in enumerate(pairs) for vertex in pair}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "within" if pair_of[left] == pair_of[right] else "between"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(vertex_color[vertex] for vertex in vertices),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(4)
    assert result.generated_group_order == 24
    assert group_order(result.group) == result.automorphism_count


def test_complete_graph_uses_colored_quotient_automorphisms() -> None:
    parts = tuple((f"{label}0", f"{label}1") for label in ("a", "b", "c", "d"))
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )

    def between_color(left: str, right: str) -> str:
        distance = abs(part_of[left] - part_of[right]) % 4
        if distance in {1, 3}:
            return "adjacent"
        return "opposite"

    edge_colors = tuple(
        "within" if part_of[left] == part_of[right] else between_color(left, right)
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == 128
    assert result.generated_group_order == 128
    assert group_order(result.group) == 128


def test_uniform_vertex_colored_cliques_keep_compact_presentation() -> None:
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
        vertex_colors=("same",) * len(vertices),
    )

    result = full_graph_automorphism_group(graph)

    # A uniform vertex color removes no automorphisms: S_3 wr S_3 again.
    assert result.automorphism_count == 1_296
    assert result.generated_group_order == 1_296
    assert len(result.generators) < result.automorphism_count


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


def test_vf2_cancellation_is_checked_while_searching_between_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chorded cycle is not a simple path or cycle, so generic VF2 search
    # still owns it and its cancellation checkpoint fires mid-search.
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("c", "d")),
        ),
        edge_colors=("ab", "ac", "ad", "bc", "cd"),
    )
    cancelled = Event()

    def checkpoint(stage: str) -> None:
        request_checkpoint(stage)
        if stage == "during full graph automorphism search":
            cancelled.set()

    monkeypatch.setattr(operations, "request_checkpoint", checkpoint)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError),
    ):
        full_graph_automorphism_group(graph)


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


def test_complete_bipartite_graph_uses_complement_compact_presentation() -> None:
    """K8,8 complement is two K8 cliques; the compact presentation transfers."""
    left = tuple(f"a{index}" for index in range(8))
    right = tuple(f"b{index}" for index in range(8))
    vertices = left + right
    edges = tuple(canonical_edge(first, second) for first in left for second in right)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == (factorial(8) ** 2) * 2
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) < 20


def test_vertex_colored_pairs_keep_the_pair_blocks() -> None:
    """Ten red/blue pairs keep the diagonal S10 rather than splitting pairs."""
    pairs = tuple((f"r{index}", f"b{index}") for index in range(10))
    vertices = tuple(vertex for pair in pairs for vertex in pair)
    pair_of = {vertex: index for index, pair in enumerate(pairs) for vertex in pair}
    vertex_color = {
        vertex: "red" if vertex.startswith("r") else "blue" for vertex in vertices
    }
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "within" if pair_of[left] == pair_of[right] else "between"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(vertex_color[vertex] for vertex in vertices),
        edge_colors=edge_colors,
    )

    result = full_graph_automorphism_group(graph)

    assert result.automorphism_count == factorial(10)
    assert result.generated_group_order == result.automorphism_count
    assert group_order(result.group) == result.automorphism_count


def test_repeated_source_swap_generators_are_rejected() -> None:
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    )
    valid = full_graph_automorphism_group(graph).model_dump()
    swap = {"generator_id": "g1", "mapping": [["a", "b"], ["b", "a"]]}
    forged = {
        **valid,
        "generators": [*valid["generators"], swap],
        "group": {"degree": 2, "generators": [[1, 0], [1, 0]]},
    }
    with pytest.raises(ValidationError):
        FullGraphAutomorphismResult.model_validate(forged)


def test_tied_edge_color_partitions_resolve_canonically() -> None:
    """Red on ab and blue on cd give equal-size parts; the sorted color wins.

    Iterating the edge colors in sorted order fixes the partition choice, so the
    generator order does not depend on the interpreter hash seed.
    """
    vertices = ("a", "b", "c", "d")
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )

    def color(left: str, right: str) -> str:
        pair = frozenset((left, right))
        if pair == frozenset(("a", "b")):
            return "red"
        if pair == frozenset(("c", "d")):
            return "blue"
        return "green"

    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=tuple(color(left, right) for left, right in edges),
    )

    result = full_graph_automorphism_group(graph)

    # "blue" sorts before "red", so the cd swap is generated first.
    assert [dict(row.mapping) for row in result.generators] == [
        {"a": "a", "b": "b", "c": "d", "d": "c"},
        {"a": "b", "b": "a", "c": "c", "d": "d"},
    ]


def test_colored_dihedral_quotient_keeps_compact_presentation() -> None:
    """Ten paired parts with adjacent/nonadjacent colors on a 10-cycle.

    The quotient is dihedral of order 20 and the full group is ``2^10 * 20``;
    the bounded quotient search must admit it without scanning ``10!``.
    """
    parts = tuple((f"p{index}a", f"p{index}b") for index in range(10))
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )

    def color(left: str, right: str) -> str:
        left_part, right_part = part_of[left], part_of[right]
        if left_part == right_part:
            return "within"
        distance = (left_part - right_part) % 10
        return "adjacent" if distance in (1, 9) else "nonadjacent"

    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=tuple(color(left, right) for left, right in edges),
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == (2**10) * 20
    assert result.generated_group_order == result.automorphism_count


def test_uniform_vertex_color_keeps_complement_presentation() -> None:
    """A uniform vertex color removes no automorphism from K8,8."""
    left = tuple(f"a{index}" for index in range(8))
    right = tuple(f"b{index}" for index in range(8))
    edges = tuple(canonical_edge(first, second) for first in left for second in right)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=left + right, edges=edges),
        vertex_colors=("same",) * 16,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == (factorial(8) ** 2) * 2
    assert result.generated_group_order == result.automorphism_count


def test_reordered_generator_rows_are_rejected() -> None:
    """Swapping source generator rows with their nested permutations must fail."""
    vertices = ("a", "b", "c", "d")
    edges = (("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"))
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )
    valid = full_graph_automorphism_group(graph).model_dump()
    rows = list(valid["generators"])
    swapped = [rows[1], rows[0]]
    for index, row in enumerate(swapped):
        row["generator_id"] = f"g{index}"
    nested = valid["group"]["generators"]
    forged = {
        **valid,
        "generators": swapped,
        "group": {**valid["group"], "generators": [nested[1], nested[0]]},
    }
    with pytest.raises(ValidationError):
        FullGraphAutomorphismResult.model_validate(forged)


def test_noncanonical_vertex_order_round_trips() -> None:
    """Generators sort on the declared source axis, not the internal axis."""
    vertices = ("d", "a", "c", "b")
    edges = tuple(
        canonical_edge(vertices[index], vertices[(index + 1) % 4]) for index in range(4)
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )
    result = full_graph_automorphism_group(graph)
    reparsed = FullGraphAutomorphismResult.model_validate_json(result.model_dump_json())
    assert reparsed.generators == result.generators


def test_uniform_edge_color_keeps_complement_presentation() -> None:
    """A uniform edge color removes no automorphism from K8,8."""
    left = tuple(f"a{index}" for index in range(8))
    right = tuple(f"b{index}" for index in range(8))
    edges = tuple(canonical_edge(first, second) for first in left for second in right)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=left + right, edges=edges),
        edge_colors=("only",) * len(edges),
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == (factorial(8) ** 2) * 2
    assert result.generated_group_order == result.automorphism_count


def test_side_colored_complete_bipartite_keeps_compact_presentation() -> None:
    """K8,8 with one color per side is the compact S8 x S8."""
    left = tuple(f"a{index}" for index in range(8))
    right = tuple(f"b{index}" for index in range(8))
    edges = tuple(canonical_edge(first, second) for first in left for second in right)
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=left + right, edges=edges),
        vertex_colors=("red",) * 8 + ("blue",) * 8,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == factorial(8) ** 2
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 4


def test_independent_swap_quotient_reduces_without_recomputation() -> None:
    """A product of independent part swaps keeps a compact presentation."""
    parts = tuple((f"p{index}a", f"p{index}b") for index in range(10))
    vertices = tuple(vertex for part in parts for vertex in part)
    part_of = {vertex: index for index, part in enumerate(parts) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "within" if part_of[left] == part_of[right] else "between"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(
            "red" if vertex.endswith("a") else "blue" for vertex in vertices
        ),
        edge_colors=edge_colors,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == factorial(10)
    assert result.generated_group_order == result.automorphism_count


def test_nonuniform_edge_colors_never_use_the_complement_shortcut() -> None:
    """A nonuniform edge coloring forbids the complement-of-cliques shortcut.

    K3,3 with a red/blue side coloring and one cross edge distinguished by a
    different color has color-preserving group of order 4 (each side fixes the
    distinguished edge endpoints, so only the two permutations within the two
    non-endpoint vertices survive per side). The complement shortcut must not
    report the full ``S3 x S3``.
    """
    left = tuple(f"a{index}" for index in range(3))
    right = tuple(f"b{index}" for index in range(3))
    edges = tuple(canonical_edge(first, second) for first in left for second in right)
    edge_colors = tuple(
        "x" if (first, second) == (left[0], right[0]) else "y"
        for first, second in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=left + right, edges=edges),
        vertex_colors=("red",) * 3 + ("blue",) * 3,
        edge_colors=edge_colors,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 4
    assert result.generated_group_order == result.automorphism_count


def test_tripartite_pair_edge_colors_keep_complement_presentation() -> None:
    """K8,8,8 with per-side vertex colors and per-pair edge colors is S8^3."""
    sides = tuple(tuple(f"v{side}{index}" for index in range(8)) for side in range(3))
    vertices = sides[0] + sides[1] + sides[2]
    side_of = {vertex: side for side, part in enumerate(sides) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
        if side_of[left] != side_of[right]
    )
    pair_names = {(0, 1): "between01", (0, 2): "between02", (1, 2): "between12"}
    edge_colors = tuple(
        pair_names[tuple(sorted((side_of[left], side_of[right])))]
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(f"side{side}" for side in range(3) for _ in range(8)),
        edge_colors=edge_colors,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == factorial(8) ** 3
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 6


def test_two_signature_quotient_keeps_compact_presentation() -> None:
    """Thirty-two pairs in two vertex-color classes keep a 62-generator S-quotient."""
    pairs = tuple((f"p{index}a", f"p{index}b") for index in range(32))
    vertices = tuple(vertex for pair in pairs for vertex in pair)
    pair_of = {vertex: index for index, pair in enumerate(pairs) for vertex in pair}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    edge_colors = tuple(
        "within" if pair_of[left] == pair_of[right] else "between"
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        vertex_colors=tuple(
            "red" if pair_of[vertex] < 16 else "blue" for vertex in vertices
        ),
        edge_colors=edge_colors,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == (2**32) * (factorial(16) ** 2)
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 62


def test_distinct_pair_colors_block_quotient_side_swaps() -> None:
    """K2,2,2 with distinct per-pair edge colors has quotient identity."""
    sides = tuple(tuple(f"w{side}{index}" for index in range(2)) for side in range(3))
    vertices = sides[0] + sides[1] + sides[2]
    side_of = {vertex: side for side, part in enumerate(sides) for vertex in part}
    edges = tuple(
        canonical_edge(left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
        if side_of[left] != side_of[right]
    )
    pair_names = {(0, 1): "between01", (0, 2): "between02", (1, 2): "between12"}
    edge_colors = tuple(
        pair_names[tuple(sorted((side_of[left], side_of[right])))]
        for left, right in edges
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=edge_colors,
    )
    result = full_graph_automorphism_group(graph)
    assert result.automorphism_count == 2**3
    assert result.generated_group_order == result.automorphism_count
    assert len(result.generators) == 3
