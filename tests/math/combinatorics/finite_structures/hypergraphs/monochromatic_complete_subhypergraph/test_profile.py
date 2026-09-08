"""Behavioral tests for complete monochromatic uniform-subhypergraph profiles."""

from itertools import combinations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    HyperedgeColorAssignment,
    IndexedHyperedgeColoring,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.monochromatic_complete_subhypergraph import (
    MonochromaticCompleteSubhypergraphProfile,
    construct,
)


def make_coloring(
    vertices: tuple[str, ...],
    members: list[tuple[str, ...]],
    colors: list[int],
) -> IndexedHyperedgeColoring:
    assert len(members) == len(colors)
    color_count = max(colors, default=-1) + 1
    return IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=vertices,
            edges=tuple((f"e{i}", edge) for i, edge in enumerate(members)),
        ),
        color_count=color_count,
        assignments=tuple(
            HyperedgeColorAssignment(edge_id=f"e{i}", color_index=color)
            for i, color in enumerate(colors)
        ),
    )


def complete_edges(vertices: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [tuple(edge) for edge in combinations(vertices, size)]


def test_all_red_k4_3_uniform_source_has_one_profile_candidate() -> None:
    source = make_coloring(
        ("0", "1", "2", "3"),
        complete_edges(("0", "1", "2", "3"), 3),
        [0] * 4,
    )

    result = construct(source, 3, 4)

    assert result.hypergraph.edges == (("c0", ("0", "1", "2", "3")),)
    assert result.candidate_colors == (0,)
    assert result.source_edge_witnesses == (("e0", "e1", "e2", "e3"),)
    assert (
        MonochromaticCompleteSubhypergraphProfile.model_validate_json(
            result.model_dump_json()
        )
        == result
    )


def test_recolouring_one_source_edge_removes_candidate() -> None:
    source = make_coloring(
        ("0", "1", "2", "3"),
        complete_edges(("0", "1", "2", "3"), 3),
        [0, 0, 0, 1],
    )

    result = construct(source, 3, 4)

    assert result.hypergraph.edges == ()
    assert result.candidate_colors == ()
    assert result.source_edge_witnesses == ()


def test_missing_source_edge_is_not_given_an_implicit_colour() -> None:
    source = make_coloring(
        ("0", "1", "2", "3"),
        complete_edges(("0", "1", "2", "3"), 3)[:-1],
        [0] * 3,
    )

    assert construct(source, 3, 4).hypergraph.edges == ()


def test_complete_two_colouring_matches_independent_subset_oracle() -> None:
    vertices = tuple(str(i) for i in range(5))
    members = complete_edges(vertices, 3)
    colors = [index % 2 for index, _ in enumerate(members)]
    source = make_coloring(vertices, members, colors)

    result = construct(source, 3, 4)
    expected: dict[tuple[str, ...], int] = {}
    lookup = {
        frozenset(edge): color for edge, color in zip(members, colors, strict=True)
    }
    for target in combinations(vertices, 4):
        target_colors = {lookup[frozenset(edge)] for edge in combinations(target, 3)}
        if len(target_colors) == 1:
            expected[target] = target_colors.pop()

    assert {
        members: color
        for (_, members), color in zip(
            result.hypergraph.edges, result.candidate_colors, strict=True
        )
    } == expected


def test_target_equal_source_uniformity_returns_each_source_edge() -> None:
    source = make_coloring(
        ("a", "b", "c", "d"),
        [("c", "a", "b"), ("d", "a", "b")],
        [1, 0],
    )

    result = construct(source, 3, 3)

    assert result.hypergraph.edges == (
        ("c0", ("a", "b", "c")),
        ("c1", ("a", "b", "d")),
    )
    assert result.candidate_colors == (1, 0)
    assert result.source_edge_witnesses == (("e0",), ("e1",))


def test_sparse_256_vertex_source_uses_target_equal_uniformity_fastpath() -> None:
    vertices = tuple(str(index) for index in range(256))
    members = tuple(str(index) for index in range(128))
    source = make_coloring(vertices, [members], [0])

    result = construct(source, 128, 128)

    assert result.hypergraph.edges == (("c0", tuple(sorted(members))),)
    assert result.candidate_colors == (0,)
    assert result.source_edge_witnesses == (("e0",),)


@pytest.mark.parametrize("source_edges", [[], [(tuple(str(i) for i in range(128)))]])
def test_fewer_source_edges_than_required_returns_exact_empty_profile(
    source_edges: list[tuple[str, ...]],
) -> None:
    vertices = tuple(str(index) for index in range(256))
    source = make_coloring(vertices, source_edges, [0] * len(source_edges))

    result = construct(source, 128, 129)

    assert result.hypergraph.edges == ()
    assert result.candidate_colors == ()
    assert result.source_edge_witnesses == ()


def test_source_edge_storage_permutation_preserves_target_profile() -> None:
    vertices = ("0", "1", "2", "3")
    members = complete_edges(vertices, 3)
    source = make_coloring(vertices, members, [0, 1, 0, 1])
    permutation = (2, 0, 3, 1)
    permuted = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=vertices,
            edges=tuple(
                (f"p{i}", members[index]) for i, index in enumerate(permutation)
            ),
        ),
        color_count=2,
        assignments=tuple(
            HyperedgeColorAssignment(edge_id=f"p{i}", color_index=[0, 1, 0, 1][index])
            for i, index in enumerate(permutation)
        ),
    )

    original = construct(source, 3, 4)
    reordered = construct(permuted, 3, 4)

    assert reordered.hypergraph.edges == original.hypergraph.edges
    assert reordered.candidate_colors == original.candidate_colors


def test_empty_source_edges_still_use_explicit_source_uniformity() -> None:
    source = make_coloring(("a", "b", "c", "d"), [], [])

    result = construct(source, 3, 4)

    assert result.hypergraph.edges == ()
    assert result.source_uniformity == 3


@pytest.mark.parametrize(
    "source_uniformity,target_uniformity",
    [(3, 2), (3, 5)],
)
def test_uniformity_domain_is_explicit(
    source_uniformity: int, target_uniformity: int
) -> None:
    source = make_coloring(("a", "b", "c", "d"), [], [])

    with pytest.raises(OperationDomainValidationError):
        construct(source, source_uniformity, target_uniformity)


def test_nonuniform_source_is_rejected() -> None:
    source = make_coloring(("a", "b", "c"), [("a",), ("a", "b")], [0, 0])

    with pytest.raises(OperationDomainValidationError, match="source edge"):
        construct(source, 1, 2)


def test_duplicate_source_vertex_sets_are_rejected() -> None:
    source = make_coloring(("a", "b", "c"), [("a", "b"), ("b", "a")], [0, 0])

    with pytest.raises(OperationDomainValidationError, match="once"):
        construct(source, 2, 2)


def test_source_sensitive_candidate_bound_rejects_large_possible_profile() -> None:
    vertices = tuple(str(index) for index in range(200))
    source = make_coloring(vertices, [(vertex,) for vertex in vertices], [0] * 200)

    with pytest.raises(OperationResourceAdmissionError, match="12000-edge"):
        construct(source, 1, 2)
