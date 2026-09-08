"""Independent set-union oracle and complete rainbow/independence equivalence."""

from collections.abc import Sequence
from itertools import combinations, product
from time import monotonic

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraphs import (
    FiniteHypergraph,
    independence_number,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    HyperedgeColorAssignment,
    IndexedHyperedgeColoring,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.same_color_conflicts import (
    SameColorConflictsResult,
    construct,
)


def coloring(
    vertices: tuple[str, ...],
    members: Sequence[tuple[str, ...]],
    colors: Sequence[int],
) -> IndexedHyperedgeColoring:
    palette = sorted(set(colors))
    return IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=vertices,
            edges=tuple((f"e{i}", edge) for i, edge in enumerate(members)),
        ),
        color_count=len(palette),
        assignments=tuple(
            HyperedgeColorAssignment(edge_id=f"e{i}", color_index=palette.index(color))
            for i, color in enumerate(colors)
        ),
    )


def check_oracle(source: IndexedHyperedgeColoring) -> SameColorConflictsResult:
    result = construct(source)
    expected = {}
    edges = source.hypergraph.edges
    for left, right in combinations(range(len(edges)), 2):
        color = source.assignments[left].color_index
        if color == source.assignments[right].color_index:
            expected[edges[left][0], edges[right][0]] = (
                frozenset(edges[left][1]) | frozenset(edges[right][1]),
                color,
            )
    conflicts = {
        edge_id: frozenset(members) for edge_id, members in result.hypergraph.edges
    }
    assert len(set(conflicts.values())) == len(conflicts)
    assert set(conflicts.values()) == {union for union, _ in expected.values()}
    assert {
        row.source_edge_ids: (conflicts[row.conflict_edge_id], row.color_index)
        for row in result.provenance
    } == expected
    assert len(result.provenance) == len(expected)
    assert [row.source_edge_ids for row in result.provenance] == list(expected)
    assert result.coloring == source
    assert result.hypergraph.vertices == source.hypergraph.vertices
    assert (
        SameColorConflictsResult.model_validate_json(result.model_dump_json()) == result
    )
    if len(source.hypergraph.vertices) <= 8:
        for size in range(len(source.hypergraph.vertices) + 1):
            for subset in combinations(source.hypergraph.vertices, size):
                selected = set(subset)
                contained_colors = [
                    entry.color_index
                    for (_, members), entry in zip(
                        edges, source.assignments, strict=True
                    )
                    if set(members) <= selected
                ]
                rainbow = len(contained_colors) == len(set(contained_colors))
                independent = not any(union <= selected for union in conflicts.values())
                assert rainbow == independent
    return result


@pytest.mark.parametrize("colors", tuple(product(range(2), repeat=4)))
def test_all_two_color_assignments_with_nonuniform_and_empty_edges(
    colors: tuple[int, ...],
) -> None:
    check_oracle(
        coloring(("a", "b", "c"), [(), ("a",), ("a", "b"), ("b", "c")], colors)
    )


@pytest.mark.parametrize(
    "vertices,members,colors",
    [
        ((), [], []),
        (("a", "b"), [], []),
        (("a",), [("a",)], [0]),
        (("a",), [(), ()], [0, 0]),
        (("a", "b"), [("a",), ("a",), ("b",), ("b",)], [0, 0, 1, 1]),
        (("a", "b", "c"), [("a", "b"), ("a", "c"), ("b", "c")], [0, 0, 0]),
    ],
)
def test_empty_duplicate_sources_and_duplicate_unions(
    vertices: tuple[str, ...], members: list[tuple[str, ...]], colors: list[int]
) -> None:
    check_oracle(coloring(vertices, members, colors))


def test_deserialized_provenance_uses_source_axis_order() -> None:
    result = check_oracle(
        coloring(("a", "b", "c"), [("a",), ("b",), ("c",)], [0, 0, 0])
    )
    payload = result.model_dump()
    payload["provenance"] = [
        {
            **row,
            "source_edge_ids": (row["source_edge_ids"][1], row["source_edge_ids"][0]),
        }
        for row in reversed(payload["provenance"])
    ]
    payload["provenance"].append(payload["provenance"][0])
    assert SameColorConflictsResult.model_validate(payload) == result
    payload["provenance"][0] = {
        **payload["provenance"][0],
        "conflict_edge_id": result.provenance[0].conflict_edge_id,
    }
    with pytest.raises(ValidationError, match="conflicting duplicate"):
        SameColorConflictsResult.model_validate(payload)


def test_deserialized_provenance_must_match_source_edge_colors() -> None:
    result = check_oracle(
        coloring(("a", "b"), [("a",), ("b",), ("a",), ("b",)], [0, 0, 1, 1])
    )
    mismatched = result.model_dump()
    mismatched["provenance"][0]["color_index"] = 1
    with pytest.raises(ValidationError, match="match both referenced source edges"):
        SameColorConflictsResult.model_validate(mismatched)
    mixed = result.model_dump()
    mixed["provenance"] = [{**mixed["provenance"][0], "source_edge_ids": ("e0", "e2")}]
    with pytest.raises(ValidationError, match="match both referenced source edges"):
        SameColorConflictsResult.model_validate(mixed)


def test_deserialized_provenance_must_be_complete_and_name_unions() -> None:
    result = check_oracle(
        coloring(("a", "b", "c"), [("a",), ("b",), ("c",)], [0, 0, 0])
    )
    omitted = result.model_dump()
    omitted["provenance"] = omitted["provenance"][1:]
    with pytest.raises(ValidationError, match="every same-colour source pair"):
        SameColorConflictsResult.model_validate(omitted)
    swapped = result.model_dump()
    first, last = swapped["provenance"][0], swapped["provenance"][-1]
    swapped["provenance"][0] = {
        **first,
        "conflict_edge_id": last["conflict_edge_id"],
    }
    with pytest.raises(ValidationError, match="union of the two source edges"):
        SameColorConflictsResult.model_validate(swapped)


def test_multiple_colors_can_produce_the_same_union() -> None:
    result = check_oracle(
        coloring(("a", "b"), [("a",), ("b",), ("a",), ("b",)], [0, 0, 1, 1])
    )
    assert result.hypergraph.edges == (("c0", ("a", "b")),)
    assert [row.color_index for row in result.provenance] == [0, 1]


def test_coherent_vertex_and_source_edge_relabeling() -> None:
    source = coloring(("z", "a", "b"), [("z", "a"), ("a", "b"), ("b",)], [0, 0, 0])
    original = check_oracle(source)
    labels = {"z": "e\u0301", "a": "é", "b": ""}
    renamed = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=tuple(labels[v] for v in source.hypergraph.vertices),
            edges=tuple(
                (f"renamed-{edge_id}", tuple(labels[v] for v in members))
                for edge_id, members in source.hypergraph.edges
            ),
        ),
        color_count=source.color_count,
        assignments=tuple(
            HyperedgeColorAssignment(
                edge_id=f"renamed-{row.edge_id}", color_index=row.color_index
            )
            for row in reversed(source.assignments)
        ),
    )
    result = check_oracle(renamed)
    assert result.hypergraph.edges == tuple(
        (edge_id, tuple(sorted(labels[v] for v in members)))
        for edge_id, members in original.hypergraph.edges
    )
    assert [row.conflict_edge_id for row in result.provenance] == [
        row.conflict_edge_id for row in original.provenance
    ]


def test_all_source_edges_with_distinct_colors_are_accepted() -> None:
    source = coloring(("v",), [("v",)] * 12_000, list(range(12_000)))
    result = construct(source)
    assert result.hypergraph.edges == ()
    assert result.provenance == ()
    assert independence_number(result.hypergraph).independence_number == 1


def test_many_duplicate_sources_fit_one_union_at_pair_boundary() -> None:
    result = construct(coloring(("v",), [("v",)] * 362, [0] * 362))
    assert result.hypergraph.edges == (("c0", ("v",)),)
    assert len(result.provenance) == 65_341
    assert result.provenance[0].source_edge_ids == ("e0", "e1")
    assert result.provenance[-1].source_edge_ids == ("e360", "e361")


def test_complete_provenance_bound_rejects() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="provenance"):
        construct(coloring(("v",), [("v",)] * 363, [0] * 363))


def test_distinct_union_carrier_bound_rejects() -> None:
    vertices = tuple(f"v{i}" for i in range(160))
    with pytest.raises(OperationResourceAdmissionError, match="12000-edge"):
        construct(coloring(vertices, [(v,) for v in vertices], [0] * 160))


def test_union_incidence_bound_rejects() -> None:
    vertices = tuple(f"v{i}" for i in range(130))
    members = [(*vertices[:100], vertices[i]) for i in range(100, 130)]
    with pytest.raises(OperationResourceAdmissionError, match="36000-incidence"):
        construct(coloring(vertices, members, [0] * 30))


@pytest.mark.parametrize("expired_bound", [False, True])
def test_expired_request_context(expired_bound: bool) -> None:
    source = coloring(("v",), [("v",)] * 2, [0, 0])
    with request_execution(monotonic() if expired_bound else monotonic() - 100):
        if expired_bound:
            bind_request_deadline(monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            construct(source)


def test_native_all_distinct_distances_compose_to_full_independent_set() -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.geometry.exact._models import (
        LabelledRationalPoint,
        PointConfiguration,
    )
    from jacobian.math.geometry.exact.distance_edge_coloring import (
        compute_distance_edge_coloring,
    )

    source = PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=f"p{i}", coordinates=(CanonicalRational(num=x, den=1),)
            )
            for i, x in enumerate((0, 1, 3, 7))
        )
    )
    distances = compute_distance_edge_coloring(source)
    result = check_oracle(
        IndexedHyperedgeColoring.model_validate_json(
            distances.coloring.model_dump_json()
        )
    )
    assert result.hypergraph.edges == ()
    independent = independence_number(result.hypergraph)
    assert independent.status == "EXACT"
    assert independent.independence_number == 4
