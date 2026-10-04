"""Indexed colouring scalars reject coerced integers."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    HyperedgeColorAssignment,
    IndexedHyperedgeColoring,
)


def _source() -> FiniteHypergraph:
    return FiniteHypergraph(vertices=("v",), edges=(("a", ("v",)),))


def test_native_ints_construct_a_total_colouring() -> None:
    coloring = IndexedHyperedgeColoring(
        hypergraph=_source(),
        color_count=1,
        assignments=(HyperedgeColorAssignment(edge_id="a", color_index=0),),
    )
    assert coloring.color_count == 1
    assert coloring.assignments[0].color_index == 0


@pytest.mark.parametrize("value", ["0", 1.0, True])
def test_color_index_rejects_coerced_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        HyperedgeColorAssignment.model_validate({"edge_id": "a", "color_index": value})


@pytest.mark.parametrize("value", ["0", 1.0, True])
def test_color_count_rejects_coerced_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        IndexedHyperedgeColoring.model_validate(
            {
                "hypergraph": {"vertices": ("v",), "edges": (("a", ("v",)),)},
                "color_count": value,
                "assignments": ({"edge_id": "a", "color_index": 0},),
            }
        )


def test_empty_and_duplicate_hyperedge_partition() -> None:
    empty = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(vertices=(), edges=()),
        color_count=0,
        assignments=(),
    )
    assert (
        IndexedHyperedgeColoring.model_validate_json(empty.model_dump_json()) == empty
    )
    duplicate = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=("v",), edges=(("a", ("v",)), ("b", ("v",)))
        ),
        color_count=1,
        assignments=(
            HyperedgeColorAssignment(edge_id="b", color_index=0),
            HyperedgeColorAssignment(edge_id="a", color_index=0),
        ),
    )
    assert tuple(item.edge_id for item in duplicate.assignments) == ("a", "b")


@pytest.mark.parametrize(
    "assignments,count",
    [([], 0), ([("a", 0), ("a", 0)], 1), ([("a", 1)], 2), ([("x", 0)], 1)],
)
def test_malformed_partition_rejects(
    assignments: list[tuple[str, int]], count: int
) -> None:
    with pytest.raises(ValidationError):
        IndexedHyperedgeColoring(
            hypergraph=FiniteHypergraph(vertices=("v",), edges=(("a", ("v",)),)),
            color_count=count,
            assignments=tuple(
                HyperedgeColorAssignment(edge_id=key, color_index=color)
                for key, color in assignments
            ),
        )
