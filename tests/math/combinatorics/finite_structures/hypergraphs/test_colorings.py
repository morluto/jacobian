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
    assert (
        IndexedHyperedgeColoring.model_validate_json(coloring.model_dump_json())
        == coloring
    )


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
