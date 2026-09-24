from __future__ import annotations

import json

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    fiber_graph,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _semigroup() -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("degree",),
        generator_labels=("a", "b", "c"),
        entries=((1, 1, 1),),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=(CanonicalRational(num=1, den=1),),
    )


def test_fiber_graph_is_exactly_the_edges_induced_by_one_move() -> None:
    result = fiber_graph(_semigroup(), (2,), ((1, -1, 0),))

    # Six weak compositions of 2 into three coordinates are the fiber. Applying
    # +/- (1,-1,0) directly gives these three undirected edges.
    assert result.vertices == (
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
    )
    assert result.edges == ((1, 3), (2, 4), (4, 5))
    assert result.components == ((0,), (1, 3), (2, 4, 5))
    assert all(sum(vertex) == 2 for vertex in result.vertices)


def test_fiber_graph_normalizes_opposite_duplicate_moves() -> None:
    result = fiber_graph(_semigroup(), (2,), ((1, -1, 0), (-1, 1, 0)))
    assert result.moves == ((1, -1, 0),)
    assert result.edges == ((1, 3), (2, 4), (4, 5))


def test_fiber_graph_requires_moves_in_the_exact_relation_lattice() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        fiber_graph(_semigroup(), (2,), ((1, 0, 0),))
    assert exc_info.value.errors()[0]["type"] == (
        "affine_semigroup.graph_move_not_relation"
    )


def test_fiber_graph_tool_is_discoverable_and_round_trips() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.fiber_graph.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    decoded = tool.result_type.model_validate_json(
        json.dumps(result.model_dump(mode="json"))
    )
    assert decoded == result
