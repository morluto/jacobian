from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    identity_simplicial_map,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _identity(simplex_set):
    return TruncatedSimplicialMap(
        source=simplex_set,
        target=simplex_set,
        maps=tuple(tuple(range(len(level))) for level in simplex_set.sets),
    )


def _collapse_first_edge(simplex_set):
    lookup = [
        {
            tuple(int(item) for item in simplex[1:-1].split(",")): index
            for index, simplex in enumerate(level)
        }
        for level in simplex_set.sets
    ]
    maps = tuple(
        tuple(
            lookup[degree][
                tuple(
                    0 if vertex <= 1 else 2
                    for vertex in (int(item) for item in simplex[1:-1].split(","))
                )
            ]
            for simplex in level
        )
        for degree, level in enumerate(simplex_set.sets)
    )
    return TruncatedSimplicialMap(
        source=simplex_set, target=simplex_set, maps=tuple(maps)
    )


def test_composition_uses_degreewise_function_composition_and_is_associative():
    simplex_set = standard_simplex(2, 2)
    identity = _identity(simplex_set)
    collapse = _collapse_first_edge(simplex_set)

    identity_after_collapse = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=collapse, second=identity)
    )
    collapse_after_identity = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=identity, second=collapse)
    )
    assert identity_after_collapse.maps == collapse.maps
    assert collapse_after_identity.maps == collapse.maps

    left = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=identity, second=collapse)
    )
    left = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=left, second=identity)
    )
    right = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=collapse, second=identity)
    )
    right = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=identity, second=right)
    )
    assert left.maps == right.maps == collapse.maps

    # Independent oracle: composition is ordinary function composition in
    # each degree, in the order f then g.
    for degree, row in enumerate(identity.maps):
        expected = tuple(collapse.maps[degree][index] for index in row)
    assert collapse_after_identity.maps[degree] == expected


def test_identity_constructor_preserves_carrier_and_composes_on_both_sides():
    simplex_set = standard_simplex(1, 2)
    identity = identity_simplicial_map(simplex_set)

    assert identity.source == simplex_set
    assert identity.target == simplex_set
    assert identity.maps == ((0, 1), (0, 1, 2), (0, 1, 2, 3))
    assert (
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(first=identity, second=identity)
        )
        == identity
    )
    assert (
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(
                first=identity, second=_collapse_first_edge(simplex_set)
            )
        ).maps
        == _collapse_first_edge(simplex_set).maps
    )
    assert (
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(
                first=_collapse_first_edge(simplex_set), second=identity
            )
        ).maps
        == _collapse_first_edge(simplex_set).maps
    )


def test_composition_rechecks_serialized_naturality_claims():
    simplex_set = standard_simplex(1, 2)
    valid = _identity(simplex_set)
    invalid = TruncatedSimplicialMap(
        source=simplex_set,
        target=simplex_set,
        maps=((0, 1), (1, 1, 2), (0, 1, 2, 3)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(first=invalid, second=valid)
        )
    assert error.value.errors()[0]["type"] == "simplicial_map.face_naturality_failed"


def test_composition_requires_identical_middle_carrier():
    source = standard_simplex(1, 1)
    distinct_middle = standard_simplex(1, 1).model_copy(
        update={"sets": (("left", "right"), source.sets[1])}
    )
    first = TruncatedSimplicialMap(
        source=source,
        target=distinct_middle,
        maps=((0, 1), (0, 1, 2)),
    )
    second = TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=((0, 1), (0, 1, 2)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compose_simplicial_maps(
            SimplicialMapCompositionRequest(first=first, second=second)
        )
    assert error.value.errors()[0]["type"] == (
        "simplicial_map.composition_carrier_mismatch"
    )
