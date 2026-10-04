"""Finite degreewise simplicial subobjects and their inclusions."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets import (
    SimplicialSubsetPrefix,
    simplicial_subset,
    standard_simplex,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    compose_simplicial_maps,
    identity_simplicial_map,
)


def _delta_one_prefix() -> FiniteTruncatedSimplicialSet:
    return standard_simplex(1, 2)


def test_constant_vertex_gives_source_bound_closed_subobject() -> None:
    ambient = _delta_one_prefix()
    result = simplicial_subset(ambient, ((0,), (0,), (0,)))

    assert result.inclusion.target == ambient
    assert result.inclusion.source.sets == (("(0)",), ("(0,0)",), ("(0,0,0)",))
    assert result.inclusion.maps == ((0,), (0,), (0,))
    assert result.inclusion.source.face_maps == (((0,), (0,)), ((0,), (0,), (0,)))
    assert result.inclusion.source.degeneracy_maps == (((0,),), ((0,), (0,)))

    subset = result.inclusion.source
    for degree in range(1, ambient.max_degree + 1):
        for operator, face_row in enumerate(ambient.face_maps[degree - 1]):
            subset_row = subset.face_maps[degree - 1][operator]
            for simplex, image in enumerate(result.inclusion.maps[degree]):
                assert (
                    result.inclusion.maps[degree - 1][subset_row[simplex]]
                    == (face_row[image])
                )
    for degree in range(ambient.max_degree):
        for operator, degeneracy_row in enumerate(ambient.degeneracy_maps[degree]):
            subset_row = subset.degeneracy_maps[degree][operator]
            for simplex, image in enumerate(result.inclusion.maps[degree]):
                assert (
                    result.inclusion.maps[degree + 1][subset_row[simplex]]
                    == (degeneracy_row[image])
                )


def test_empty_degree_families_form_the_initial_subobject() -> None:
    result = simplicial_subset(_delta_one_prefix(), ((), (), ()))
    assert result.inclusion.source.sets == ((), (), ())
    assert result.inclusion.source.face_maps == (((), ()), ((), (), ()))
    assert result.inclusion.source.degeneracy_maps == (((),), ((), ()))
    assert result.inclusion.maps == ((), (), ())


def test_boundary_of_the_one_simplex_retains_both_vertices_and_degeneracies() -> None:
    ambient = _delta_one_prefix()
    result = simplicial_subset(ambient, ((0, 1), (0, 2), (0, 3)))
    assert result.inclusion.maps == ((0, 1), (0, 2), (0, 3))
    assert result.inclusion.source.sets == (
        ("(0)", "(1)"),
        ("(0,0)", "(1,1)"),
        ("(0,0,0)", "(1,1,1)"),
    )


def test_missing_face_or_degeneracy_rejects_the_proposed_family() -> None:
    ambient = _delta_one_prefix()
    with pytest.raises(OperationDomainValidationError) as exc_info:
        simplicial_subset(ambient, ((), (1,), ()))
    assert exc_info.value.errors()[0]["type"] == "simplicial_set.subset.not_face_closed"
    with pytest.raises(OperationDomainValidationError) as exc_info:
        simplicial_subset(ambient, ((0,), (), ()))
    assert (
        exc_info.value.errors()[0]["type"]
        == "simplicial_set.subset.not_degeneracy_closed"
    )


def test_serialized_subobject_inclusion_composes_as_a_simplicial_map() -> None:
    ambient = _delta_one_prefix()
    result = simplicial_subset(ambient, ((0,), (0,), (0,)))
    restored = SimplicialSubsetPrefix.model_validate_json(result.model_dump_json())
    identity = identity_simplicial_map(ambient)
    composite = compose_simplicial_maps(
        SimplicialMapCompositionRequest(first=restored.inclusion, second=identity)
    )
    assert composite.source == restored.inclusion.source
    assert composite.target == ambient
    assert composite.maps == restored.inclusion.maps
