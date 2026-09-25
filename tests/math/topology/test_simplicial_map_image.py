from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets._tools import TOOLS
from jacobian.math.topology.simplicial_sets.image import (
    SimplicialMapImageRequest,
    simplicial_map_image,
)
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    identity_simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _map(source, target, rows):
    return TruncatedSimplicialMap(source=source, target=target, maps=rows)


def test_image_factorization_collapses_simplex_and_composes_to_input():
    source = standard_simplex(1, 2)
    target = standard_simplex(1, 2)
    collapse = _map(source, target, ((0, 0), (0, 0, 0), (0, 0, 0, 0)))

    result = simplicial_map_image(SimplicialMapImageRequest(simplicial_map=collapse))

    assert tuple(map(len, result.image.sets)) == (1, 1, 1)
    assert result.surjection.source == source
    assert result.surjection.target == result.image
    assert result.inclusion.source == result.image
    assert result.inclusion.target == target
    for row in result.surjection.maps:
        assert set(row) == {0}
    for degree, row in enumerate(result.inclusion.maps):
        assert len(row) == len(set(row))
        assert row == (0,)
        assert len(row) < len(target.sets[degree])
    composite = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=result.surjection, second=result.inclusion
        )
    )
    assert composite == collapse
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_image_operation_is_published_in_the_owner_manifest():
    assert "topology.simplicial_set.map.image.compute" in {
        tool.operation_id for tool in TOOLS
    }


def test_identity_map_has_full_image_and_identity_factors():
    source = standard_simplex(1, 2)
    identity = identity_simplicial_map(source)

    result = simplicial_map_image(SimplicialMapImageRequest(simplicial_map=identity))

    assert result.image == source
    assert result.surjection == identity
    assert result.inclusion == identity


def test_image_preserves_the_initial_all_empty_prefix():
    source_result = from_tables(1, ((), ()), (((), ()),), (((),),))
    assert source_result.simplicial_set is not None
    empty_map = _map(source_result.simplicial_set, standard_simplex(0, 1), ((), ()))

    result = simplicial_map_image(SimplicialMapImageRequest(simplicial_map=empty_map))

    assert result.image.sets == ((), ())
    assert result.surjection.maps == ((), ())
    assert result.inclusion.maps == ((), ())


def test_image_rechecks_caller_supplied_carrier_identities():
    source = standard_simplex(1, 1)
    bad_target = source.model_copy(
        update={"face_maps": (((1, 1, 1), source.face_maps[0][1]),)}
    )
    identity_rows = tuple(tuple(range(len(level))) for level in source.sets)
    forged = _map(source, bad_target, identity_rows)

    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_map_image(SimplicialMapImageRequest(simplicial_map=forged))

    assert error.value.errors()[0]["type"] == "simplicial_map.image_carrier_invalid"


def test_image_rejects_a_non_natural_map():
    source = standard_simplex(1, 1)
    bad = _map(source, source, ((0, 1), (1, 1, 2)))

    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_map_image(SimplicialMapImageRequest(simplicial_map=bad))

    assert error.value.errors()[0]["type"] == "simplicial_map.face_naturality_failed"
