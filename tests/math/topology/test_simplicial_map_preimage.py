"""Pullbacks of finite simplicial subobjects along exact simplicial maps."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets import (
    FiniteTruncatedSimplicialSet,
    SimplicialMapPreimageRequest,
    SimplicialMapPreimageResult,
    SimplicialSubsetPrefix,
    SimplicialSubsetRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    simplicial_map_preimage,
    simplicial_subset,
    standard_simplex,
)
from jacobian.math.topology.simplicial_sets._tools import TOOLS
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
)


def _subset(
    source: FiniteTruncatedSimplicialSet,
    degree_indices: tuple[tuple[int, ...], ...],
) -> SimplicialSubsetPrefix:
    return simplicial_subset(
        SimplicialSubsetRequest(
            simplicial_set=source,
            degree_indices=degree_indices,
        )
    )


def _identity(source: FiniteTruncatedSimplicialSet) -> TruncatedSimplicialMap:
    return TruncatedSimplicialMap(
        source=source,
        target=source,
        maps=tuple(tuple(range(len(level))) for level in source.sets),
    )


def test_identity_preimage_is_the_same_subobject_and_commuting_square() -> None:
    delta_one = standard_simplex(1, 2)
    subset = _subset(delta_one, ((0,), (0,), (0,)))
    identity = _identity(delta_one)

    result = simplicial_map_preimage(
        SimplicialMapPreimageRequest(
            simplicial_map=identity,
            target_subset=subset,
        )
    )

    assert result.preimage.inclusion.maps == subset.inclusion.maps
    assert result.preimage.inclusion.source == subset.inclusion.source
    assert result.restricted_map.maps == tuple(
        tuple(range(len(level))) for level in result.preimage.inclusion.source.sets
    )
    left = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=result.restricted_map,
            second=result.target_subset.inclusion,
        )
    )
    right = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=result.preimage.inclusion,
            second=result.simplicial_map,
        )
    )
    assert left.maps == right.maps
    assert (
        SimplicialMapPreimageResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_disjoint_vertex_preimage_is_the_empty_prefix() -> None:
    delta_one = standard_simplex(1, 1)
    target_vertex_zero = _subset(delta_one, ((0,), (0,)))
    constant_at_vertex_one = TruncatedSimplicialMap(
        source=delta_one,
        target=delta_one,
        maps=((1, 1), (2, 2, 2)),
    )

    result = simplicial_map_preimage(
        SimplicialMapPreimageRequest(
            simplicial_map=constant_at_vertex_one,
            target_subset=target_vertex_zero,
        )
    )

    assert result.preimage.inclusion.source.sets == ((), ())
    assert result.preimage.inclusion.maps == ((), ())
    assert result.restricted_map.maps == ((), ())


def test_operation_is_published_in_the_owner_manifest() -> None:
    assert "topology.simplicial_set.map.preimage.compute" in {
        tool.operation_id for tool in TOOLS
    }


def test_nonnatural_map_is_rejected() -> None:
    simplex = standard_simplex(1, 1)
    subset = _subset(simplex, ((0,), (0,)))
    nonnatural = TruncatedSimplicialMap(
        source=simplex,
        target=simplex,
        maps=((0, 1), (2, 1, 0)),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_map_preimage(
            SimplicialMapPreimageRequest(
                simplicial_map=nonnatural,
                target_subset=subset,
            )
        )
    assert error.value.errors()[0]["type"] == "simplicial_map.face_naturality_failed"
