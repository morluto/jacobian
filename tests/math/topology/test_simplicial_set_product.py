from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.operations import (
    differential_squares_to_zero,
)
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    SimplicialMapRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    normalized_chains,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.product import simplicial_set_product
from jacobian.math.topology.simplicial_sets.product_models import (
    SimplicialSetProductRequest,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def test_delta_one_product_has_exact_componentwise_tables_and_factor_axes() -> None:
    left = standard_simplex(1, 2)
    right = standard_simplex(1, 2)
    result = simplicial_set_product(SimplicialSetProductRequest(left=left, right=right))
    product = result.simplicial_set

    for degree, axis in enumerate(result.pair_axes):
        assert axis == tuple(
            (i, j)
            for i in range(len(left.sets[degree]))
            for j in range(len(right.sets[degree]))
        )
        assert result.left_projection.maps[degree] == tuple(i for i, _ in axis)
        assert result.right_projection.maps[degree] == tuple(j for _, j in axis)

    # Independent oracle: each output map applies the same factor map to both
    # coordinates, then indexes the target pair using the right factor width.
    for degree in range(1, product.max_degree + 1):
        for face_index, actual in enumerate(product.face_maps[degree - 1]):
            expected = tuple(
                left.face_maps[degree - 1][face_index][i] * len(right.sets[degree - 1])
                + right.face_maps[degree - 1][face_index][j]
                for i, j in result.pair_axes[degree]
            )
            assert actual == expected
    for degree in range(product.max_degree):
        for degeneracy_index, actual in enumerate(product.degeneracy_maps[degree]):
            expected = tuple(
                left.degeneracy_maps[degree][degeneracy_index][i]
                * len(right.sets[degree + 1])
                + right.degeneracy_maps[degree][degeneracy_index][j]
                for i, j in result.pair_axes[degree]
            )
            assert actual == expected

    assert product.checked_identities > 0
    normalized = normalized_chains(product)
    assert differential_squares_to_zero(normalized.chain_complex).is_valid


def test_product_projections_are_simplicial_maps() -> None:
    left, right = standard_simplex(1, 2), standard_simplex(0, 2)
    result = simplicial_set_product(SimplicialSetProductRequest(left=left, right=right))
    assert simplicial_map(
        SimplicialMapRequest(
            source=result.simplicial_set,
            target=left,
            maps=result.left_projection.maps,
        )
    ).identities_preserved
    identity_left = TruncatedSimplicialMap(
        source=left,
        target=left,
        maps=tuple(tuple(range(len(level))) for level in left.sets),
    )
    composite = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=result.left_projection,
            second=identity_left,
        )
    )
    assert composite.maps == result.left_projection.maps
    assert simplicial_map(
        SimplicialMapRequest(
            source=result.simplicial_set,
            target=right,
            maps=result.right_projection.maps,
        )
    ).identities_preserved


def test_product_canonicalizes_factors_with_forged_identity_summary() -> None:
    factor = standard_simplex(1, 1)
    forged = factor.model_copy(update={"checked_identities": factor.checked_identities + 1})
    result = simplicial_set_product(SimplicialSetProductRequest(left=forged, right=factor))
    assert result.left == factor
    assert result.left_projection.target == factor


def test_product_rejects_degree_size_overflow_before_expansion() -> None:
    labels = tuple(f"x{i}" for i in range(6))
    identity = tuple(range(6))
    discrete = from_tables(
        1,
        (labels, labels),
        ((identity, identity),),
        ((identity,),),
    ).simplicial_set
    assert discrete is not None
    with pytest.raises(OperationResourceAdmissionError):
        simplicial_set_product(
            SimplicialSetProductRequest(left=discrete, right=discrete)
        )
