from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.coproduct import simplicial_set_coproduct
from jacobian.math.topology.simplicial_sets.coproduct_models import (
    SimplicialSetCoproductRequest,
)
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    SimplicialMapRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def test_coproduct_matches_disjoint_union_and_componentwise_map_oracle() -> None:
    left = standard_simplex(1, 2)
    right = standard_simplex(0, 2)
    result = simplicial_set_coproduct(
        SimplicialSetCoproductRequest(left=left, right=right)
    )
    union = result.simplicial_set

    for degree, axis in enumerate(result.tagged_axes):
        expected_axis = tuple(
            [("left", i) for i in range(len(left.sets[degree]))]
            + [("right", i) for i in range(len(right.sets[degree]))]
        )
        assert axis == expected_axis
        assert union.sets[degree] == tuple(
            [f"L{i}" for i in range(len(left.sets[degree]))]
            + [f"R{i}" for i in range(len(right.sets[degree]))]
        )
        assert len(set(union.sets[degree])) == len(axis)
        assert len(axis) == len(left.sets[degree]) + len(right.sets[degree])

    # Independent set-theoretic oracle. The component tag stays fixed while
    # the corresponding factor's exact map is applied.
    for degree in range(1, union.max_degree + 1):
        offset = len(left.sets[degree - 1])
        for map_index, actual in enumerate(union.face_maps[degree - 1]):
            expected = tuple(left.face_maps[degree - 1][map_index]) + tuple(
                offset + image for image in right.face_maps[degree - 1][map_index]
            )
            assert actual == expected
    for degree in range(union.max_degree):
        offset = len(left.sets[degree + 1])
        for map_index, actual in enumerate(union.degeneracy_maps[degree]):
            expected = tuple(left.degeneracy_maps[degree][map_index]) + tuple(
                offset + image for image in right.degeneracy_maps[degree][map_index]
            )
            assert actual == expected


def test_coproduct_inclusions_are_natural_and_compose() -> None:
    left = standard_simplex(1, 2)
    right = standard_simplex(0, 2)
    result = simplicial_set_coproduct(
        SimplicialSetCoproductRequest(left=left, right=right)
    )
    assert simplicial_map(
        SimplicialMapRequest(
            source=left,
            target=result.simplicial_set,
            maps=result.left_inclusion.maps,
        )
    ).identities_preserved
    assert simplicial_map(
        SimplicialMapRequest(
            source=right,
            target=result.simplicial_set,
            maps=result.right_inclusion.maps,
        )
    ).identities_preserved

    identity = TruncatedSimplicialMap(
        source=result.simplicial_set,
        target=result.simplicial_set,
        maps=tuple(tuple(range(len(level))) for level in result.simplicial_set.sets),
    )
    composite = compose_simplicial_maps(
        SimplicialMapCompositionRequest(
            first=result.left_inclusion,
            second=identity,
        )
    )
    assert composite.maps == result.left_inclusion.maps


def test_coproduct_rejects_level_overflow_before_expansion() -> None:
    def discrete(prefix: str, size: int):
        labels = tuple(f"{prefix}{i}" for i in range(size))
        identity = tuple(range(size))
        result = from_tables(
            1,
            (labels, labels),
            ((identity, identity),),
            ((identity,),),
        )
        assert result.simplicial_set is not None
        return result.simplicial_set

    left, right = discrete("a", 17), discrete("b", 16)
    with pytest.raises(OperationResourceAdmissionError):
        simplicial_set_coproduct(SimplicialSetCoproductRequest(left=left, right=right))


def test_coproduct_rechecks_caller_authored_factor_identities() -> None:
    valid = standard_simplex(1, 1)
    malformed = FiniteTruncatedSimplicialSet._from_kernel(
        **{
            **valid.model_dump(),
            "degeneracy_maps": (((2, 0),),),
        }
    )
    with pytest.raises(OperationDomainValidationError) as failure:
        simplicial_set_coproduct(
            SimplicialSetCoproductRequest(left=malformed, right=valid)
        )
    assert (
        failure.value.errors()[0]["type"] == "simplicial_set.coproduct_factor_invalid"
    )
