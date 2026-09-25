from __future__ import annotations

from itertools import combinations_with_replacement

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets import from_tables
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.skeleton import (
    SimplicialSetSkeletonRequest,
    simplicial_set_skeleton,
)
from jacobian.math.topology.simplicial_sets.skeleton_tools import TOOLS
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def test_delta_two_one_skeleton_matches_independent_monotone_map_oracle() -> None:
    source = standard_simplex(2, 2)
    result = simplicial_set_skeleton(
        SimplicialSetSkeletonRequest(simplicial_set=source, k=1)
    )

    # In Delta[2], a k-skeleton consists exactly of monotone maps whose image
    # has at most k+1 vertices. Enumerate those maps directly, independently of
    # the operation's degeneracy-closure algorithm.
    for degree in range(3):
        expected = tuple(
            "(" + ",".join(map(str, simplex)) + ")"
            for simplex in combinations_with_replacement(range(3), degree + 1)
            if len(set(simplex)) <= 2
        )
        assert result.skeleton.sets[degree] == expected
        assert result.inclusion.maps[degree] == tuple(
            source.sets[degree].index(label) for label in expected
        )

    assert len(result.skeleton.sets[2]) == 9
    assert "(0,1,2)" not in result.skeleton.sets[2]
    assert result.inclusion.source == result.skeleton
    assert result.inclusion.target == source

    for degree in range(1, 3):
        for face_index, face in enumerate(result.skeleton.face_maps[degree - 1]):
            for simplex_index, face_image in enumerate(face):
                assert (
                    source.face_maps[degree - 1][face_index][
                        result.inclusion.maps[degree][simplex_index]
                    ]
                    == result.inclusion.maps[degree - 1][face_image]
                )
    for degree in range(2):
        for degeneracy_index, degeneracy in enumerate(
            result.skeleton.degeneracy_maps[degree]
        ):
            for simplex_index, degeneracy_image in enumerate(degeneracy):
                assert (
                    source.degeneracy_maps[degree][degeneracy_index][
                        result.inclusion.maps[degree][simplex_index]
                    ]
                    == result.inclusion.maps[degree + 1][degeneracy_image]
                )


def test_zero_skeleton_contains_only_repeated_vertices_in_higher_degrees() -> None:
    source = standard_simplex(2, 3)
    result = simplicial_set_skeleton(
        SimplicialSetSkeletonRequest(simplicial_set=source, k=0)
    )
    for degree, level in enumerate(result.skeleton.sets):
        expected = tuple(
            "(" + ",".join(map(str, simplex)) + ")"
            for simplex in combinations_with_replacement(range(3), degree + 1)
            if len(set(simplex)) == 1
        )
        assert level == expected


def test_maximal_skeleton_is_the_identity_subset_and_round_trips() -> None:
    source = standard_simplex(1, 2)
    result = simplicial_set_skeleton(
        SimplicialSetSkeletonRequest(simplicial_set=source, k=2)
    )
    assert result.skeleton == source
    assert result.inclusion.maps == tuple(
        tuple(range(len(level))) for level in source.sets
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_empty_simplicial_set_has_empty_skeleton_and_inclusion() -> None:
    source = from_tables(
        2,
        ((), (), ()),
        (((), ()), ((), (), ())),
        (((),), ((), ())),
    ).simplicial_set
    assert source is not None
    result = simplicial_set_skeleton(
        SimplicialSetSkeletonRequest(simplicial_set=source, k=1)
    )
    assert result.skeleton.sets == ((), (), ())
    assert result.inclusion.maps == ((), (), ())


def test_skeleton_rejects_unseen_degree_and_invalid_source_tables() -> None:
    source = standard_simplex(1, 1)
    with pytest.raises(OperationDomainValidationError, match="visible"):
        simplicial_set_skeleton(
            SimplicialSetSkeletonRequest.model_construct(simplicial_set=source, k=2)
        )

    corrupted = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=source.sets,
        face_maps=((source.face_maps[0][0], (1, 1, 1)),),
        degeneracy_maps=source.degeneracy_maps,
        total_simplices=source.total_simplices,
        checked_identities=source.checked_identities,
    )
    with pytest.raises(OperationDomainValidationError, match="visible simplicial"):
        simplicial_set_skeleton(
            SimplicialSetSkeletonRequest.model_construct(simplicial_set=corrupted, k=0)
        )


def test_request_rejects_boolean_degree() -> None:
    with pytest.raises(ValidationError):
        SimplicialSetSkeletonRequest(simplicial_set=standard_simplex(0, 0), k=True)


def test_skeleton_is_published_in_the_immutable_operation_manifest() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "topology.simplicial_set.skeleton.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.k == 1
    assert "(0,1,2)" not in result.skeleton.sets[2]
