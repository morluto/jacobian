"""Diagonal tuple-family orbit profiles."""

from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.tuple_orbits._models import TupleFamilyOrbitRequest
from jacobian.math.groups.tuple_orbits.operations import tuple_family_orbit_profile


def test_repeated_coordinates_and_duplicate_sources_are_retained() -> None:
    request = TupleFamilyOrbitRequest(
        group=PermutationGroup(degree=3, generators=((1, 2, 0),)),
        arity=2,
        family=((2, 2), (0, 0), (0, 1), (1, 2), (0, 1)),
    )
    result = tuple_family_orbit_profile(request)
    assert [row.representative for row in result.rows] == [(0, 0), (0, 1)]
    assert result.rows[0].source_indices == (0, 1)
    assert result.rows[1].source_indices == (2, 3, 4)
    assert all(row.orbit_size == 3 and row.stabilizer_size == 1 for row in result.rows)


def test_empty_family_has_empty_profile() -> None:
    request = TupleFamilyOrbitRequest(
        group=PermutationGroup(degree=1, generators=((0,),)), arity=0, family=()
    )
    assert tuple_family_orbit_profile(request).rows == ()
