from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.group._models import GroupOrbitRequest


def test_group_orbit_contract_binds_the_point_to_the_declared_degree() -> None:
    with pytest.raises(ValidationError, match="point"):
        GroupOrbitRequest(degree=2, generators=((1, 0),), point=3)


def test_group_stabilizer_orbit_stabilizer_theorem() -> None:
    """|G| = |orbit(point)| * |stabilizer(point)| for S3."""
    from jacobian.math.group._models import (
        GroupOrbitRequest,
        GroupStabilizerRequest,
        PermutationGroupRequest,
    )
    from jacobian.math.group._operations import (
        compute_group_orbit,
        compute_group_order,
        compute_group_stabilizer,
    )

    gens = ((1, 2, 0), (1, 0, 2))
    degree = 3
    order = int(
        compute_group_order(
            PermutationGroupRequest(degree=degree, generators=gens)
        ).order
    )
    orbit = compute_group_orbit(
        GroupOrbitRequest(degree=degree, generators=gens, point=0)
    ).orbit
    stab = compute_group_stabilizer(
        GroupStabilizerRequest(degree=degree, generators=gens, point=0)
    )

    # Stabilizer generators must fix the point.
    for gen in stab.generators:
        assert gen[0] == 0

    # Stabilizer subgroup order.
    stab_gens = stab.generators or ((0, 1, 2),)
    stab_order = int(
        compute_group_order(
            PermutationGroupRequest(degree=degree, generators=tuple(stab_gens)),
        ).order
    )
    assert order == len(orbit) * stab_order


def test_group_stabilizer_of_identity_is_full_group_generators_fixing_point() -> None:
    from jacobian.math.group._models import GroupStabilizerRequest
    from jacobian.math.group._operations import compute_group_stabilizer

    # A generator that already fixes point 0 is its own stabilizer generator.
    stab = compute_group_stabilizer(
        GroupStabilizerRequest(degree=3, generators=((0, 2, 1),), point=0),
    )
    for gen in stab.generators:
        assert gen[0] == 0


def test_group_stabilizer_rejects_invalid_point() -> None:
    from jacobian.math.group._models import GroupStabilizerRequest

    with pytest.raises(ValidationError, match="point"):
        GroupStabilizerRequest(degree=2, generators=((1, 0),), point=5)
