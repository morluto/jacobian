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
    for gen in stab.stabilizer.generators:
        assert gen[0] == 0

    # Stabilizer as a canonical group value is directly consumable.
    assert stab.source == PermutationGroupRequest(degree=degree, generators=gens)
    assert stab.stabilizer.degree == degree
    stab_order = int(compute_group_order(stab.stabilizer).order)
    assert order == len(orbit) * stab_order


def test_group_stabilizer_of_identity_is_full_group_generators_fixing_point() -> None:
    from jacobian.math.group._models import GroupStabilizerRequest
    from jacobian.math.group._operations import compute_group_stabilizer

    # A generator that already fixes point 0 is its own stabilizer generator.
    stab = compute_group_stabilizer(
        GroupStabilizerRequest(degree=3, generators=((0, 2, 1),), point=0),
    )
    for gen in stab.stabilizer.generators:
        assert gen[0] == 0


def test_group_stabilizer_trivial_is_consumer_compatible() -> None:
    """Trivial stabilizer must be consumable without reshaping."""

    from jacobian.math.group._models import GroupStabilizerRequest
    from jacobian.math.group._operations import (
        compute_group_order,
        compute_group_stabilizer,
    )

    # Regular cyclic group C4: stabilizer of any point is trivial (order 1).
    # The result must carry the identity so the consumer accepts it directly.
    stab = compute_group_stabilizer(
        GroupStabilizerRequest(degree=4, generators=((1, 2, 3, 0),), point=0),
    )
    # Stabilizer is the identity group, still degree 4, consumable.
    assert stab.stabilizer.generators == ((0, 1, 2, 3),)
    assert int(compute_group_order(stab.stabilizer).order) == 1
    # No reshaping or forbidden extras: the nested value validates as is.
    from jacobian.math.group._models import PermutationGroupRequest

    PermutationGroupRequest.model_validate(stab.stabilizer.model_dump())


def test_group_stabilizer_rejects_invalid_point() -> None:
    from jacobian.math.group._models import GroupStabilizerRequest

    with pytest.raises(ValidationError, match="point"):
        GroupStabilizerRequest(degree=2, generators=((1, 0),), point=5)
