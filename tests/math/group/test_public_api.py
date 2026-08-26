"""Public native API contract for finite groups."""

from __future__ import annotations

from jacobian.math import group


def test_native_group_api_exports_the_canonical_group_value() -> None:
    assert tuple(group.__all__) == (
        "PermutationGroup",
        "element_order",
        "group_conjugacy_classes",
        "group_orbit",
        "group_order",
        "group_stabilizer",
        "subgroup_lattice",
    )
    assert all(not name.endswith(("Request", "Input")) for name in group.__all__)

    cyclic_four = group.PermutationGroup(degree=4, generators=((1, 2, 3, 0),))
    assert group.group_order(cyclic_four) == 4
