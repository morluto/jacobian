"""Supported exact finite group API."""

from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.operations import (
    element_order,
    group_conjugacy_classes,
    group_orbit,
    group_order,
    group_stabilizer,
    subgroup_lattice,
    verify_element_order,
    verify_group_conjugacy_classes,
    verify_group_orbit,
    verify_group_order,
    verify_group_stabilizer,
    verify_subgroup_lattice,
)

__all__ = [
    "PermutationGroup",
    "element_order",
    "group_conjugacy_classes",
    "group_orbit",
    "group_order",
    "group_stabilizer",
    "subgroup_lattice",
    "verify_element_order",
    "verify_group_conjugacy_classes",
    "verify_group_orbit",
    "verify_group_order",
    "verify_group_stabilizer",
    "verify_subgroup_lattice",
]
