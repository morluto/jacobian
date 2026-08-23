"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.combinatorics import PermutationGroup

__all__ = [
    "conjugacy_classes",
    "element_order",
    "group_orbit",
    "group_order",
    "subgroup_lattice",
]

_MAX_DEGREE = 64


def _bounded_permutation_group(
    degree: int, generators: list[list[int]]
) -> PermutationGroup:
    """Validate bounded native inputs and build the permutation group."""
    from sympy.combinatorics import Permutation, PermutationGroup

    from jacobian.math.group._models import MAX_GROUP_DEGREE

    if not 1 <= degree <= _MAX_DEGREE:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    if len(generators) > MAX_GROUP_DEGREE:
        raise ValueError(f"at most {MAX_GROUP_DEGREE} generators are supported")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    return PermutationGroup(perms)


def group_order(degree: int, generators: list[list[int]]) -> int:
    """Return the exact order of a permutation group via Schreier-Sims."""
    group = _bounded_permutation_group(degree, generators)
    return int(group.order())


def element_order(degree: int, generator: list[int]) -> int:
    """Return the exact order of one permutation."""
    from sympy.combinatorics import Permutation

    if len(generator) != degree or sorted(generator) != list(range(degree)):
        raise ValueError("generator must be a permutation of 0..n-1")
    return int(Permutation(list(generator)).order())


def group_orbit(degree: int, generators: list[list[int]], point: int) -> list[int]:
    """Return the orbit of a point under a permutation group."""
    if not 0 <= point < degree:
        raise ValueError("point must be in 0..n-1")
    group = _bounded_permutation_group(degree, generators)
    return sorted(group.orbit(point))


def conjugacy_classes(
    degree: int, generators: list[list[int]]
) -> list[tuple[list[list[int]], int]]:
    """Return conjugacy classes of a permutation group.

    Returns a list of (class_elements, class_size) tuples where each
    class_element is a list of permutation lists.
    """
    from jacobian.math.group._models import MAX_GROUP_ORDER

    group = _bounded_permutation_group(degree, generators)
    order = int(group.order())
    if order > MAX_GROUP_ORDER:
        raise ValueError(
            f"group order {order} exceeds the bounded maximum "
            f"{MAX_GROUP_ORDER} for full-element enumeration"
        )
    classes = group.conjugacy_classes()
    result = []
    for cls in classes:
        elements = [list(p.array_form) for p in cls]
        result.append((elements, len(cls)))
    return result


def subgroup_lattice(
    degree: int, generators: list[list[int]]
) -> list[tuple[list[list[int]], int]]:
    """Return all subgroups of a permutation group.

    Returns a list of (generators_of_subgroup, subgroup_order) tuples.
    Bounded to groups of order at most 64 to avoid exponential blowup.
    """
    from sympy.combinatorics import Permutation

    group = _bounded_permutation_group(degree, generators)
    if int(group.order()) > 64:
        raise ValueError(
            "subgroup lattice computation is bounded to groups of order at most 64"
        )
    # Traverse distinct subgroups instead of the 2^|G|-element power set:
    # every subgroup is the closure of an existing subgroup extended by one
    # element, so a fixpoint frontier enumerates each exactly once with
    # |subgroups| x |G| closure constructions - feasible under the admitted
    # order bound.
    identity = Permutation(list(range(degree)))
    trivial_key = frozenset((tuple(identity.array_form),))
    subgroup_keys: set[frozenset[tuple[int, ...]]] = {trivial_key}
    result: list[tuple[list[list[int]], int]] = [([list(identity.array_form)], 1)]
    frontier = [trivial_key]
    elements = list(group.elements)
    while frontier:
        current = frontier.pop()
        [Permutation(list(member)) for member in current]
        for element in elements:
            element_form = tuple(element.array_form)
            if element_form in current:
                continue
            generated = group.subgroup(
                [Permutation(list(member)) for member in current] + [element]
            )
            key = frozenset(tuple(p.array_form) for p in generated.elements)
            if key not in subgroup_keys:
                subgroup_keys.add(key)
                result.append(
                    (
                        sorted(
                            (list(p.array_form) for p in generated.generators),
                        ),
                        int(generated.order()),
                    )
                )
                frontier.append(key)
    return result
