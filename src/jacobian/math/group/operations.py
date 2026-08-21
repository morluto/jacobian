"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

__all__ = [
    "conjugacy_classes",
    "element_order",
    "group_orbit",
    "group_order",
    "subgroup_lattice",
]


def group_order(degree: int, generators: list[list[int]]) -> int:
    """Return the exact order of a permutation group via Schreier-Sims."""
    from sympy.combinatorics import Permutation, PermutationGroup

    if not 1 <= degree <= 64:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
    return int(group.order())


def element_order(degree: int, generator: list[int]) -> int:
    """Return the exact order of one permutation."""
    from sympy.combinatorics import Permutation

    if len(generator) != degree or sorted(generator) != list(range(degree)):
        raise ValueError("generator must be a permutation of 0..n-1")
    return int(Permutation(list(generator)).order())


def group_orbit(degree: int, generators: list[list[int]], point: int) -> list[int]:
    """Return the orbit of a point under a permutation group."""
    from sympy.combinatorics import Permutation, PermutationGroup

    if not 0 <= point < degree:
        raise ValueError("point must be in 0..n-1")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
    orbit = group.orbit(point)
    return sorted(orbit)


def conjugacy_classes(
    degree: int, generators: list[list[int]]
) -> list[tuple[list[list[int]], int]]:
    """Return conjugacy classes of a permutation group.

    Returns a list of (class_elements, class_size) tuples where each
    class_element is a list of permutation lists.
    """
    from sympy.combinatorics import Permutation, PermutationGroup

    if not 1 <= degree <= 64:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
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
    Bounded to groups of order at most 512 to avoid exponential blowup.
    """
    from itertools import combinations

    from sympy.combinatorics import Permutation, PermutationGroup

    if not 1 <= degree <= 64:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
    order = int(group.order())
    if order > 512:
        raise ValueError(
            "subgroup lattice computation is bounded to groups of order at most 512"
        )
    elements = list(group.elements)
    subgroup_keys: set[frozenset] = set()
    result: list[tuple[list[list[int]], int]] = []
    for size in range(1, len(elements) + 1):
        for subset in combinations(elements, size):
            sg = group.subgroup(list(subset))
            key = frozenset(tuple(p.array_form) for p in sg.elements)
            if key not in subgroup_keys:
                subgroup_keys.add(key)
                result.append(
                    ([list(p.array_form) for p in sg.generators], int(sg.order()))
                )
    return result
