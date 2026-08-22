"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

__all__ = [
    "element_order",
    "group_conjugacy_classes",
    "group_orbit",
    "group_order",
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


def group_conjugacy_classes(
    degree: int, generators: list[list[int]]
) -> list[list[list[int]]]:
    """Return conjugacy classes as lists of permutation array forms.

    Two elements are conjugate iff they lie in the same class.  The returned
    classes partition the group; each class is a list of permutations (as
    array forms over ``0..n-1``). The result is canonically ordered: members
    of each class are sorted lexicographically and classes are sorted by
    their smallest member, so equal groups serialize identically. The
    generated group must have order at most 5000 (degree up to 64 alone
    does not bound enumeration; e.g., S8 has order 40320); larger groups
    are rejected before enumeration.
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
    # Bound enumeration by group order before materializing all |G| elements.
    # S12 has order 479M and would exhaust memory; reject conservatively.
    from jacobian.math.group._models import MAX_CONJUGACY_CLASSES_GROUP_ORDER

    order = int(group.order())
    if order > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
        raise ValueError(
            f"group order {order} exceeds the bounded maximum "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} for conjugacy classes "
            f"(would materialize |G|={order} elements)"
        )
    classes = group.conjugacy_classes()
    canonical = [sorted(list(p.array_form) for p in cls) for cls in classes]
    canonical.sort(key=lambda cls: tuple(cls[0]))
    return canonical
