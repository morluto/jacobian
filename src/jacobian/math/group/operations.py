"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

__all__ = ["element_order", "group_orbit", "group_order"]


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
