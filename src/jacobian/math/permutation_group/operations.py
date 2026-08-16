"""Permutation group operations backed by SymPy."""

from __future__ import annotations

__all__ = ["pg_orbit", "pg_order"]


def pg_order(degree, generators):
    from sympy.combinatorics import Permutation, PermutationGroup

    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    return int(PermutationGroup(perms).order())


def pg_orbit(degree, generators, point):
    from sympy.combinatorics import Permutation, PermutationGroup

    if not 0 <= point < degree:
        raise ValueError("point must be in 0..n-1")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    return sorted(PermutationGroup(perms).orbit(point))
