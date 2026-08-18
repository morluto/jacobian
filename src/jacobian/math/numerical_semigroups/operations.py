"""Native numerical-semigroup operations on ordinary Python values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import gcd

from jacobian.math.numerical_semigroups._algorithms import (
    apery_set,
    belongs,
    catenary_degree_from_factorizations,
    factorization_count,
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)


@dataclass(frozen=True)
class FactorizationGraph:
    """The edges and connected components of a factorization graph."""

    edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]


def _generators(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("generators must be positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("generators must be strictly increasing")
    if gcd(*values) != 1:
        raise ValueError("generators must have gcd 1")
    if minimal_generating_system(values) != values:
        raise ValueError("generators must be a minimal generating system")
    return values


def factorization_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    """Return the standard distance between equal-degree factorizations."""
    if len(first) != len(second) or not first:
        raise ValueError("factorizations must have the same positive dimension")
    if any(type(value) is not int or value < 0 for value in (*first, *second)):
        raise ValueError("factorization coordinates must be nonnegative integers")
    common_length = sum(
        min(left, right) for left, right in zip(first, second, strict=True)
    )
    return max(sum(first) - common_length, sum(second) - common_length)


def factorization_graph(family: tuple[tuple[int, ...], ...]) -> FactorizationGraph:
    """Build the graph joining factorizations that share a generator."""
    if not family:
        return FactorizationGraph(edges=(), components=())
    dimension = len(family[0])
    if not dimension or any(len(item) != dimension for item in family):
        raise ValueError("factorizations must have one common positive dimension")
    if any(type(value) is not int or value < 0 for item in family for value in item):
        raise ValueError("factorization coordinates must be nonnegative integers")
    edges = tuple(
        (left, right)
        for left in range(len(family))
        for right in range(left + 1, len(family))
        if any(min(a, b) > 0 for a, b in zip(family[left], family[right], strict=True))
    )
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(family))}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(family)):
        if start in seen:
            continue
        pending = [start]
        seen.add(start)
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return FactorizationGraph(edges=edges, components=tuple(components))


def element_delta_set(generators: tuple[int, ...], value: int) -> tuple[int, ...]:
    """Return the successive factorization-length gaps of one element."""
    lengths = factorization_lengths(_generators(generators), value)
    return tuple(sorted({right - left for left, right in pairwise(lengths)}))


def element_elasticity(generators: tuple[int, ...], value: int) -> Fraction:
    """Return the exact elasticity of one nonzero semigroup element."""
    lengths = factorization_lengths(_generators(generators), value)
    if not lengths:
        raise ValueError("value is not in the numerical semigroup")
    if lengths[0] == 0:
        raise ValueError("elasticity is undefined for zero")
    return Fraction(lengths[-1], lengths[0])


def element_catenary_degree(generators: tuple[int, ...], value: int) -> int:
    """Return the exact catenary degree of one semigroup element."""
    family = factorizations(_generators(generators), value)
    if not family:
        raise ValueError("value is not in the numerical semigroup")
    return catenary_degree_from_factorizations(family)


def elasticity(generators: tuple[int, ...]) -> Fraction:
    """Return the exact global elasticity of a numerical semigroup."""
    values = _generators(generators)
    return Fraction(values[-1], values[0])


__all__ = [
    "FactorizationGraph",
    "apery_set",
    "belongs",
    "elasticity",
    "element_catenary_degree",
    "element_delta_set",
    "element_elasticity",
    "factorization_count",
    "factorization_distance",
    "factorization_graph",
    "factorization_lengths",
    "factorizations",
    "minimal_generating_system",
]
