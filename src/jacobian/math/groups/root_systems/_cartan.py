"""Exact structural algorithms for bounded finite Cartan data."""

from __future__ import annotations

from collections import deque
from fractions import Fraction

from jacobian.math._exact_linear_algebra import symmetric_inertia

MAX_POSITIVE_ROOTS = 120  # E8 is maximal among crystallographic rank <= 8.
MAX_ROOT_COORDINATE = 6  # Maximal coefficient of an E8 positive root.


def connected_components(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(matrix)))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            vertex = stack.pop()
            if vertex in component:
                continue
            component.add(vertex)
            remaining.discard(vertex)
            stack.extend(
                neighbor for neighbor in remaining if matrix[vertex][neighbor] != 0
            )
        components.append(tuple(sorted(component)))
    return tuple(components)


def positive_symmetrizer(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[Fraction, ...]:
    values: list[Fraction | None] = [None] * len(matrix)
    for component in connected_components(matrix):
        values[component[0]] = Fraction(1)
        queue = deque([component[0]])
        while queue:
            left = queue.popleft()
            left_value = values[left]
            assert left_value is not None
            for right in component:
                if matrix[left][right] == 0:
                    continue
                candidate = left_value * Fraction(
                    matrix[left][right], matrix[right][left]
                )
                if values[right] is None:
                    values[right] = candidate
                    queue.append(right)
                elif values[right] != candidate:
                    raise ValueError("Cartan matrix is not symmetrizable")
    return tuple(value for value in values if value is not None)


def require_finite_type(matrix: tuple[tuple[int, ...], ...]) -> None:
    symmetrizer = positive_symmetrizer(matrix)
    symmetric = tuple(
        tuple(symmetrizer[row] * matrix[row][column] for column in range(len(matrix)))
        for row in range(len(matrix))
    )
    denominators = [entry.denominator for row in symmetric for entry in row]
    scale = 1
    from math import lcm

    for denominator in denominators:
        scale = lcm(scale, denominator)
    integral = tuple(tuple(int(entry * scale) for entry in row) for row in symmetric)
    positive, negative, zero = symmetric_inertia(integral)
    if (positive, negative, zero) != (len(matrix), 0, 0):
        raise ValueError("Cartan matrix must be of finite type")


def simple_reflection(
    root: tuple[int, ...], index: int, matrix: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    coefficient = sum(root[j] * matrix[index][j] for j in range(len(matrix)))
    reflected = list(root)
    reflected[index] -= coefficient
    return tuple(reflected)


def positive_roots(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    rank = len(matrix)
    initial = [tuple(int(i == j) for j in range(rank)) for i in range(rank)]
    seen = set(initial)
    queue = deque(initial)
    reflection_applications = 0
    while queue:
        root = queue.popleft()
        for index in range(rank):
            reflection_applications += 1
            if reflection_applications > MAX_POSITIVE_ROOTS * rank:
                raise RuntimeError("finite root closure exceeded its reflection bound")
            reflected = simple_reflection(root, index, matrix)
            if not all(value >= 0 for value in reflected) or not any(reflected):
                continue
            if max(reflected) > MAX_ROOT_COORDINATE:
                raise RuntimeError("finite root closure exceeded its coordinate bound")
            if reflected not in seen:
                seen.add(reflected)
                if len(seen) > MAX_POSITIVE_ROOTS:
                    raise RuntimeError(
                        "finite root closure exceeded its root-count bound"
                    )
                queue.append(reflected)
    return tuple(sorted(seen))


__all__ = [
    "connected_components",
    "positive_roots",
    "positive_symmetrizer",
    "require_finite_type",
    "simple_reflection",
]
