"""Shared exact geometry primitives for rational polytope owners.

The helpers establish geometric facts only.  Callers own every request's
combination, intermediate-height, scan, and result-envelope admission.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from sympy import Matrix, Rational
from sympy.matrices.exceptions import NonInvertibleMatrixError

RationalRow = tuple[Sequence[Rational], Rational]


def hyperplane_normal(points: Sequence[Sequence[Rational]]) -> Matrix | None:
    """Return the unique normal through ``dim`` affine points, if one exists."""

    dimension = len(points)
    if dimension == 1:
        return Matrix([Rational(1)])
    differences = Matrix(
        [
            [points[index][axis] - points[0][axis] for axis in range(dimension)]
            for index in range(1, dimension)
        ]
    )
    basis = differences.nullspace()
    return basis[0] if len(basis) == 1 else None


def facets_from_points(
    vertices: Sequence[Sequence[Rational]], dimension: int
) -> list[tuple[Matrix, Rational]]:
    """Enumerate distinct oriented supporting rows of ``conv(vertices)``."""

    candidates: list[tuple[Matrix, Rational]] = []
    for indices in combinations(range(len(vertices)), dimension):
        points = [vertices[index] for index in indices]
        normal = hyperplane_normal(points)
        if normal is None:
            continue
        offset = sum(normal[axis] * points[0][axis] for axis in range(dimension))
        residuals = [
            sum(normal[axis] * vertex[axis] for axis in range(dimension)) - offset
            for vertex in vertices
        ]
        if all(value <= 0 for value in residuals):
            candidates.append((normal, offset))
        elif all(value >= 0 for value in residuals):
            candidates.append((Matrix([-value for value in normal]), -offset))

    seen: set[tuple[tuple[Rational, ...], Rational]] = set()
    facets: list[tuple[Matrix, Rational]] = []
    for normal, offset in candidates:
        key = tuple(Rational(value) for value in normal), offset
        if key not in seen:
            seen.add(key)
            facets.append((normal, offset))
    return facets


def recession_cone_is_trivial(
    normals: Sequence[Sequence[Rational]], dimension: int
) -> bool:
    """Decide whether ``{y : Ay <= 0}`` contains only the zero vector."""

    normals = [list(normal) for normal in normals]
    if dimension == 1:
        return any(row[0] > 0 for row in normals) and any(row[0] < 0 for row in normals)
    if len(normals) < dimension + 1:
        return False
    differences = [
        [normals[index][axis] - normals[0][axis] for axis in range(dimension)]
        for index in range(1, len(normals))
    ]
    try:
        if Matrix(differences).rank() < dimension:
            return False
    except Exception:
        return False
    hull_facets = facets_from_points(normals, dimension)
    return bool(hull_facets) and all(
        Rational(0) < offset for _normal, offset in hull_facets
    )


def vertices_from_halfspaces(
    rows: Sequence[RationalRow], dimension: int
) -> list[tuple[Rational, ...]]:
    """Enumerate feasible, distinct vertices of a bounded H-representation."""

    vertices: list[tuple[Rational, ...]] = []
    for indices in combinations(range(len(rows)), dimension):
        selected = [rows[index] for index in indices]
        matrix = Matrix([coefficients for coefficients, _offset in selected])
        rhs = Matrix([[_offset] for _coefficients, _offset in selected])
        try:
            if matrix.det() == 0:
                continue
            solution = matrix.solve(rhs)
        except (NonInvertibleMatrixError, ValueError):
            continue
        point = tuple(Rational(solution[axis, 0]) for axis in range(dimension))
        if all(
            sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(coefficients, point, strict=True)
            )
            <= offset
            for coefficients, offset in rows
        ):
            vertices.append(point)

    return list(dict.fromkeys(vertices))


__all__ = [
    "RationalRow",
    "facets_from_points",
    "hyperplane_normal",
    "recession_cone_is_trivial",
    "vertices_from_halfspaces",
]
