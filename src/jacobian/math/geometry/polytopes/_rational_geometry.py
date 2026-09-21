"""Shared exact geometry primitives for rational polytope owners.

The helpers establish geometric facts only. The shared conversion kernel owns
its ray, pair, and coefficient-growth admission; callers own operation-specific
scan and result envelopes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from math import lcm
from typing import cast

from sympy import Matrix, Rational

from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    PolyhedralConversionError,
    PolyhedronConversion,
    halfspaces_to_generators,
    points_to_facets,
)

RationalRow = tuple[Sequence[Rational], Rational]


class RecessionConeComputationError(RuntimeError):
    """Raised when exact recession-cone boundedness cannot be established."""


def determinant_sign(rows: Sequence[Sequence[Rational | Fraction | int]]) -> int:
    """Return the sign of one exact rational orientation determinant."""

    from flint import fmpq, fmpq_mat

    determinant = fmpq_mat(
        [
            [
                fmpq(value)
                if isinstance(value, int)
                else (
                    fmpq(value.numerator, value.denominator)
                    if isinstance(value, Fraction)
                    else fmpq(int(value.p), int(value.q))
                )
                for value in row
            ]
            for row in rows
        ]
    ).det()
    return int(determinant > 0) - int(determinant < 0)


def hyperplane_normal(points: Sequence[Sequence[Rational]]) -> Matrix | None:
    """Return the unique normal through ``dim`` affine points, if one exists."""

    dimension = len(points)
    if dimension == 1:
        return Matrix([Rational(1)])
    from flint import fmpz_mat

    integer_rows = []
    for index in range(1, dimension):
        differences = [
            points[index][axis] - points[0][axis] for axis in range(dimension)
        ]
        ratios = [
            (value.numerator, value.denominator)
            if isinstance(value, Fraction)
            else (
                int(cast(Rational, value).p),
                int(cast(Rational, value).q),
            )
            for value in differences
        ]
        denominator = lcm(*(ratio[1] for ratio in ratios))
        integer_rows.append(
            [numerator * (denominator // divisor) for numerator, divisor in ratios]
        )
    basis, nullity = fmpz_mat(integer_rows).nullspace()
    if int(nullity) != 1:
        return None
    return Matrix([int(basis[row, 0]) for row in range(dimension)])


def iter_facets_from_points(
    vertices: Sequence[Sequence[Rational]], dimension: int
) -> Iterator[tuple[Matrix, Rational]]:
    """Enumerate primitive supporting rows of ``conv(vertices)`` by dual DD."""

    conversion = points_to_facets(vertices, dimension)
    for normal, offset in conversion.facets:
        yield Matrix(normal), cast(Rational, Rational(offset))


def facets_from_points(
    vertices: Sequence[Sequence[Rational]], dimension: int
) -> list[tuple[Matrix, Rational]]:
    """Return distinct oriented supporting rows of ``conv(vertices)``."""

    return list(iter_facets_from_points(vertices, dimension))


def recession_cone_is_trivial(
    normals: Sequence[Sequence[Rational]], dimension: int
) -> bool:
    """Decide whether ``{y : Ay <= 0}`` contains only the zero vector."""

    rows = [(list(normal), Rational(0)) for normal in normals]
    try:
        conversion = halfspaces_to_generators(rows, dimension)
    except PolyhedralConversionError as exc:
        raise RecessionConeComputationError(
            "exact recession-cone computation failed"
        ) from exc
    return not conversion.recession_rays and not conversion.lineality_basis


def polyhedron_from_halfspaces(
    rows: Sequence[RationalRow], dimension: int
) -> PolyhedronConversion:
    """Return one exact H-conversion with generators and classification."""

    try:
        return halfspaces_to_generators(rows, dimension)
    except PolyhedralConversionError as exc:
        raise RuntimeError("exact vertex enumeration computation failed") from exc


def vertices_from_halfspaces(
    rows: Sequence[RationalRow], dimension: int
) -> list[tuple[Rational, ...]]:
    """Enumerate feasible, distinct vertices of a bounded H-representation."""

    conversion = polyhedron_from_halfspaces(rows, dimension)
    return [
        tuple(
            cast(Rational, Rational(value.numerator, value.denominator))
            for value in point
        )
        for point in conversion.vertices
    ]


__all__ = [
    "RationalRow",
    "RecessionConeComputationError",
    "determinant_sign",
    "facets_from_points",
    "hyperplane_normal",
    "iter_facets_from_points",
    "polyhedron_from_halfspaces",
    "recession_cone_is_trivial",
    "vertices_from_halfspaces",
]
