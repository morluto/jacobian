"""Bounded owner-local admission kernels for rational polytopes.

These adapters keep request admission separate from result construction. They
are intentionally private and are called by the owning operation adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy import Rational

    from jacobian.math.geometry.polytopes.values import Halfspace, Vertex


def volume_vertices_for_admission(
    vertices: tuple[Vertex, ...],
) -> tuple[tuple[Rational, ...], int]:
    """Convert a V-representation for its bounded volume admission check."""

    from jacobian.math.geometry.polytopes._operations import (
        _vertices_from_v_representation,
    )

    return _vertices_from_v_representation(vertices)


def bounded_h_vertices_for_admission(
    halfspaces: tuple[Halfspace, ...],
) -> tuple[bool, list[tuple[Rational, ...]], int]:
    """Decide boundedness and enumerate the admitted H-representation vertices."""

    from jacobian.math.geometry.polytopes._operations import (
        _is_bounded_h,
        _vertices_from_h_representation,
    )

    if not _is_bounded_h(halfspaces):
        return False, [], len(halfspaces[0].coefficients)
    vertices, dimension = _vertices_from_h_representation(halfspaces)
    return True, vertices, dimension


def triangulation_for_volume_admission(
    points: list[list[Rational]], dimension: int
) -> tuple[list[list[Rational]], list[tuple[int, ...]]]:
    """Mirror the kernel's redundant-row filtering and triangulation path."""

    from jacobian.math.geometry.polytopes._operations import (
        _filter_redundant_vertices,
        _triangulate,
    )

    reduced = _filter_redundant_vertices(points, dimension)
    return reduced, _triangulate(reduced, dimension)


__all__ = [
    "bounded_h_vertices_for_admission",
    "triangulation_for_volume_admission",
    "volume_vertices_for_admission",
]
