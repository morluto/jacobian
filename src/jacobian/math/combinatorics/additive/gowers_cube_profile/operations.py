"""Gowers cube profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.gowers_cube_profile._models import (
    MAX_GOWERS_CUBE_ORDER,
    MAX_GOWERS_CUBE_VERTEX_CHECKS,
    GowersCubeResult,
    gowers_cube_work,
)

__all__ = ["compute_gowers_cube_profile"]


def compute_gowers_cube_profile(
    modulus: int,
    subset: tuple[int, ...],
    order: int,
) -> GowersCubeResult:
    """Count the exact number of labelled s-dimensional affine cubes in A.

    An s-dimensional cube is parameterized by a base point x and s direction
    vectors e_1,...,e_s. The 2^s vertices are x + sum_{i in S} e_i for
    every subset S of {1,...,s}. All vertices must lie in A.

    For Z/mZ, we enumerate over all bases and directions.
    """
    if modulus <= 0 or not 1 <= order <= MAX_GOWERS_CUBE_ORDER:
        raise OperationDomainValidationError(
            location=("modulus", "order"),
            code="gowers_cube.positive_bounded_parameters",
            message=(
                "Gowers cube enumeration requires a positive modulus and order "
                f"from 1 through {MAX_GOWERS_CUBE_ORDER}"
            ),
        )
    if len(subset) != len(set(subset)) or any(
        not 0 <= value < modulus for value in subset
    ):
        raise OperationDomainValidationError(
            location=("subset",),
            code="gowers_cube.canonical_subset",
            message="subset must contain distinct canonical residues modulo modulus",
        )
    if len(subset) > 1 and gowers_cube_work(modulus, order) > MAX_GOWERS_CUBE_VERTEX_CHECKS:
        raise OperationDomainValidationError(
            location=("modulus", "order"),
            code="gowers_cube.work_exceeded",
            message="Gowers cube enumeration exceeds the 2000000-vertex-check bound",
        )
    subset_values = set(subset)
    n = modulus

    if len(subset_values) <= 1:
        cube_count = 0 if not subset_values else 1
        return GowersCubeResult(
            modulus=modulus,
            subset=subset,
            order=order,
            cube_count=cube_count,
            normalized_count=CanonicalRational.from_fraction(
                Fraction(cube_count, modulus ** (order + 1))
            ),
        )

    cube_count = 0

    # For order s, we need 2^s vertices to all be in A
    # Parameters: base point x and s directions e_1, ..., e_s
    # Vertex (x + sum_{i in S} e_i) for each S subset of {0,...,s-1}

    # Enumerate all possible base points and direction combinations
    # For small modulus and order, this is feasible
    for x in range(n):
        for e_tuple in product(range(n), repeat=order):
            # Compute all 2^s vertices
            all_in = True
            for mask in range(1 << order):
                vertex = x
                for bit in range(order):
                    if mask & (1 << bit):
                        vertex = (vertex + e_tuple[bit]) % n
                if vertex not in subset_values:
                    all_in = False
                    break
            if all_in:
                cube_count += 1

    normalized = CanonicalRational.from_fraction(
        Fraction(cube_count, modulus ** (order + 1))
    )

    return GowersCubeResult(
        modulus=modulus,
        subset=subset,
        order=order,
        cube_count=cube_count,
        normalized_count=normalized,
    )
