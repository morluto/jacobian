"""Gowers cube profile kernel."""

from __future__ import annotations

from itertools import product

from jacobian.math.combinatorics.additive.gowers_cube_profile._models import (
    GowersCubeResult,
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
    subset_values = set(subset)
    n = modulus

    cube_count = 0

    # For order s, we need 2^s vertices to all be in A
    # Parameters: base point x and s directions e_1, ..., e_s
    # Vertex (x + sum_{i in S} e_i) for each S subset of {0,...,s-1}

    if order < 1:
        return GowersCubeResult(
            modulus=modulus,
            subset=subset,
            order=order,
            cube_count=0,
            normalized_count=0,
        )

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

    normalized = cube_count  # ||1_A||_{U^s}^{2^s} = cube_count / n^(s+1)

    return GowersCubeResult(
        modulus=modulus,
        subset=subset,
        order=order,
        cube_count=cube_count,
        normalized_count=normalized,
    )
