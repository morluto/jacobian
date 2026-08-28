"""Shared Hochschild-differential construction for augmented algebras."""

from __future__ import annotations

from itertools import product as iproduct

StructureConstants = tuple[tuple[tuple[int, ...], ...], ...]


def bar_differential_entries(
    structure_constants: StructureConstants,
    prime: int,
    degree: int,
    augmentation: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Dense Hochschild boundary ``C_degree -> C_{degree-1}`` over ``GF(prime)``.

    Chains are ``A^tensor degree`` and the coefficient module is the trivial
    module ``K`` defined by the augmentation ``epsilon`` (validated to be an
    algebra map by ``AlgebraStructure``). The full differential combines the
    interior multiplication faces with the two augmentation-dependent endpoint
    faces::

        d(a_1 ox ... ox a_n) =
            epsilon(a_1) a_2 ox ... ox a_n
            + sum_j (-1)^j ... ox a_j a_{j+1} ox ...
            + (-1)^n epsilon(a_n) a_1 ox ... ox a_{n-1}

    Both endpoint faces are required for exact ``HH(A, K)``; omitting them
    computes the homology of a different complex. The whole family squares to
    zero because epsilon is multiplicative and the multiplication is
    associative. The operation and the source-binding result validator build
    their matrices through this one construction so an accepted result can
    only revalidate when its entries are exactly the differential of the
    retained augmented algebra.
    """

    dimension = len(structure_constants)
    source_basis = list(iproduct(range(dimension), repeat=degree))
    target_basis = list(iproduct(range(dimension), repeat=degree - 1))
    target_index = {wedge: index for index, wedge in enumerate(target_basis)}
    matrix = [[0] * len(source_basis) for _ in range(len(target_basis))]
    for j, wedge in enumerate(source_basis):
        # Left endpoint face: epsilon(a_1) applied to (a_2, ..., a_n).
        left = augmentation[wedge[0]] % prime
        if left:
            target_idx = target_index[wedge[1:]]
            matrix[target_idx][j] = (matrix[target_idx][j] + left) % prime
        # Interior faces: (-1)^j at each adjacent pair.
        for position in range(degree - 1):
            product = structure_constants[wedge[position]][wedge[position + 1]]
            remaining = wedge[:position] + wedge[position + 2 :]
            sign = (-1) ** (position + 1)
            for coefficient_index, coefficient in enumerate(product):
                if coefficient == 0:
                    continue
                new_wedge = (
                    *remaining[:position],
                    coefficient_index,
                    *remaining[position:],
                )
                target_idx = target_index[new_wedge]
                entry = (sign * int(coefficient)) % prime
                matrix[target_idx][j] = (matrix[target_idx][j] + entry) % prime
        # Wraparound endpoint face: (-1)^n epsilon(a_n) on (a_1, ..., a_{n-1}).
        right = augmentation[wedge[-1]] % prime
        if right:
            entry = ((-1) ** degree * right) % prime
            target_idx = target_index[wedge[:-1]]
            matrix[target_idx][j] = (matrix[target_idx][j] + entry) % prime
    return tuple(tuple(row) for row in matrix)


__all__ = ["bar_differential_entries"]
