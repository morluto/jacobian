"""Shared bar-differential construction for Hochschild contracts."""

from __future__ import annotations

from itertools import product as iproduct

StructureConstants = tuple[tuple[tuple[int, ...], ...], ...]


def bar_differential_entries(
    structure_constants: StructureConstants,
    prime: int,
    degree: int,
) -> tuple[tuple[int, ...], ...]:
    """Dense adjacent-multiplication boundary ``C_degree -> C_{degree-1}``.

    The bar differential multiplies adjacent factors,
    ``b'(a_1 ox ... ox a_k) = sum_j (-1)^j ... ox a_j a_{j+1} ox ...``,
    over ``GF(prime)`` with trivial endpoint actions. Both the operation and
    the source-binding result validator build their matrices through this one
    construction so an accepted result can only revalidate when its entries
    are exactly the differential of the retained algebra.
    """

    dimension = len(structure_constants)
    source_basis = list(iproduct(range(dimension), repeat=degree))
    target_basis = list(iproduct(range(dimension), repeat=degree - 1))
    target_index = {wedge: index for index, wedge in enumerate(target_basis)}
    matrix = [[0] * len(source_basis) for _ in range(len(target_basis))]
    for j, wedge in enumerate(source_basis):
        for position in range(degree - 1):
            product = structure_constants[wedge[position]][wedge[position + 1]]
            remaining = wedge[:position] + wedge[position + 2 :]
            sign = (-1) ** position
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
    return tuple(tuple(row) for row in matrix)


__all__ = ["bar_differential_entries"]
