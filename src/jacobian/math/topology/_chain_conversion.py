"""Canonical-value conversion for simplicial chain-complex results."""

from __future__ import annotations

from jacobian.math.topology._models import (
    ChainCoefficientRing,
    HomologyConvention,
    SimplexBasis,
    SparseBoundaryMatrix,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


def canonical_chain_complex_value_from_parts(
    coefficient_ring: ChainCoefficientRing,
    convention: HomologyConvention,
    prime: int | None,
    simplex_bases: tuple[SimplexBasis, ...],
    boundary_matrices: tuple[SparseBoundaryMatrix, ...],
    augmentation: SparseBoundaryMatrix | None,
) -> ChainComplexValue:
    """Return the canonical based complex carried by a simplicial producer.

    Reduced homology is represented without an exceptional side channel: the
    augmentation target is the rank-one group in degree ``-1`` and the
    augmentation itself is the ordinary differential ``C_0 -> C_-1``.  The
    ordered lexicographic simplex bases remain the implicit coordinates of the
    dense matrices; the canonical chain value is based but unlabeled.
    """

    if coefficient_ring is ChainCoefficientRing.PRIME_FIELD and prime is None:
        raise ValueError("prime-field chains must declare their modulus")
    if coefficient_ring is ChainCoefficientRing.INTEGER and prime is not None:
        raise ValueError("integer chains must not declare a prime modulus")
    simplicial_sizes = tuple(len(basis.simplices) for basis in simplex_bases)
    reduced = convention is HomologyConvention.REDUCED
    if reduced and augmentation is None:
        raise ValueError("reduced chains require an augmentation matrix")
    if not reduced and augmentation is not None:
        raise ValueError("unreduced chains must not carry an augmentation matrix")
    basis_sizes = (1, *simplicial_sizes) if reduced else simplicial_sizes
    matrices = (
        (augmentation, *boundary_matrices[1:])
        if augmentation is not None
        else boundary_matrices[1:]
    )
    # Boundary matrix k maps C_k -> C_{k-1}; the canonical value stores the
    # same maps in increasing target degree. Reduced chains prepend the
    # augmentation C_0 -> C_-1.
    differential_matrices = []
    for matrix in matrices:
        if matrix is None:
            raise ValueError("canonical differential is unexpectedly absent")
        dense = [[0] * matrix.columns for _ in range(matrix.rows)]
        for entry in matrix.entries:
            dense[entry.row][entry.column] = (
                entry.value if prime is None else entry.value % prime
            )
        differential_matrices.append(
            tuple(tuple(str(value) for value in row) for row in dense)
        )
    return ChainComplexValue(
        coefficient_ring=(
            CoefficientRing.INTEGER
            if coefficient_ring is ChainCoefficientRing.INTEGER
            else CoefficientRing.PRIME_FIELD
        ),
        prime=prime,
        degree_min=-1 if reduced else 0,
        degree_max=(-1 if reduced else 0) + len(basis_sizes) - 1,
        basis_sizes=basis_sizes,
        differential_matrices=tuple(differential_matrices),
    )


__all__ = ["canonical_chain_complex_value_from_parts"]
