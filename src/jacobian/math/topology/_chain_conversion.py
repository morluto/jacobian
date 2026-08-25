"""Canonical-value conversion for simplicial chain-complex results."""

from __future__ import annotations

from jacobian.math.chain_complexes.values import (
    MAX_BASIS_SIZE,
    MAX_MATRIX_CELLS,
    ChainComplexValue,
    CoefficientField,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    HomologyConvention,
    SimplexBasis,
    SparseBoundaryMatrix,
)


def canonical_chain_complex_value_from_parts(
    coefficient_ring: ChainCoefficientRing,
    convention: HomologyConvention,
    prime: int | None,
    simplex_bases: tuple[SimplexBasis, ...],
    boundary_matrices: tuple[SparseBoundaryMatrix, ...],
) -> ChainComplexValue | None:
    """Return the canonical value for one eligible simplicial chain complex.

    Only unreduced prime-field results convert: integral boundaries live over
    ZZ rather than QQ or GF(p), and reduced chains retain an augmentation map
    outside the canonical value's representation.  The ordered lexicographic
    face bases remain the implicit column/row ordering of each dense
    differential; simplex labels do not survive because the canonical value
    is based but unlabeled.
    """

    if coefficient_ring is not ChainCoefficientRing.PRIME_FIELD:
        return None
    if convention is HomologyConvention.REDUCED:
        return None
    if prime is None:
        raise ValueError("prime-field chains must declare their modulus")
    basis_sizes = tuple(len(basis.simplices) for basis in simplex_bases)
    total_cells = sum(matrix.rows * matrix.columns for matrix in boundary_matrices)
    if any(size > MAX_BASIS_SIZE for size in basis_sizes):
        raise ValueError(
            f"simplicial chain group exceeds the canonical basis bound {MAX_BASIS_SIZE}"
        )
    if total_cells > MAX_MATRIX_CELLS:
        raise ValueError(
            f"simplicial boundary data exceeds the canonical cell bound "
            f"{MAX_MATRIX_CELLS}"
        )
    # Boundary matrix k maps C_k -> C_{k-1}; the canonical value stores
    # differentials[i] as C_{i+1} -> C_i, i.e. boundary_matrices[i + 1].
    differential_matrices = []
    for matrix in boundary_matrices[1:]:
        dense = [[0] * matrix.columns for _ in range(matrix.rows)]
        for entry in matrix.entries:
            dense[entry.row][entry.column] = entry.value % prime
        differential_matrices.append(
            tuple(tuple(str(value) for value in row) for row in dense)
        )
    return ChainComplexValue(
        coefficient_field=CoefficientField.PRIME_FIELD,
        prime=prime,
        degree_min=0,
        degree_max=len(basis_sizes) - 1,
        basis_sizes=basis_sizes,
        differential_matrices=tuple(differential_matrices),
    )


__all__ = ["canonical_chain_complex_value_from_parts"]
