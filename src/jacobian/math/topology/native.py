"""Native topology functions exposing cross-domain canonical values."""

from __future__ import annotations

from jacobian.math.chain_complexes.values import (
    MAX_BASIS_SIZE,
    MAX_MATRIX_CELLS,
    ChainComplexValue,
    CoefficientField,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexResult,
    HomologyConvention,
)

__all__ = ["simplicial_chain_complex_value"]


def simplicial_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """The canonical chain-complex value of one simplicial chain complex.

    Accepts the producer's ``ChainComplexResult`` unchanged and returns the
    domain-owned ``ChainComplexValue`` consumed by homology, tensor, map,
    and cone operations, so no caller-side reconstruction is needed. The
    ordered lexicographic face bases remain the implicit column/row
    ordering of each dense differential; simplex labels do not survive
    because the canonical value is based but unlabeled.

    Unreduced prime-field results only: integral boundaries live over ZZ
    rather than QQ or GF(p), and reduced chains carry an augmentation map
    outside the canonical value's representation.
    """
    if result.coefficient_ring is not ChainCoefficientRing.PRIME_FIELD:
        raise ValueError(
            "only prime-field simplicial chain complexes convert to a "
            "canonical chain-complex value"
        )
    if result.convention is HomologyConvention.REDUCED:
        raise ValueError(
            "reduced simplicial chains carry an augmentation map outside "
            "the canonical chain-complex value"
        )
    if result.prime is None:
        raise ValueError("prime-field chains must declare their modulus")
    prime = result.prime
    basis_sizes = tuple(len(basis.simplices) for basis in result.simplex_bases)
    total_cells = sum(
        matrix.rows * matrix.columns for matrix in result.boundary_matrices
    )
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
    for matrix in result.boundary_matrices[1:]:
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
