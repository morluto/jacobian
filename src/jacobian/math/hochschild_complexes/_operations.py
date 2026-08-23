"""Domain-owned Hochschild complex operations."""

from __future__ import annotations

from jacobian.math.hochschild_complexes._bar import bar_differential_entries
from jacobian.math.hochschild_complexes._models import (
    MAX_HOCHSCHILD_TENSOR_ELEMENTS,
    AlgebraStructure,
    HochschildChainComplexRequest,
    HochschildChainComplexResult,
    HochschildDifferential,
    HochschildHomologyGroup,
    HochschildHomologyRequest,
    HochschildHomologyResult,
)
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
)


def _pivot_row(
    aug: list[list[int]], col: int, start: int, rows: int, prime: int
) -> int | None:
    """First row at or below ``start`` with a nonzero entry in ``col``."""
    for r in range(start, rows):
        if aug[r][col] % prime != 0:
            return r
    return None


def _scale_and_clear(
    aug: list[list[int]], rank: int, col: int, cols: int, rows: int, prime: int
) -> None:
    """Scale the pivot row to a unit leading entry and clear its column."""
    inv_pivot = pow(aug[rank][col] % prime, prime - 2, prime)
    for c in range(cols):
        aug[rank][c] = (aug[rank][c] * inv_pivot) % prime
    for r, row in enumerate(aug):
        factor = row[col] % prime
        if r != rank and factor != 0:
            for c in range(cols):
                row[c] = (row[c] - factor * aug[rank][c]) % prime


def _gaussian_rank(matrix: list[list[int]], prime: int) -> int:
    """Compute rank of a matrix over GF(prime)."""
    rows = len(matrix)
    if rows == 0:
        return 0
    cols = len(matrix[0])
    if cols == 0:
        return 0
    aug = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = _pivot_row(aug, col, rank, rows, prime)
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        _scale_and_clear(aug, rank, col, cols, rows, prime)
        rank += 1
        if rank >= rows:
            break
    return rank


def _boundary_rank(
    algebra: AlgebraStructure,
    degree: int,
    element_budget: int = MAX_HOCHSCHILD_TENSOR_ELEMENTS,
) -> int:
    """GF(p)-rank of the Hochschild boundary C_degree -> C_{degree-1}."""
    n = algebra.dimension
    if n**degree > element_budget:
        raise ValueError("requested degree exceeds the supported tensor-element budget")
    if degree == 0:
        return 0
    matrix = bar_differential_entries(
        algebra.structure_constants,
        algebra.prime,
        degree,
        algebra.augmentation,
    )
    return _gaussian_rank([list(row) for row in matrix], algebra.prime)


def compute_hochschild_chain_complex(
    request: HochschildChainComplexRequest,
) -> HochschildChainComplexResult:
    """Compute the exact Hochschild chain complex with trivial coefficients.

    K = GF(p) carries the trivial A-bimodule structure defined by the retained
    augmentation epsilon. The differential on C_k = A^tensor-k is the full
    Hochschild boundary: interior adjacent multiplications plus both
    augmentation-dependent endpoint faces. It squares to zero because epsilon
    is an algebra homomorphism and the multiplication is associative.
    """
    alg = request.algebra
    n = alg.dimension
    p = alg.prime
    max_deg = request.max_degree

    group_dims = [1]
    for k in range(1, max_deg + 1):
        group_dims.append(n**k)

    differentials = []

    for degree in range(1, max_deg + 1):
        differentials.append(
            HochschildDifferential(
                degree=degree,
                matrix=PrimeFieldMatrix(
                    prime=p,
                    entries=tuple(
                        bar_differential_entries(
                            alg.structure_constants, p, degree, alg.augmentation
                        )
                    ),
                    columns=n**degree,
                ),
            )
        )

    return HochschildChainComplexResult(
        algebra=alg,
        algebra_dimension=n,
        group_dimensions=tuple(group_dims),
        differentials=tuple(differentials),
        prime=p,
    )


def hochschild_homology_groups(
    algebra: AlgebraStructure,
    max_degree: int,
) -> tuple[HochschildHomologyGroup, ...]:
    """Pure Hochschild-homology core returning the exact groups for one algebra.

    Kept free of result-model construction so result validation can replay
    the bounded rank computation without recursion.
    """
    n = algebra.dimension

    group_dims = [1]
    for k in range(1, max_degree + 1):
        group_dims.append(n**k)

    ranks = [_boundary_rank(algebra, degree) for degree in range(1, max_degree + 2)]

    groups = []
    for k in range(max_degree + 1):
        rank_d_out = ranks[k]
        rank_d_in = ranks[k - 1] if k > 0 else 0
        dim = group_dims[k]
        betti = dim - rank_d_out - rank_d_in
        if betti < 0:
            raise ValueError(
                "internal inconsistency: differentials do not square to zero"
            )
        groups.append(
            HochschildHomologyGroup(
                degree=k,
                betti=betti,
            )
        )
    return tuple(groups)


def compute_hochschild_homology(
    request: HochschildHomologyRequest,
) -> HochschildHomologyResult:
    """Compute exact Hochschild homology HH_n(A, K) for trivial coefficients."""
    alg = request.algebra

    return HochschildHomologyResult(
        algebra=alg,
        max_degree=request.max_degree,
        groups=hochschild_homology_groups(alg, request.max_degree),
        prime=alg.prime,
    )


__all__ = [
    "compute_hochschild_chain_complex",
    "compute_hochschild_homology",
]
