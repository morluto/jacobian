"""Domain-owned Hochschild complex operations."""

from __future__ import annotations

from itertools import product as iproduct

from jacobian.math.hochschild_complexes._bar import (
    StructureConstants,
    bar_differential_entries,
)
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


def _pivot_row(aug: list[list[int]], col: int, start: int, rows: int, prime: int) -> int | None:
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


def _adjacent_boundary_rank(
    c: StructureConstants,
    n: int,
    p: int,
    degree: int,
    element_budget: int = MAX_HOCHSCHILD_TENSOR_ELEMENTS,
) -> int:
    """GF(p)-rank of the adjacent-multiplication boundary C_degree -> C_{degree-1}."""
    source_dim = n ** degree
    if source_dim > element_budget:
        raise ValueError(
            "requested degree exceeds the supported tensor-element budget"
        )
    target_dim = n ** (degree - 1)
    matrix = [[0] * source_dim for _ in range(target_dim)]
    source_basis = list(iproduct(range(n), repeat=degree))
    target_basis = list(iproduct(range(n), repeat=degree - 1))
    for j, wedge in enumerate(source_basis):
        for k_pos in range(degree - 1):
            product = c[wedge[k_pos]][wedge[k_pos + 1]]
            remaining = wedge[:k_pos] + wedge[k_pos + 2:]
            sign = (-1) ** k_pos
            for coeff_idx, coeff in enumerate(product):
                if coeff == 0:
                    continue
                new_wedge = (*remaining[:k_pos], coeff_idx, *remaining[k_pos:])
                target_idx = target_basis.index(tuple(new_wedge))
                entry = (sign * int(coeff)) % p
                matrix[target_idx][j] = (matrix[target_idx][j] + entry) % p
    return _gaussian_rank(matrix, p)


def compute_hochschild_chain_complex(
    request: HochschildChainComplexRequest,
) -> HochschildChainComplexResult:
    """Compute the reduced bar chain complex with trivial coefficients.

    The bar differential multiplies adjacent factors:
    b'(e_{i1} ⊗ ... ⊗ e_{ik}) = Σ_j (-1)^j ...⊗ e_{ij}·e_{ij+1} ⊗...
    This is the normalized bar differential, which squares to zero for any
    associative algebra. Its homology is the bar homology with trivial
    coefficients; it coincides with Hochschild homology HH(A,K) only when the
    bimodule action of A on K is via the zero augmentation (endpoint terms
    vanish). For unital augmented algebras, the full Hochschild differential
    includes augmentation-dependent endpoint faces; this operation exposes the
    reduced bar complex. No cyclic wraparound term is applied.
    """
    alg = request.algebra
    n = alg.dimension
    p = alg.prime
    max_deg = request.max_degree
    c = alg.structure_constants

    group_dims = [1]
    for k in range(1, max_deg + 1):
        group_dims.append(n ** k)

    differentials = []

    for degree in range(1, max_deg + 1):
        differentials.append(HochschildDifferential(
            degree=degree,
            source_dim=n ** degree,
            target_dim=n ** (degree - 1),
            entries=bar_differential_entries(c, p, degree),
        ))

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
    """Pure bar-homology core returning the exact groups for one algebra.

    Kept free of result-model construction so result validation can replay
    the bounded rank computation without recursion.
    """
    n = algebra.dimension
    p = algebra.prime
    c = algebra.structure_constants

    group_dims = [1]
    for k in range(1, max_degree + 1):
        group_dims.append(n ** k)

    ranks = [
        _adjacent_boundary_rank(c, n, p, degree)
        for degree in range(1, max_degree + 1)
    ]
    ranks.append(_adjacent_boundary_rank(c, n, p, max_degree + 1))

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
        groups.append(HochschildHomologyGroup(
            degree=k,
            betti=betti,
        ))
    return tuple(groups)


def compute_hochschild_homology(
    request: HochschildHomologyRequest,
) -> HochschildHomologyResult:
    """Compute reduced bar (trivial-coefficient Hochschild) homology."""
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
