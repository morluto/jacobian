"""Domain-owned Lie algebra homology operations."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.lie_algebra_homology._models import (
    ChevalleyEilenbergComplexRequest,
    ChevalleyEilenbergComplexResult,
    DifferentialMatrix,
    LieHomologyGroup,
    LieHomologyRequest,
    LieHomologyResult,
)


def _find_pivot_row(aug: list[list[int]], prime: int, rank: int, col: int) -> int | None:
    """Return the first row index >= rank with a nonzero entry in col."""
    for r in range(rank, len(aug)):
        if aug[r][col] % prime != 0:
            return r
    return None


def _eliminate_column(
    aug: list[list[int]], prime: int, rank: int, col: int, rows: int, cols: int
) -> None:
    """Scale the pivot row to 1 and clear col in every other row."""
    inv_pivot = pow(aug[rank][col] % prime, prime - 2, prime)
    for c in range(cols):
        aug[rank][c] = (aug[rank][c] * inv_pivot) % prime
    for r in range(rows):
        if r == rank:
            continue
        factor = aug[r][col] % prime
        if factor != 0:
            for c in range(cols):
                aug[r][c] = (aug[r][c] - factor * aug[rank][c]) % prime


def _gaussian_rank(matrix: list[list[int]], prime: int) -> int:
    """Compute rank of a matrix over GF(prime) via Gaussian elimination."""
    rows = len(matrix)
    if rows == 0:
        return 0
    cols = len(matrix[0])
    if cols == 0:
        return 0
    aug = [row[:] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = _find_pivot_row(aug, prime, rank, col)
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        _eliminate_column(aug, prime, rank, col, rows, cols)
        rank += 1
        if rank >= rows:
            break
    return rank


def _wedge_index(indices: tuple[int, ...], dim: int) -> int:
    """Convert a sorted tuple of indices to a lexicographic wedge basis index."""
    result = 0
    for i, idx in enumerate(indices):
        result += idx * (dim ** i)
    return result


def compute_chevalley_eilenberg_complex(
    request: ChevalleyEilenbergComplexRequest,
) -> ChevalleyEilenbergComplexResult:
    """Compute the Chevalley-Eilenberg chain complex for a Lie algebra with trivial coefficients.

    For a Lie algebra g of dimension n, the chain groups are:
    C_p = Lambda^p(g) with dimension C(n, p) for p = 0, ..., n.

    The differential d_p: C_p -> C_{p-1} is defined by:
    d_p(e_{i1} ^ ... ^ e_{ip}) = sum_{a<b} (-1)^(a+b+pi) * [e_ia, e_ib] ^ e_{i1} ^ ... ^ hat(e_ia) ^ ... ^ hat(e_ib) ^ ...

    where pi is the parity of the permutation inserting the bracket basis
    factor into the canonical sorted wedge position of the remaining
    factors. With trivial coefficients, the module is the base field, so
    the chain complex is purely determined by the Lie bracket structure
    constants, which the request model validates as a Lie algebra.
    """
    g = request.lie_algebra
    n = g.dimension
    p = g.prime
    c = g.structure_constants

    group_dims = []
    for k in range(n + 1):
        from math import comb
        group_dims.append(comb(n, k))

    differentials = []

    for degree in range(1, n + 1):
        source_dim = group_dims[degree]
        target_dim = group_dims[degree - 1]

        if source_dim == 0 or target_dim == 0:
            continue

        diff_matrix = [[0] * source_dim for _ in range(target_dim)]

        source_basis = list(combinations(range(n), degree))
        target_basis = list(combinations(range(n), degree - 1))

        for j, wedge in enumerate(source_basis):
            for a_idx, a in enumerate(wedge):
                for b_idx in range(a_idx + 1, len(wedge)):
                    b = wedge[b_idx]
                    bracket = c[a][b]
                    remaining = tuple(x for k, x in enumerate(wedge) if k != a_idx and k != b_idx)

                    for k, coeff in enumerate(bracket):
                        if coeff == 0:
                            continue
                        if k in remaining:
                            continue
                        insertion_pos = sum(1 for x in remaining if x < k)
                        new_wedge = tuple(sorted((*remaining, k)))
                        # Moving the bracket factor e_k from the front of the
                        # term into its canonical sorted wedge position costs
                        # the parity of the insertion; without it d^2 != 0.
                        entry_sign = (-1) ** (a_idx + b_idx + insertion_pos)
                        target_idx = target_basis.index(new_wedge)
                        entry = (entry_sign * int(coeff)) % p
                        diff_matrix[target_idx][j] = (diff_matrix[target_idx][j] + entry) % p

        differentials.append(DifferentialMatrix(
            degree=degree,
            source_dim=source_dim,
            target_dim=target_dim,
            entries=tuple(tuple(row) for row in diff_matrix),
        ))

    return ChevalleyEilenbergComplexResult(
        dimension=n,
        group_dimensions=tuple(group_dims),
        differentials=tuple(differentials),
        prime=p,
    )


def compute_lie_homology(request: LieHomologyRequest) -> LieHomologyResult:
    """Compute Lie algebra homology with trivial coefficients.

    H_p(g, K) = ker(d_p) / im(d_{p+1})
    betti_p = dim(C_p) - rank(d_p) - rank(d_{p+1})
    """
    g = request.lie_algebra
    n = g.dimension
    p = g.prime

    from math import comb
    group_dims = [comb(n, k) for k in range(n + 1)]

    ranks = []
    for degree in range(1, n + 1):
        source_dim = group_dims[degree]
        target_dim = group_dims[degree - 1]

        if source_dim == 0 or target_dim == 0:
            ranks.append(0)
            continue

        source_basis = list(combinations(range(n), degree))
        target_basis = list(combinations(range(n), degree - 1))

        diff_matrix = [[0] * len(source_basis) for _ in range(len(target_basis))]
        c = g.structure_constants

        for j, wedge in enumerate(source_basis):
            for a_idx, a in enumerate(wedge):
                for b_idx in range(a_idx + 1, len(wedge)):
                    b = wedge[b_idx]
                    bracket = c[a][b]
                    remaining = tuple(x for k, x in enumerate(wedge) if k != a_idx and k != b_idx)
                    for k, coeff in enumerate(bracket):
                        if coeff == 0:
                            continue
                        if k in remaining:
                            continue
                        insertion_pos = sum(1 for x in remaining if x < k)
                        new_wedge = tuple(sorted((*remaining, k)))
                        # Same graded-antisymmetric sign as the complex
                        # operation; see the comment there.
                        entry_sign = (-1) ** (a_idx + b_idx + insertion_pos)
                        target_idx = target_basis.index(new_wedge)
                        entry = (entry_sign * int(coeff)) % p
                        diff_matrix[target_idx][j] = (diff_matrix[target_idx][j] + entry) % p

        ranks.append(_gaussian_rank(diff_matrix, p))

    ranks.append(0)

    groups = []
    for k in range(n + 1):
        rank_d_k = ranks[k] if k < len(ranks) else 0
        rank_d_k1 = ranks[k - 1] if k > 0 else 0
        dim = group_dims[k]
        # The request boundary guarantees a Lie bracket, so im(d_{k+1}) is
        # contained in ker(d_k) and this difference cannot be negative; an
        # inconsistency must surface rather than be clamped away.
        betti = dim - rank_d_k - rank_d_k1
        groups.append(LieHomologyGroup(
            degree=k,
            betti=betti,
            dimension=dim,
        ))

    return LieHomologyResult(
        groups=tuple(groups),
        dimension=n,
        prime=p,
    )


__all__ = [
    "compute_chevalley_eilenberg_complex",
    "compute_lie_homology",
]
