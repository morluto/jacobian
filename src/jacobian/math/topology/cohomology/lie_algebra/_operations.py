"""Domain-owned Lie algebra homology operations."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as pf_rank,
)
from jacobian.math.topology.cohomology.lie_algebra._models import (
    ChevalleyEilenbergComplexRequest,
    ChevalleyEilenbergComplexResult,
    DifferentialMatrix,
    LieAlgebra,
    LieHomologyGroup,
    LieHomologyRequest,
    LieHomologyResult,
)


def _chain_group_dimensions(dimension: int) -> tuple[int, ...]:
    from math import comb

    return tuple(comb(dimension, k) for k in range(dimension + 1))


def _ce_differentials(
    g: LieAlgebra,
) -> tuple[DifferentialMatrix, ...]:
    """Build every CE differential matrix from the validated structure constants."""

    n = g.dimension
    p = g.prime
    c = g.structure_constants
    group_dims = _chain_group_dimensions(n)

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
                    remaining = tuple(
                        x for k, x in enumerate(wedge) if k != a_idx and k != b_idx
                    )

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
                        diff_matrix[target_idx][j] = (
                            diff_matrix[target_idx][j] + entry
                        ) % p

        differentials.append(
            DifferentialMatrix(
                degree=degree,
                matrix=PrimeFieldMatrix(
                    prime=p,
                    entries=tuple(tuple(row) for row in diff_matrix),
                    columns=source_dim,
                ),
            )
        )
    return tuple(differentials)


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
    return ChevalleyEilenbergComplexResult._from_kernel(
        g,
        _chain_group_dimensions(g.dimension),
        _ce_differentials(g),
    )


def lie_homology_groups(lie_algebra: LieAlgebra) -> tuple[LieHomologyGroup, ...]:
    """Pure Lie-homology core returning the exact groups for one algebra.

    Kept free of result-model construction for the owner-private verifier.
    """
    n = lie_algebra.dimension

    group_dims = _chain_group_dimensions(n)

    # The maintained SymPy DomainMatrix backend is the one exact rank kernel;
    # each CE differential already carries the canonical PrimeFieldMatrix value.
    ranks = [pf_rank(d.matrix) for d in _ce_differentials(lie_algebra)]
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
        groups.append(
            LieHomologyGroup(
                degree=k,
                betti=betti,
                chain_dimension=dim,
            )
        )
    return tuple(groups)


def compute_lie_homology(request: LieHomologyRequest) -> LieHomologyResult:
    """Compute Lie algebra homology with trivial coefficients.

    H_p(g, K) = ker(d_p) / im(d_{p+1})
    betti_p = dim(C_p) - rank(d_p) - rank(d_{p+1})
    """
    g = request.lie_algebra

    return LieHomologyResult._from_kernel(
        g,
        lie_homology_groups(g),
    )


__all__ = [
    "compute_chevalley_eilenberg_complex",
    "compute_lie_homology",
]
