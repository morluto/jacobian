"""Domain-owned cluster algebra operations."""

from __future__ import annotations

from jacobian.math.cluster_algebras._models import (
    ExchangeMatrix,
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
    SeedMutationResult,
)


def mutate_seed(request: SeedMutationRequest) -> SeedMutationResult:
    """Apply the Fomin-Zelevinsky mutation mu_k to the exchange matrix.

    The mutation at index k transforms the exchange matrix B as follows:
    1. For each pair (i, j), if b_{ik} > 0 and b_{kj} > 0:
       add b_{ik} * b_{kj} to b_{ij}
    2. Negate row k and column k
    3. Leave diagonal entries at 0

    The formula is:
    b'_{ij} = -sgn(i-k) * sgn(j-k) * b_{ij}  if i=k or j=k
    b'_{ij} = b_{ij} + max(0, b_{ik}) * max(0, b_{kj}) + min(0, b_{ik}) * min(0, b_{kj})  otherwise
    """
    matrix = request.exchange_matrix
    k = request.mutation_index
    n = matrix.n
    old = [list(row) for row in matrix.entries]

    new = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == k or j == k:
                if i == k and j == k:
                    new[i][j] = 0
                elif i == k:
                    new[i][j] = -old[i][j]
                else:
                    new[i][j] = -old[i][j]
            else:
                b_ik = old[i][k]
                b_kj = old[k][j]
                new[i][j] = old[i][j] + (
                    max(0, b_ik) * max(0, b_kj) + min(0, b_ik) * min(0, b_kj)
                )

    mutated = ExchangeMatrix(
        n=n,
        entries=tuple(tuple(row) for row in new),
        symmetrizer=matrix.symmetrizer,
    )
    return SeedMutationResult(
        exchange_matrix=mutated,
        mutation_index=k,
    )


def compute_g_vectors(request: GVectorRequest) -> GVectorResult:
    """Compute the g-vector matrix for principal coefficients.

    For the initial seed, the g-vector matrix is the identity matrix.
    """
    n = request.exchange_matrix.n
    g = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
    return GVectorResult(
        n=n,
        g_matrix=g,
    )


__all__ = [
    "compute_g_vectors",
    "mutate_seed",
]
