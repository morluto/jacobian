"""Domain-owned cluster algebra operations."""

from __future__ import annotations

from jacobian.math.cluster_algebras._models import (
    ExchangeMatrix,
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
    SeedMutationResult,
    _identity_matrix,
    encoded_entries,
    parsed_entries,
)


def _mutation_of(matrix: ExchangeMatrix, k: int) -> ExchangeMatrix:
    """Pure Fomin-Zelevinsky mutation mu_k of one exchange matrix.

    Kept free of result-model construction so result validation can replay
    the mutation without recursion.
    """
    n = matrix.n
    old = [list(row) for row in parsed_entries(matrix)]

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
                # Fomin-Zelevinsky: b_ij + [b_ik]_+[b_kj]_+ - [-b_ik]_+[-b_kj]_+
                new[i][j] = old[i][j] + (
                    max(0, b_ik) * max(0, b_kj) - min(0, b_ik) * min(0, b_kj)
                )
    return ExchangeMatrix(
        n=n,
        entries=encoded_entries(tuple(tuple(row) for row in new)),
        symmetrizer=matrix.symmetrizer,
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
    return SeedMutationResult._from_kernel(
        request,
        exchange_matrix=_mutation_of(request.exchange_matrix, request.mutation_index),
    )


def compute_g_vectors(request: GVectorRequest) -> GVectorResult:
    """Compute the g-vector matrix for principal coefficients.

    For the initial seed, the g-vector matrix is the identity matrix.
    """
    return GVectorResult._from_kernel(request)


def verify_seed_mutation_result(result: SeedMutationResult) -> bool:
    """Verify an independently supplied mutation claim within its input bound."""

    if result.mutation_index >= result.source_exchange_matrix.n:
        return False
    return result.exchange_matrix == _mutation_of(
        result.source_exchange_matrix, result.mutation_index
    )


def verify_g_vector_result(result: GVectorResult) -> bool:
    """Verify the fixed initial-seed g-vector convention."""

    return result.g_matrix == _identity_matrix(result.exchange_matrix.n)


__all__ = [
    "compute_g_vectors",
    "mutate_seed",
    "verify_g_vector_result",
    "verify_seed_mutation_result",
]
