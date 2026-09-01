"""Typed declarations for cluster algebra operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.cluster_algebras._models import (
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
    SeedMutationResult,
)
from jacobian.math.cluster_algebras.operations import g_vectors, mutate_seed


def _run_seed_mutation(request: SeedMutationRequest) -> SeedMutationResult:
    return mutate_seed(request.exchange_matrix, request.mutation_index)


def _run_g_vectors(request: GVectorRequest) -> GVectorResult:
    return g_vectors(request.exchange_matrix)


_MUTATION_EXAMPLE: dict[str, Any] = {
    "exchange_matrix": {
        "n": 2,
        "entries": [["0", "1"], ["-1", "0"]],
        "symmetrizer": ["1", "1"],
    },
    "mutation_index": 0,
}

_GVECTOR_EXAMPLE: dict[str, Any] = {
    "exchange_matrix": {
        "n": 2,
        "entries": [["0", "1"], ["-1", "0"]],
        "symmetrizer": ["1", "1"],
    },
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="cluster_algebra.seed.mutate.compute",
        title="Mutate a cluster seed at a specified index",
        description="Apply the Fomin-Zelevinsky mutation mu_k to a skew-symmetrizable "
        "exchange matrix B. The mutation negates row k and column k, and for "
        "i,j != k updates b_{ij} to b_{ij} + max(0,b_{ik})*max(0,b_{kj}) "
        "- max(0,-b_{ik})*max(0,-b_{kj}); equivalently, add b_{ik}*b_{kj} when "
        "both are positive and subtract |b_{ik}*b_{kj}| when both are negative "
        "(no change when signs differ).",
        request_type=SeedMutationRequest,
        result_type=SeedMutationResult,
        run=_run_seed_mutation,
        tags=("cluster-algebra", "mutation", "exact"),
        examples=(
            OperationExample(
                name="a2_mutation",
                description="Mutate the A2 cluster seed at index 0.",
                input=_MUTATION_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="cluster_algebra.g_vector.compute",
        title="Compute the g-vector matrix for principal coefficients",
        description="Compute the initial g-vector matrix for a cluster seed with principal "
        "coefficients, retaining the source exchange matrix and Fomin-Zelevinsky "
        "convention.",
        request_type=GVectorRequest,
        result_type=GVectorResult,
        run=_run_g_vectors,
        tags=("cluster-algebra", "g-vector", "exact"),
        examples=(
            OperationExample(
                name="a2_g_vectors",
                description="Compute the initial g-vectors for the A2 seed.",
                input=_GVECTOR_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
