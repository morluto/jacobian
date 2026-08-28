"""Typed declarations for cluster algebra operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.cluster_algebras._models import (
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
    SeedMutationResult,
)
from jacobian.math.cluster_algebras._operations import (
    compute_g_vectors,
    mutate_seed,
)


def cluster_algebra_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    cluster_algebra_operation(
        "cluster_algebra.seed.mutate.compute",
        "Mutate a cluster seed at a specified index",
        "Apply the Fomin-Zelevinsky mutation mu_k to a skew-symmetrizable "
        "exchange matrix B. The mutation negates row k and column k, and for "
        "i,j != k updates b_{ij} to b_{ij} + max(0,b_{ik})*max(0,b_{kj}) "
        "- max(0,-b_{ik})*max(0,-b_{kj}); equivalently, add b_{ik}*b_{kj} when "
        "both are positive and subtract |b_{ik}*b_{kj}| when both are negative "
        "(no change when signs differ).",
        SeedMutationRequest,
        SeedMutationResult,
        mutate_seed,
        "cluster-algebra",
        "mutation",
        "exact",
        examples=(
            example(
                "a2_mutation",
                "Mutate the A2 cluster seed at index 0.",
                _MUTATION_EXAMPLE,
            ),
        ),
    ),
    cluster_algebra_operation(
        "cluster_algebra.g_vector.compute",
        "Compute the g-vector matrix for principal coefficients",
        "Compute the initial g-vector matrix for a cluster seed with principal "
        "coefficients, retaining the source exchange matrix and Fomin-Zelevinsky "
        "convention.",
        GVectorRequest,
        GVectorResult,
        compute_g_vectors,
        "cluster-algebra",
        "g-vector",
        "exact",
        examples=(
            example(
                "a2_g_vectors",
                "Compute the initial g-vectors for the A2 seed.",
                _GVECTOR_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
