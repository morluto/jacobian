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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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
        "entries": [[0, 1], [-1, 0]],
        "symmetrizer": [1, 1],
    },
    "mutation_index": 0,
}

_GVECTOR_EXAMPLE: dict[str, Any] = {
    "exchange_matrix": {
        "n": 2,
        "entries": [[0, 1], [-1, 0]],
        "symmetrizer": [1, 1],
    },
}


CLUSTER_ALGEBRA_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    cluster_algebra_operation(
        "cluster_algebra.seed.mutate.compute",
        "Mutate a cluster seed at a specified index",
        "Apply the Fomin-Zelevinsky mutation mu_k to a skew-symmetrizable "
        "exchange matrix B. The mutation transforms B in place: it negates "
        "row k and column k, and adds the rank-1 update "
        "b_{ik} * b_{kj} to b_{ij} for entries where both have the same sign.",
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
        "Compute the initial g-vector matrix (identity) for a cluster seed "
        "with principal coefficients. The g-vectors index the cluster "
        "variables as Laurent polynomials in the initial cluster.",
        GVectorRequest,
        GVectorResult,
        compute_g_vectors,
        "cluster-algebra",
        "g-vector",
        "exact",
        examples=(
            example(
                "a2_g_vectors",
                "Compute g-vectors for the A2 seed.",
                _GVECTOR_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = CLUSTER_ALGEBRA_OPERATIONS

__all__ = ["TOOLS"]
