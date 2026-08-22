"""Typed declarations for Hochschild complex operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.hochschild_complexes._models import (
    HochschildChainComplexRequest,
    HochschildChainComplexResult,
    HochschildHomologyRequest,
    HochschildHomologyResult,
)
from jacobian.math.hochschild_complexes._operations import (
    compute_hochschild_chain_complex,
    compute_hochschild_homology,
)


def hochschild_operation[
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


_COMPLEX_EXAMPLE: dict[str, Any] = {
    "algebra": {
        "prime": 5,
        "dimension": 1,
        "structure_constants": [[[1]]],
    },
    "max_degree": 2,
}


HOCHSCHILD_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    hochschild_operation(
        "hochschild.chain_complex.compute",
        "Compute the Hochschild chain complex with trivial bimodule",
        "Given a finite-dimensional algebra over GF(p) with structure "
        "constants, compute the Hochschild chain complex C_n = A^⊗n with "
        "trivial bimodule coefficients. The differential uses the algebra "
        "multiplication and the standard Hochschild boundary formula.",
        HochschildChainComplexRequest,
        HochschildChainComplexResult,
        compute_hochschild_chain_complex,
        "hochschild",
        "chain-complex",
        "exact",
        examples=(
            example(
                "one_dim_algebra",
                "Compute the Hochschild complex of a 1D algebra over GF(5).",
                _COMPLEX_EXAMPLE,
            ),
        ),
    ),
    hochschild_operation(
        "hochschild.homology.compute",
        "Compute Hochschild homology with trivial bimodule",
        "Given a finite-dimensional algebra over GF(p) with structure "
        "constants, compute the exact Hochschild homology groups HH_n(A, K) "
        "with trivial bimodule coefficients using Gaussian elimination over "
        "GF(p).",
        HochschildHomologyRequest,
        HochschildHomologyResult,
        compute_hochschild_homology,
        "hochschild",
        "homology",
        "exact",
        examples=(
            example(
                "one_dim_homology",
                "Compute Hochschild homology of a 1D algebra over GF(5).",
                _COMPLEX_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = HOCHSCHILD_OPERATIONS

__all__ = ["TOOLS"]
