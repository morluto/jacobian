"""Typed declarations for cohomology operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.cohomology_operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)
from jacobian.math.cohomology_operations._operations import (
    compute_bockstein,
    compute_steenrod_square,
)


def cohomology_operation[
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


_SQ_EXAMPLE: dict[str, Any] = {
    "cochain_degree": 1,
    "simplex_values": [[0, 1], [1, 2], [0, 2]],
    "simplex_coefficients": [1, 1, 1],
    "square_degree": 0,
}

_BOCKSTEIN_EXAMPLE: dict[str, Any] = {
    "prime": 2,
    "cochain_degree": 1,
    "simplex_values": [[0, 1], [1, 2], [0, 2]],
    "simplex_coefficients": [2, 2, 2],
}


COHOMOLOGY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    cohomology_operation(
        "cohomology.steenrod_square.compute",
        "Compute Steenrod squares Sq^k(x) for a cocycle over GF(2)",
        "Given a simplicial cocycle x of degree n over GF(2) and an integer k, "
        "compute the Steenrod square Sq^k(x). Sq^0 is the identity, Sq^n(x) = "
        "x cup x for a degree-n cocycle, and Sq^k(x) = 0 for k > n (instability). "
        "Steenrod squares are fundamental cohomology operations that distinguish "
        "spaces whose ordinary cohomology groups agree.",
        SteenrodSquareRequest,
        SteenrodSquareResult,
        compute_steenrod_square,
        "cohomology",
        "steenrod",
        "exact",
        examples=(
            example(
                "sq0_identity",
                "Compute Sq^0(x) which is the identity on a 1-cochain.",
                _SQ_EXAMPLE,
            ),
        ),
    ),
    cohomology_operation(
        "cohomology.bockstein.compute",
        "Compute the Bockstein homomorphism for a cocycle over Z/p",
        "Given a cocycle x of degree n with coefficients in Z/p, compute "
        "the Bockstein homomorphism beta(x) in H^{n+1}(Z/p). The Bockstein "
        "is the connecting homomorphism for the short exact sequence "
        "0 -> Z/p -> Z/p^2 -> Z/p -> 0.",
        BocksteinRequest,
        BocksteinResult,
        compute_bockstein,
        "cohomology",
        "bockstein",
        "exact",
        examples=(
            example(
                "bockstein_gf2",
                "Compute the Bockstein of the trivial cocycle over GF(2).",
                _BOCKSTEIN_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = COHOMOLOGY_OPERATIONS

__all__ = ["TOOLS"]
