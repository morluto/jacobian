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


_SQ_EXAMPLE: dict[str, Any] = {
    "cochain_degree": 1,
    "simplex_values": [[0, 1], [0, 2]],
    "simplex_coefficients": [1, 1],
    "square_degree": 0,
    "ambient_simplices": [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]],
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
        "Compute Sq^0, Sq^n (cup) and Sq^k=0 for k>n over GF(2) for cocycles",
        "Given a cochain x of degree n over GF(2) compute Sq^k(x). "
        "Supported: Sq^0(x)=x (identity), Sq^n(x)=x cup x (top, targets "
        "2n-simplices), Sq^k=0 for k>n (instability; constant work, admitted "
        "whenever the returned degree n+k stays within the declared "
        "result-degree budget); intermediate 0<k<n need cup-i and are "
        "rejected. Nonzero cochains require "
        "ambient_simplices or ambient_complex for cocycle verification; only "
        "the zero cochain is admissible without ambient. Top squares require "
        "ambient to locate targets.",
        SteenrodSquareRequest,
        SteenrodSquareResult,
        compute_steenrod_square,
        "cohomology",
        "steenrod",
        "exact",
        examples=(
            example(
                "sq0_identity",
                "Compute Sq^0(x)=x for the 1-cocycle d(vertex 0) on the triangle; nonzero cochains require ambient for cocycle verification.",
                _SQ_EXAMPLE,
            ),
        ),
    ),
    cohomology_operation(
        "cohomology.bockstein.compute",
        "Compute the Bockstein homomorphism of the trivial cocycle over Z/p",
        "Given a degree-n cochain over Z/p that is zero modulo p, return the "
        "exact zero Bockstein beta(x) in H^{n+1}(Z/p). Nontrivial cocycles "
        "require the ambient simplicial complex and are rejected as "
        "unsupported by this bounded operation.",
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
