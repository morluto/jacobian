"""Typed declarations for cohomology operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cohomology.operations._models import (
    BocksteinRequest,
    BocksteinResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)
from jacobian.math.topology.cohomology.operations.operations import (
    bockstein,
    steenrod_square,
)


def _run_steenrod_square(request: SteenrodSquareRequest) -> SteenrodSquareResult:
    return steenrod_square(
        request.cochain_degree,
        request.simplex_values,
        request.simplex_coefficients,
        request.square_degree,
        request.ambient_simplices,
        request.ambient_complex,
    )


def _run_bockstein(request: BocksteinRequest) -> BocksteinResult:
    return bockstein(
        request.prime,
        request.cochain_degree,
        request.simplex_values,
        request.simplex_coefficients,
        request.ambient_simplices,
        request.ambient_complex,
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="cohomology.steenrod_square.compute",
        title="Compute Sq^0, Sq^n (cup) and Sq^k=0 for k>n over GF(2) for cocycles",
        description="Given a cochain x of degree n over GF(2) compute Sq^k(x). "
        "Supported: Sq^0(x)=x (identity), Sq^n(x)=x cup x (top, targets "
        "2n-simplices), Sq^k=0 for k>n (instability; constant work, admitted "
        "whenever the returned degree n+k stays within the declared "
        "result-degree budget); intermediate 0<k<n need cup-i and are "
        "rejected. Nonzero cochains require "
        "ambient_simplices or ambient_complex for cocycle verification; only "
        "the zero cochain is admissible without ambient. Top squares require "
        "ambient to locate targets.",
        request_type=SteenrodSquareRequest,
        result_type=SteenrodSquareResult,
        run=_run_steenrod_square,
        tags=("cohomology", "steenrod", "exact"),
        examples=(
            OperationExample(
                name="sq0_identity",
                description="Compute Sq^0(x)=x for the 1-cocycle d(vertex 0) on the triangle; nonzero cochains require ambient for cocycle verification.",
                input=_SQ_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="cohomology.bockstein.compute",
        title="Compute the bounded Bockstein homomorphism",
        description="Compute the supported exact zero Bockstein branch for a bounded cochain "
        "over Z/p. Nonzero cocycles are rejected as unsupported by this operation.",
        request_type=BocksteinRequest,
        result_type=BocksteinResult,
        run=_run_bockstein,
        tags=("cohomology", "bockstein", "exact"),
        examples=(
            OperationExample(
                name="bockstein_gf2",
                description="Compute the Bockstein of the trivial cocycle over GF(2).",
                input=_BOCKSTEIN_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
