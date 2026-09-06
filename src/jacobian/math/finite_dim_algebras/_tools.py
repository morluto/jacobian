"""Finite-dimensional algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    CenterResult,
)
from jacobian.math.finite_dim_algebras.operations import center_basis
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def compute_center(request: CenterRequest) -> CenterResult:
    basis = center_basis(request.algebra)
    return CenterResult(
        algebra=request.algebra,
        basis_matrix=PrimeFieldMatrix(
            prime=request.algebra.field_order,
            entries=basis,
            columns=request.algebra.dimension,
        ),
    )


_ZERO_ALG_2 = {
    "dimension": 2,
    "field_order": 2,
    "multiplication": [
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebra.center.compute",
        title="Compute the center of a finite-dimensional algebra",
        description="Compute the center {z : z*a = a*z for all a} of a finite-dimensional "
        "algebra given by structure constants over a prime field.",
        request_type=CenterRequest,
        result_type=CenterResult,
        run=compute_center,
        tags=("algebra", "center", "exact"),
        examples=(
            OperationExample(
                name="zero_algebra",
                description="Center of the 2-dimensional zero algebra over F_2.",
                input={"algebra": _ZERO_ALG_2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
