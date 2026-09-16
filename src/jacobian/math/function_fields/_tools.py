"""Finite function-field operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.function_fields._models import (
    FunctionFieldElementMultiplyRequest,
    FunctionFieldElementMultiplyResult,
)
from jacobian.math.function_fields.operations import function_field_element_multiply


def _run_element_multiply(
    request: FunctionFieldElementMultiplyRequest,
) -> FunctionFieldElementMultiplyResult:
    return function_field_element_multiply(request.left, request.right)


def _rational_function(numerator: list[int], denominator: list[int]) -> dict[str, Any]:
    return {
        "numerator": {"characteristic": 2, "coefficients": numerator},
        "denominator": {"characteristic": 2, "coefficients": denominator},
    }


def _gf2_elliptic_field() -> dict[str, Any]:
    return {
        "characteristic": 2,
        "variable": "x",
        "generator": "y",
        "defining_polynomial": [
            _rational_function([0, 1], [1]),
            _rational_function([1], [1]),
            _rational_function([1], [1]),
        ],
    }


_GF2_Y = {
    "field": _gf2_elliptic_field(),
    "coordinates": [
        _rational_function([0], [1]),
        _rational_function([1], [1]),
    ],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="function_field.element.multiply.compute",
        title="Multiply two elements of a finite function field",
        description=(
            "Multiply two reduced elements of GF(p)(x)[y]/(f) by exact "
            "polynomial multiplication in the generator followed by reduction "
            "modulo the monic separable defining polynomial using exact "
            "GF(p)(x) arithmetic. Return the canonical reduced product and the "
            "complete raw-product and reduction ledger. Both elements must be "
            "bound to the identical function field; reducible or inseparable "
            "defining polynomials are rejected."
        ),
        request_type=FunctionFieldElementMultiplyRequest,
        result_type=FunctionFieldElementMultiplyResult,
        run=_run_element_multiply,
        tags=("algebra", "function-field", "function-field-element", "exact"),
        discovery_terms=(
            "function field multiplication",
            "algebraic function field element product",
            "GF(p)(x)[y] reduction",
            "rational function field extension arithmetic",
        ),
        examples=(
            OperationExample(
                name="gf2_y_squared",
                description=(
                    "Multiply the generator y by itself in GF(2)(x)[y]/(y^2+y+x); "
                    "the field must be a monic separable extension and both "
                    "elements must share it. The reduction y^2 = y+x is returned "
                    "in the ledger."
                ),
                input={"left": _GF2_Y, "right": _GF2_Y},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
