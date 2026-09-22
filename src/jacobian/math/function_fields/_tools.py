"""Finite function-field operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.function_fields._models import (
    FunctionFieldDivisorDegreeResult,
    FunctionFieldDivisorRequest,
    FunctionFieldElementMultiplyRequest,
    FunctionFieldElementMultiplyResult,
    FunctionFieldPlaceValuationRequest,
    FunctionFieldPlaceValuationResult,
    FunctionFieldPrincipalDivisorRequest,
    FunctionFieldPrincipalDivisorResult,
)
from jacobian.math.function_fields.operations import (
    function_field_divisor_degree,
    function_field_element_multiply,
    function_field_place_valuation,
    function_field_principal_divisor,
)


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


_RATIONAL_FIELD = {
    "characteristic": 5,
    "variable": "x",
    "generator": "y",
    "defining_polynomial": [
        {
            "numerator": {"characteristic": 5, "coefficients": [1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        }
    ],
}
_RATIONAL_X = {
    "field": _RATIONAL_FIELD,
    "coordinates": [
        {
            "numerator": {"characteristic": 5, "coefficients": [0, 1]},
            "denominator": {"characteristic": 5, "coefficients": [1]},
        }
    ],
}


def _run_place_valuation(
    request: FunctionFieldPlaceValuationRequest,
) -> FunctionFieldPlaceValuationResult:
    return FunctionFieldPlaceValuationResult(
        place=request.place,
        element=request.element,
        valuation=function_field_place_valuation(request.place, request.element),
    )


def _run_principal(
    request: FunctionFieldPrincipalDivisorRequest,
) -> FunctionFieldPrincipalDivisorResult:
    return function_field_principal_divisor(request.field, request.element)


_GF2_Y = {
    "field": _gf2_elliptic_field(),
    "coordinates": [
        _rational_function([0], [1]),
        _rational_function([1], [1]),
    ],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="function_field.place.valuation.compute",
        title="Compute a function-field valuation",
        description="Compute the exact discrete valuation of a rational-function element at a finite or infinite place; the place and element must share the same rational function field.",
        request_type=FunctionFieldPlaceValuationRequest,
        result_type=FunctionFieldPlaceValuationResult,
        run=_run_place_valuation,
        tags=("function-field", "place", "valuation", "exact"),
        examples=(
            OperationExample(
                name="valuation_of_x_at_zero",
                description="Compute v_(x)(x); the finite place must be the irreducible polynomial x in the rational field GF(5)(x).",
                input={
                    "place": {
                        "field": _RATIONAL_FIELD,
                        "kind": "FINITE",
                        "prime_polynomial": {
                            "characteristic": 5,
                            "coefficients": [0, 1],
                        },
                        "degree": 1,
                    },
                    "element": _RATIONAL_X,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.element.principal_divisor.compute",
        title="Compute a principal divisor",
        description="Compute the complete finite zero/pole divisor of a nonzero rational-function element, retaining every place and exact degree-zero parent identity.",
        request_type=FunctionFieldPrincipalDivisorRequest,
        result_type=FunctionFieldPrincipalDivisorResult,
        run=_run_principal,
        tags=("function-field", "divisor", "exact"),
        examples=(
            OperationExample(
                name="principal_divisor_of_x",
                description="Compute div(x); the element must be nonzero in the rational function field GF(5)(x).",
                input={"field": _RATIONAL_FIELD, "element": _RATIONAL_X},
            ),
        ),
    ),
    MathTool(
        operation_id="function_field.divisor.degree.compute",
        title="Compute a divisor degree",
        description="Compute the exact degree of a finite function-field divisor from its place degrees and multiplicities; every place must belong to its declared field.",
        request_type=FunctionFieldDivisorRequest,
        result_type=FunctionFieldDivisorDegreeResult,
        run=lambda request: function_field_divisor_degree(request.divisor),
        tags=("function-field", "divisor", "degree", "exact"),
        examples=(
            OperationExample(
                name="degree_of_principal_x",
                description="Compute the degree of div(x); every place must retain the rational-field parent.",
                input={
                    "divisor": {
                        "field": _RATIONAL_FIELD,
                        "terms": [
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "FINITE",
                                    "prime_polynomial": {
                                        "characteristic": 5,
                                        "coefficients": [0, 1],
                                    },
                                    "degree": 1,
                                },
                                "multiplicity": 1,
                            },
                            {
                                "place": {
                                    "field": _RATIONAL_FIELD,
                                    "kind": "INFINITE",
                                    "degree": 1,
                                },
                                "multiplicity": -1,
                            },
                        ],
                    }
                },
            ),
        ),
    ),
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
