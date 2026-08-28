"""MathTool declarations for exact truncated formal power series operations."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.polynomials.series._models import (
    InputTruncatedSeries,
    SeriesArithmeticResult,
    SeriesComposeRequest,
    SeriesComposeResult,
    SeriesDerivativeResult,
    SeriesDivideRequest,
    SeriesDivideResult,
    SeriesFromPolynomialRequest,
    SeriesFromPolynomialResult,
    SeriesIdentityCheckResult,
    SeriesIntegralRequest,
    SeriesIntegralResult,
    SeriesInverseRequest,
    SeriesInverseResult,
    SeriesMultiplyResult,
    SeriesPowerRequest,
    SeriesPowerResult,
    SeriesReversionRequest,
    SeriesReversionResult,
    SeriesScalarMultiplyRequest,
    SeriesScalarMultiplyResult,
    SeriesToPolynomialResult,
    SeriesTruncateRequest,
    SeriesTruncateResult,
    _SeriesAddSubtractRequest,
    _SeriesIdentityCheckRequest,
    _SeriesMultiplyRequest,
)
from jacobian.math.polynomials.series.operations import (
    add,
    compose,
    derivative,
    divide,
    from_polynomial,
    identity_check,
    integral_zero_constant,
    inverse,
    multiply,
    power,
    reversion,
    scalar_multiply,
    subtract,
    to_polynomial,
    truncate,
)

_ZERO = {"num": "0", "den": "1"}
_ONE = {"num": "1", "den": "1"}
_TWO = {"num": "2", "den": "1"}
_ONE_PLUS_X = {
    "variable": "x",
    "truncation_order": 3,
    "coefficients": [_ONE, _ONE, _ZERO],
}
_X = {
    "variable": "x",
    "truncation_order": 3,
    "coefficients": [_ZERO, _ONE, _ZERO],
}


def _input_series_from_request(
    request: SeriesInverseRequest | SeriesReversionRequest,
) -> InputTruncatedSeries:
    try:
        return request.as_series()
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("coefficients",),
            code="formal_series.coefficient_count_mismatch",
            message="coefficient count must equal truncation_order",
        ) from exc


TOOLS = (
    MathTool(
        operation_id="formal_series.rational.add.compute",
        title="Add two truncated formal power series",
        description=(
            "Compute the exact coefficientwise sum of two truncated rational "
            "formal power series with the same variable and order."
        ),
        request_type=_SeriesAddSubtractRequest,
        result_type=SeriesArithmeticResult,
        run=lambda request: add(request.left, request.right),
        tags=("formal-series", "arithmetic", "addition", "rational", "exact"),
        examples=(
            example(
                "add_one_plus_x_and_x",
                "Add 1+x and x modulo x^3.",
                {"left": _ONE_PLUS_X, "right": _X},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.subtract.compute",
        title="Subtract two truncated formal power series",
        description=(
            "Compute the exact coefficientwise difference of two truncated "
            "rational formal power series with the same variable and order."
        ),
        request_type=_SeriesAddSubtractRequest,
        result_type=SeriesArithmeticResult,
        run=lambda request: subtract(request.left, request.right),
        tags=("formal-series", "arithmetic", "subtraction", "rational", "exact"),
        examples=(
            example(
                "subtract_x",
                "Subtract x from 1+x modulo x^3.",
                {"left": _ONE_PLUS_X, "right": _X},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.multiply.compute",
        title="Multiply two truncated formal power series",
        description=(
            "Compute the exact Cauchy convolution of two truncated series in "
            "QQ[[x]]/(x^N).  Both operands must share the same variable and "
            "truncation order."
        ),
        request_type=_SeriesMultiplyRequest,
        result_type=SeriesMultiplyResult,
        run=lambda request: multiply(request.left, request.right),
        tags=(
            "formal-series",
            "power-series",
            "arithmetic",
            "multiplication",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "multiply_1_plus_x",
                "Multiply (1+x) * (1+x) = 1+2x+x^2 at order 3.",
                {
                    "left": {
                        "variable": "x",
                        "truncation_order": 3,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "right": {
                        "variable": "x",
                        "truncation_order": 3,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.scalar_multiply.compute",
        title="Multiply a truncated formal power series by a rational scalar",
        description=(
            "Multiply a truncated formal power series by an exact rational "
            "scalar coefficientwise."
        ),
        request_type=SeriesScalarMultiplyRequest,
        result_type=SeriesScalarMultiplyResult,
        run=lambda request: scalar_multiply(request.series, request.scalar),
        tags=("formal-series", "arithmetic", "scalar", "rational", "exact"),
        examples=(
            example(
                "double_one_plus_x",
                "Multiply 1+x by two modulo x^3.",
                {"series": _ONE_PLUS_X, "scalar": _TWO},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.power.compute",
        title="Raise a truncated formal power series to a nonnegative integer power",
        description=(
            "Compute the exact power of a truncated series in QQ[[x]]/(x^N) via "
            "binary exponentiation."
        ),
        request_type=SeriesPowerRequest,
        result_type=SeriesPowerResult,
        run=lambda request: power(request.series, request.exponent),
        tags=(
            "formal-series",
            "power-series",
            "arithmetic",
            "power",
            "exponentiation",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "power_3_of_1_plus_x",
                "Compute (1+x)^3 at order 4.",
                {
                    "series": {
                        "variable": "x",
                        "truncation_order": 4,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "exponent": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.inverse.compute",
        title="Invert a truncated formal power series",
        description=(
            "Compute the multiplicative inverse B(x) of A(x) modulo x^N, requiring "
            "a_0 != 0.  Returns the exact product residual A*B - 1."
        ),
        request_type=SeriesInverseRequest,
        result_type=SeriesInverseResult,
        run=lambda request: inverse(_input_series_from_request(request)),
        tags=(
            "formal-series",
            "power-series",
            "inverse",
            "unit",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "inverse_1_plus_x",
                "Invert (1+x) at order 4: 1-x+x^2-x^3.",
                {
                    "variable": "x",
                    "truncation_order": 4,
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.divide.compute",
        title="Divide two truncated formal power series",
        description=(
            "Compute the exact quotient Q = A/B modulo x^N, requiring b_0 != 0. "
            "Returns the exact residual B*Q - A."
        ),
        request_type=SeriesDivideRequest,
        result_type=SeriesDivideResult,
        run=lambda request: divide(request.left, request.right),
        tags=(
            "formal-series",
            "power-series",
            "division",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "divide_1_by_1_minus_x",
                "Divide 1 by (1-x) at order 4: 1+x+x^2+x^3.",
                {
                    "left": {
                        "variable": "x",
                        "truncation_order": 4,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "right": {
                        "variable": "x",
                        "truncation_order": 4,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "-1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.compose.compute",
        title="Compose two truncated formal power series",
        description=(
            "Compute the composition F(G(x)) mod x^N.  The inner series G must "
            "have zero constant term."
        ),
        request_type=SeriesComposeRequest,
        result_type=SeriesComposeResult,
        run=lambda request: compose(request.outer, request.inner),
        tags=(
            "formal-series",
            "power-series",
            "composition",
            "rational",
            "truncated",
            "exact",
        ),
        discovery_terms=("series",),
        examples=(
            example(
                "compose_x_with_x_squared",
                "Compose (1+x) with (x^2) at order 4: 1+x^2.",
                {
                    "outer": {
                        "variable": "x",
                        "truncation_order": 4,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "inner": {
                        "variable": "x",
                        "truncation_order": 4,
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.reversion.compute",
        title="Compositional inverse of a truncated formal power series",
        description=(
            "Compute the compositional inverse G(x) of F(x) mod x^N, requiring "
            "F(0)=0 and f_1 != 0.  Validates both left and right identities "
            "exactly."
        ),
        request_type=SeriesReversionRequest,
        result_type=SeriesReversionResult,
        run=lambda request: reversion(_input_series_from_request(request)),
        tags=(
            "formal-series",
            "power-series",
            "reversion",
            "compositional-inverse",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "reversion_of_2x",
                "Reversion of (2x) at order 4: (1/2)x.",
                {
                    "variable": "x",
                    "truncation_order": 4,
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.derivative.compute",
        title="Differentiate a truncated formal power series",
        description=(
            "Compute the exact formal derivative of a truncated rational power "
            "series, retaining the documented output-order convention."
        ),
        request_type=InputTruncatedSeries,
        result_type=SeriesDerivativeResult,
        run=derivative,
        tags=("formal-series", "calculus", "derivative", "rational", "exact"),
        examples=(
            example(
                "differentiate_one_plus_x",
                "Differentiate 1+x modulo x^3.",
                _ONE_PLUS_X,
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.integral_zero_constant.compute",
        title="Integrate a truncated formal power series",
        description=(
            "Compute the unique exact formal antiderivative with zero constant "
            "term at the requested output order."
        ),
        request_type=SeriesIntegralRequest,
        result_type=SeriesIntegralResult,
        run=lambda request: integral_zero_constant(
            request.series, request.output_order
        ),
        tags=("formal-series", "calculus", "integral", "rational", "exact"),
        examples=(
            example(
                "integrate_x",
                "Integrate x with zero constant term through order three.",
                {"series": _X, "output_order": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.truncate.compute",
        title="Truncate a formal power series",
        description=(
            "Return the exact coefficient prefix of a truncated formal power "
            "series at a smaller requested order."
        ),
        request_type=SeriesTruncateRequest,
        result_type=SeriesTruncateResult,
        run=lambda request: truncate(request.series, request.target_order),
        tags=("formal-series", "truncation", "rational", "exact"),
        examples=(
            example(
                "truncate_one_plus_x",
                "Truncate 1+x modulo x^3 to order two.",
                {"series": _ONE_PLUS_X, "target_order": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.identity.check",
        title="Check identity of truncated formal power series",
        description=(
            "Check exact equality modulo x^N and report the first differing "
            "coefficient when the series are not identical."
        ),
        request_type=_SeriesIdentityCheckRequest,
        result_type=SeriesIdentityCheckResult,
        run=lambda request: identity_check(request.left, request.right),
        tags=("formal-series", "identity", "rational", "exact"),
        examples=(
            example(
                "identity_of_one_plus_x",
                "Check 1+x against itself modulo x^3.",
                {"left": _ONE_PLUS_X, "right": _ONE_PLUS_X},
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.from_polynomial.compute",
        title="Convert a rational polynomial to a formal power series",
        description=(
            "Convert a dense canonical rational polynomial coefficient sequence "
            "to an exact truncated formal power series."
        ),
        request_type=SeriesFromPolynomialRequest,
        result_type=SeriesFromPolynomialResult,
        run=lambda request: from_polynomial(
            request.variable, request.coefficients, request.truncation_order
        ),
        tags=("formal-series", "polynomial", "conversion", "rational", "exact"),
        examples=(
            example(
                "polynomial_one_plus_x",
                "Convert the dense polynomial 1+x to a series modulo x^3.",
                {
                    "variable": "x",
                    "coefficients": [_ONE, _ONE, _ZERO],
                    "truncation_order": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.to_polynomial.compute",
        title="Convert a formal power series to its polynomial representative",
        description=(
            "Return the exact dense canonical polynomial representative of the "
            "known coefficients below the truncation order."
        ),
        request_type=InputTruncatedSeries,
        result_type=SeriesToPolynomialResult,
        run=to_polynomial,
        tags=("formal-series", "polynomial", "conversion", "rational", "exact"),
        examples=(
            example(
                "series_one_plus_x",
                "Return the polynomial representative of 1+x modulo x^3.",
                _ONE_PLUS_X,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
