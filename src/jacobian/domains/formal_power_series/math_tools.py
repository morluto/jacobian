"""MathTool declarations for exact truncated formal power series operations."""

from __future__ import annotations

from jacobian.contracts.formal_power_series import (
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
    _SeriesPairRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.formal_power_series.operations import (
    compute_add,
    compute_compose,
    compute_derivative,
    compute_divide,
    compute_from_polynomial,
    compute_identity_check,
    compute_integral,
    compute_inverse,
    compute_multiply,
    compute_power,
    compute_reversion,
    compute_scalar_multiply,
    compute_subtract,
    compute_to_polynomial,
    compute_truncate,
)
from jacobian.math_tools import MathTool


class _SeriesArithmeticRequest(_SeriesPairRequest):
    """Request for add/subtract/multiply/divide operations."""


class _SeriesIdentityCheckRequest(_SeriesPairRequest):
    """Request for identity check operation."""


FORMAL_POWER_SERIES_OPERATIONS = (
    MathTool(
        operation_id="formal_series.rational.add.compute",
        version="1",
        title="Add two truncated formal power series",
        description=(
            "Compute the exact rational sum of two truncated series in QQ[[x]]/(x^N) "
            "coefficient-wise.  Both operands must share the same variable and "
            "truncation order."
        ),
        request_type=_SeriesArithmeticRequest,
        result_type=SeriesArithmeticResult,
        run=lambda request: compute_add(request.left, request.right),
        tags=(
            "formal-series",
            "power-series",
            "arithmetic",
            "addition",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "add_one_plus_x",
                "Add (1+x) + (1+2x) = 2+3x at order 2.",
                {
                    "left": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "right": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.subtract.compute",
        version="1",
        title="Subtract two truncated formal power series",
        description=(
            "Compute the exact rational difference of two truncated series in "
            "QQ[[x]]/(x^N) coefficient-wise.  Both operands must share the same "
            "variable and truncation order."
        ),
        request_type=_SeriesArithmeticRequest,
        result_type=SeriesArithmeticResult,
        run=lambda request: compute_subtract(request.left, request.right),
        tags=(
            "formal-series",
            "power-series",
            "arithmetic",
            "subtraction",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "subtract_1_plus_x",
                "Subtract (1+x) - (2+x) = -1 at order 2.",
                {
                    "left": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "right": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.multiply.compute",
        version="1",
        title="Multiply two truncated formal power series",
        description=(
            "Compute the exact Cauchy convolution of two truncated series in "
            "QQ[[x]]/(x^N).  Both operands must share the same variable and "
            "truncation order."
        ),
        request_type=_SeriesArithmeticRequest,
        result_type=SeriesMultiplyResult,
        run=lambda request: compute_multiply(request.left, request.right),
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
        version="1",
        title="Multiply a series by a rational scalar",
        description=(
            "Multiply a truncated formal power series by an exact rational scalar. "
        ),
        request_type=SeriesScalarMultiplyRequest,
        result_type=SeriesScalarMultiplyResult,
        run=lambda request: compute_scalar_multiply(request.series, request.scalar),
        tags=(
            "formal-series",
            "power-series",
            "arithmetic",
            "scalar",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "scalar_multiply_3",
                "Multiply (1+x) by 3 = 3+3x at order 2.",
                {
                    "series": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "scalar": {"num": "3", "den": "1"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.power.compute",
        version="1",
        title="Raise a truncated formal power series to a nonnegative integer power",
        description=(
            "Compute the exact power of a truncated series in QQ[[x]]/(x^N) via "
            "binary exponentiation."
        ),
        request_type=SeriesPowerRequest,
        result_type=SeriesPowerResult,
        run=lambda request: compute_power(request.series, request.exponent),
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
        version="1",
        title="Invert a truncated formal power series",
        description=(
            "Compute the multiplicative inverse B(x) of A(x) modulo x^N, requiring "
            "a_0 != 0.  Returns the exact product residual A*B - 1."
        ),
        request_type=SeriesInverseRequest,
        result_type=SeriesInverseResult,
        run=lambda request: compute_inverse(request.as_series()),
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
        version="1",
        title="Divide two truncated formal power series",
        description=(
            "Compute the exact quotient Q = A/B modulo x^N, requiring b_0 != 0. "
            "Returns the exact residual B*Q - A."
        ),
        request_type=SeriesDivideRequest,
        result_type=SeriesDivideResult,
        run=lambda request: compute_divide(request.left, request.right),
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
        version="1",
        title="Compose two truncated formal power series",
        description=(
            "Compute the composition F(G(x)) mod x^N.  The inner series G must "
            "have zero constant term."
        ),
        request_type=SeriesComposeRequest,
        result_type=SeriesComposeResult,
        run=lambda request: compute_compose(request.outer, request.inner),
        tags=(
            "formal-series",
            "power-series",
            "composition",
            "rational",
            "truncated",
            "exact",
        ),
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
        version="1",
        title="Compositional inverse of a truncated formal power series",
        description=(
            "Compute the compositional inverse G(x) of F(x) mod x^N, requiring "
            "F(0)=0 and f_1 != 0.  Validates both left and right identities "
            "exactly."
        ),
        request_type=SeriesReversionRequest,
        result_type=SeriesReversionResult,
        run=lambda request: compute_reversion(request.as_series()),
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
        version="1",
        title="Formal derivative of a truncated power series",
        description=(
            "Compute the formal derivative of a truncated series in QQ[[x]]/(x^N). "
            "Output order convention: max(N-1, 1)."
        ),
        request_type=InputTruncatedSeries,
        result_type=SeriesDerivativeResult,
        run=lambda request: compute_derivative(request),
        tags=(
            "formal-series",
            "power-series",
            "derivative",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "derivative_of_1_plus_2x_plus_3x_squared",
                "Derivative of (1+2x+3x^2) at order 3: 2+6x at order 2.",
                {
                    "variable": "x",
                    "truncation_order": 3,
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.integral_zero_constant.compute",
        version="1",
        title="Zero-constant formal integral of a truncated power series",
        description=(
            "Compute the unique formal antiderivative B'(x)=A(x), B(0)=0 of a "
            "truncated series."
        ),
        request_type=SeriesIntegralRequest,
        result_type=SeriesIntegralResult,
        run=lambda request: compute_integral(request.series, request.output_order),
        tags=(
            "formal-series",
            "power-series",
            "integral",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "integral_of_1_plus_2x",
                "Integral of (1+2x) at order 3: x+x^2 at order 3.",
                {
                    "series": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                    "output_order": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.truncate.compute",
        version="1",
        title="Truncate a formal power series to a smaller order",
        description=(
            "Return the same coefficients through M-1 at order M, where M <= N."
        ),
        request_type=SeriesTruncateRequest,
        result_type=SeriesTruncateResult,
        run=lambda request: compute_truncate(request.series, request.target_order),
        tags=(
            "formal-series",
            "power-series",
            "truncate",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "truncate_to_2",
                "Truncate (1+x+x^2) at order 2: (1+x).",
                {
                    "series": {
                        "variable": "x",
                        "truncation_order": 3,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "target_order": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.identity.check",
        version="1",
        title="Check whether two truncated formal power series are identical",
        description=(
            "Check if two series are equal mod x^N, returning either EQUAL_MOD_X_TO_N "
            "or the first differing index with the exact difference."
        ),
        request_type=_SeriesIdentityCheckRequest,
        result_type=SeriesIdentityCheckResult,
        run=lambda request: compute_identity_check(request.left, request.right),
        tags=(
            "formal-series",
            "power-series",
            "identity",
            "check",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "identity_1_plus_x",
                "Check (1+x) = (1+x): EQUAL.",
                {
                    "left": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "right": {
                        "variable": "x",
                        "truncation_order": 2,
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.from_polynomial.compute",
        version="1",
        title="Convert a dense rational polynomial into a truncated series",
        description=(
            "Project a dense rational coefficient tuple onto a series value with "
            "explicit truncation order."
        ),
        request_type=SeriesFromPolynomialRequest,
        result_type=SeriesFromPolynomialResult,
        run=lambda request: compute_from_polynomial(
            request.variable, request.coefficients, request.truncation_order
        ),
        tags=(
            "formal-series",
            "power-series",
            "polynomial",
            "conversion",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "from_polynomial_1_plus_x",
                "Convert (1+x) into a series at order 2.",
                {
                    "variable": "x",
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "truncation_order": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="formal_series.rational.to_polynomial.compute",
        version="1",
        title="Return the canonical truncated polynomial representative",
        description=(
            "Return the canonical truncated polynomial containing exactly the "
            "known coefficients below x^N."
        ),
        request_type=InputTruncatedSeries,
        result_type=SeriesToPolynomialResult,
        run=lambda request: compute_to_polynomial(request),
        tags=(
            "formal-series",
            "power-series",
            "polynomial",
            "conversion",
            "rational",
            "truncated",
            "exact",
        ),
        examples=(
            example(
                "to_polynomial_1_plus_x",
                "Return the truncated polynomial representative of (1+x).",
                {
                    "variable": "x",
                    "truncation_order": 2,
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)
