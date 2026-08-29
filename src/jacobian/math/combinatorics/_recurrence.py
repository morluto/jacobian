"""Recurrence-owned exact combinatorics operations."""

from jacobian.catalog._examples import example
from jacobian.math.combinatorics._recurrence_models import (
    LinearRecurrenceEvaluationRequest,
    LinearRecurrenceEvaluationResult,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.math.combinatorics._support import (
    combinatorics_operation,
)
from jacobian.math.combinatorics.operations import (
    evaluate_linear_recurrence,
    evaluate_polynomial_coefficient_recurrence,
    rational_generating_function_coefficients,
)
from jacobian.math.combinatorics.recurrence_tables import (
    PolynomialCoefficientRecurrenceTableRequest,
    PolynomialCoefficientRecurrenceTableResult,
    _compute_recurrence_table_residuals,
)


def _run_linear_recurrence(
    request: LinearRecurrenceEvaluationRequest,
) -> LinearRecurrenceEvaluationResult:
    return evaluate_linear_recurrence(
        request.coefficients,
        request.initial_values,
        request.coefficient_convention,
        request.scope,
        request.term_count,
        request.indices,
    )


def _run_polynomial_coefficient_recurrence(
    request: PolynomialCoefficientRecurrenceEvaluationRequest,
) -> PolynomialCoefficientRecurrenceEvaluationResult:
    return evaluate_polynomial_coefficient_recurrence(
        request.coefficient_polynomials,
        request.initial_values,
        request.coefficient_convention,
        request.polynomial_convention,
        request.scope,
        request.term_count,
        request.indices,
    )


def _run_rational_generating_function_coefficients(
    request: RationalGeneratingFunctionCoefficientsRequest,
) -> RationalGeneratingFunctionCoefficientsResult:
    return rational_generating_function_coefficients(
        request.numerator,
        request.denominator,
        request.coefficient_convention,
        request.expansion_point,
        request.truncation_order,
    )


RECURRENCE_OPERATIONS = (
    combinatorics_operation(
        "combinatorics.recurrence.linear.evaluate",
        "Evaluate an exact linear recurrence",
        (
            "Evaluate requested terms of one bounded constant-coefficient rational "
            "recurrence."
        ),
        LinearRecurrenceEvaluationRequest,
        LinearRecurrenceEvaluationResult,
        _run_linear_recurrence,
        "combinatorics",
        "recurrence",
        "linear-recurrence",
        "exact-rational",
        examples=(
            example(
                "generic_fibonacci_prefix",
                "Evaluate the first eight terms of the Fibonacci recurrence.",
                {
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "initial_values": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "coefficient_convention": (
                        "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
                    ),
                    "scope": "PREFIX",
                    "term_count": 8,
                    "indices": [],
                },
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.recurrence.p_recursive.evaluate",
        "Evaluate an exact polynomial-coefficient recurrence",
        (
            "Evaluate requested terms of a bounded rational recurrence "
            "sum p_j(n)a_(n-j)=0."
        ),
        PolynomialCoefficientRecurrenceEvaluationRequest,
        PolynomialCoefficientRecurrenceEvaluationResult,
        _run_polynomial_coefficient_recurrence,
        "combinatorics",
        "recurrence",
        "sequence",
        "polynomial",
        "p-recursive",
        "polynomial-coefficients",
        "exact-rational",
        examples=(
            example(
                "factorial_prefix",
                "Evaluate the first seven terms of a_n=n*a_(n-1).",
                {
                    "coefficient_polynomials": [
                        [{"num": "1", "den": "1"}],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                    ],
                    "initial_values": [{"num": "1", "den": "1"}],
                    "coefficient_convention": (
                        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
                    ),
                    "polynomial_convention": "ASCENDING_POWERS_OF_N",
                    "scope": "PREFIX",
                    "term_count": 7,
                    "indices": [],
                },
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.recurrence.p_recursive.table_residuals.compute",
        "Compute residuals for a submitted P-recursive table",
        (
            "Compute every exact residual of a bounded caller-supplied rational "
            "table against sum p_j(n)a_(n-j)=0 without generating or repairing terms."
        ),
        PolynomialCoefficientRecurrenceTableRequest,
        PolynomialCoefficientRecurrenceTableResult,
        _compute_recurrence_table_residuals,
        "combinatorics",
        "recurrence",
        "p-recursive",
        "submitted-table",
        "exact-rational",
        examples=(
            example(
                "factorial_table_residuals",
                "Check a supplied factorial prefix against a_n=n*a_(n-1).",
                {
                    "coefficient_polynomials": [
                        [{"num": "1", "den": "1"}],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                    ],
                    "values": [
                        {"num": value, "den": "1"}
                        for value in ("1", "1", "2", "6", "24", "120")
                    ],
                    "coefficient_convention": (
                        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
                    ),
                    "polynomial_convention": "ASCENDING_POWERS_OF_N",
                    "table_convention": "VALUES_A_0_THROUGH_A_N_IN_ORDER",
                },
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.generating_function.coefficients.compute",
        "Compute a rational generating-function coefficient prefix",
        (
            "Expand one exact rational function N(x)/D(x) at zero through a "
            "bounded finite truncation and expose the residual congruence."
        ),
        RationalGeneratingFunctionCoefficientsRequest,
        RationalGeneratingFunctionCoefficientsResult,
        _run_rational_generating_function_coefficients,
        "combinatorics",
        "generating-function",
        "rational-series",
        "exact-rational",
        examples=(
            example(
                "geometric_series_prefix",
                "Expand 1/(1-x) through six coefficients.",
                {
                    "numerator": [{"num": "1", "den": "1"}],
                    "denominator": [
                        {"num": "1", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ],
                    "coefficient_convention": "ASCENDING_POWERS_OF_X",
                    "expansion_point": "0",
                    "truncation_order": 6,
                },
            ),
        ),
    ),
)
