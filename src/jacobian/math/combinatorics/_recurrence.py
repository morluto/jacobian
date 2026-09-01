"""Recurrence-owned exact combinatorics operations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics._recurrence_models import (
    LinearRecurrenceEvaluationRequest,
    LinearRecurrenceEvaluationResult,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
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
    MathTool(
        operation_id="combinatorics.recurrence.linear.evaluate",
        title="Evaluate an exact linear recurrence",
        description=(
            "Evaluate requested terms of one bounded constant-coefficient rational "
            "recurrence."
        ),
        request_type=LinearRecurrenceEvaluationRequest,
        result_type=LinearRecurrenceEvaluationResult,
        run=_run_linear_recurrence,
        tags=("combinatorics", "recurrence", "linear-recurrence", "exact-rational"),
        examples=(
            OperationExample(
                name="generic_fibonacci_prefix",
                description="Evaluate the first eight terms of the Fibonacci recurrence.",
                input={
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
    MathTool(
        operation_id="combinatorics.recurrence.p_recursive.evaluate",
        title="Evaluate an exact polynomial-coefficient recurrence",
        description=(
            "Evaluate requested terms of a bounded rational recurrence "
            "sum p_j(n)a_(n-j)=0."
        ),
        request_type=PolynomialCoefficientRecurrenceEvaluationRequest,
        result_type=PolynomialCoefficientRecurrenceEvaluationResult,
        run=_run_polynomial_coefficient_recurrence,
        tags=(
            "combinatorics",
            "recurrence",
            "sequence",
            "polynomial",
            "p-recursive",
            "polynomial-coefficients",
            "exact-rational",
        ),
        examples=(
            OperationExample(
                name="factorial_prefix",
                description="Evaluate the first seven terms of a_n=n*a_(n-1).",
                input={
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
    MathTool(
        operation_id="combinatorics.recurrence.p_recursive.table_residuals.compute",
        title="Compute residuals for a submitted P-recursive table",
        description=(
            "Compute every exact residual of a bounded caller-supplied rational "
            "table against sum p_j(n)a_(n-j)=0 without generating or repairing terms."
        ),
        request_type=PolynomialCoefficientRecurrenceTableRequest,
        result_type=PolynomialCoefficientRecurrenceTableResult,
        run=_compute_recurrence_table_residuals,
        tags=(
            "combinatorics",
            "recurrence",
            "p-recursive",
            "submitted-table",
            "exact-rational",
        ),
        examples=(
            OperationExample(
                name="factorial_table_residuals",
                description="Check a supplied factorial prefix against a_n=n*a_(n-1).",
                input={
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
    MathTool(
        operation_id="combinatorics.generating_function.coefficients.compute",
        title="Compute a rational generating-function coefficient prefix",
        description=(
            "Expand one exact rational function N(x)/D(x) at zero through a "
            "bounded finite truncation and expose the residual congruence."
        ),
        request_type=RationalGeneratingFunctionCoefficientsRequest,
        result_type=RationalGeneratingFunctionCoefficientsResult,
        run=_run_rational_generating_function_coefficients,
        tags=(
            "combinatorics",
            "generating-function",
            "rational-series",
            "exact-rational",
        ),
        examples=(
            OperationExample(
                name="geometric_series_prefix",
                description="Expand 1/(1-x) through six coefficients.",
                input={
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
