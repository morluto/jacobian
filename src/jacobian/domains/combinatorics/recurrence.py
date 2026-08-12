"""Recurrence-owned exact combinatorics capabilities."""

from jacobian.contracts.combinatorics import (
    FibonacciPairRequest,
    FibonacciPairResult,
    IntegerResult,
    LinearRecurrenceEvaluationRequest,
    LinearRecurrenceEvaluationResult,
    NonnegativeIntegerRequest,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
    RationalResult,
)
from jacobian.domains._examples import example
from jacobian.domains.combinatorics._support import (
    combinatorics_operation,
)
from jacobian.domains.combinatorics.operations import (
    bernoulli,
    fibonacci,
    fibonacci_pair,
    lucas,
)
from jacobian.domains.combinatorics.recurrence_series_operations import (
    compute_rational_generating_function_coefficients,
    evaluate_linear_recurrence,
    evaluate_polynomial_coefficient_recurrence,
)

RECURRENCE_CAPABILITIES = (
    combinatorics_operation(
        "combinatorics.compute.fibonacci",
        "Compute Fibonacci number",
        "Compute the nth Fibonacci number exactly.",
        NonnegativeIntegerRequest,
        IntegerResult,
        fibonacci,
        "combinatorics",
        "sequence",
        invocation_examples=(
            example("fibonacci_10", "Compute the tenth Fibonacci number.", {"n": 10}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.fibonacci_pair",
        "Compute consecutive Fibonacci values",
        "Return F_n and F_(n+1) as one exact recurrence boundary.",
        FibonacciPairRequest,
        FibonacciPairResult,
        fibonacci_pair,
        "combinatorics",
        "fibonacci",
        "recurrence-boundary",
        invocation_examples=(
            example(
                "fibonacci_pair_8",
                "Return consecutive Fibonacci values at n=8.",
                {"n": 8},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.lucas",
        "Compute Lucas number",
        "Compute the nth Lucas number exactly.",
        NonnegativeIntegerRequest,
        IntegerResult,
        lucas,
        "combinatorics",
        "sequence",
        invocation_examples=(
            example("lucas_7", "Compute the seventh Lucas number.", {"n": 7}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.compute.bernoulli",
        "Compute Bernoulli number",
        "Compute the nth Bernoulli number as a reduced rational.",
        NonnegativeIntegerRequest,
        RationalResult,
        bernoulli,
        "combinatorics",
        "sequence",
        invocation_examples=(
            example("bernoulli_4", "Compute the fourth Bernoulli number.", {"n": 4}),
        ),
    ),
    combinatorics_operation(
        "combinatorics.recurrence.linear.evaluate",
        "Evaluate an exact linear recurrence",
        (
            "Evaluate requested terms of one bounded constant-coefficient rational "
            "recurrence and preserve the complete replay prefix through the "
            "greatest requested index."
        ),
        LinearRecurrenceEvaluationRequest,
        LinearRecurrenceEvaluationResult,
        evaluate_linear_recurrence,
        "combinatorics",
        "recurrence",
        "linear-recurrence",
        "exact-rational",
        invocation_examples=(
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
            "Evaluate a bounded rational recurrence sum p_j(n)a_(n-j)=0, "
            "preserving the complete replay prefix and exact residuals."
        ),
        PolynomialCoefficientRecurrenceEvaluationRequest,
        PolynomialCoefficientRecurrenceEvaluationResult,
        evaluate_polynomial_coefficient_recurrence,
        "combinatorics",
        "recurrence",
        "p-recursive",
        "polynomial-coefficients",
        "exact-rational",
        invocation_examples=(
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
        "combinatorics.generating_function.coefficients.compute",
        "Compute a rational generating-function coefficient prefix",
        (
            "Expand one exact rational function N(x)/D(x) at zero through a "
            "bounded finite truncation and expose the residual congruence."
        ),
        RationalGeneratingFunctionCoefficientsRequest,
        RationalGeneratingFunctionCoefficientsResult,
        compute_rational_generating_function_coefficients,
        "combinatorics",
        "generating-function",
        "rational-series",
        "exact-rational",
        invocation_examples=(
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
