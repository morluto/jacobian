"""Declarations for bounded integer-interval arithmetic-function profiles."""

from typing import NoReturn

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory._interval_profile_models import (
    DivisorCountProfileRequest,
    DivisorCountProfileResult,
    DivisorSumProfileRequest,
    DivisorSumProfileResult,
    EulerTotientProfileRequest,
    EulerTotientProfileResult,
    GreatestPrimeFactorProfileRequest,
    GreatestPrimeFactorProfileResult,
    LeastPrimeFactorProfileRequest,
    LeastPrimeFactorProfileResult,
    PrimeGapProfileRequest,
    PrimeGapProfileResult,
    SquarefreeProfileRequest,
    SquarefreeProfileResult,
)
from jacobian.math.number_theory.interval_profiles.operations import (
    IntervalAdmissionError,
    divisor_count_profile,
    divisor_sum_profile,
    euler_totient_profile,
    greatest_prime_factor_profile,
    least_prime_factor_profile,
    prime_gap_profile,
    squarefree_profile,
)


def _raise_interval_admission(exc: IntervalAdmissionError) -> NoReturn:
    """Translate native interval admission into catalog's field error type."""
    raise OperationDomainValidationError(
        location=("upper_bound",),
        code=f"number_theory.interval.{exc.reason}",
        message=str(exc),
    ) from exc


def compute_squarefree_profile(
    request: SquarefreeProfileRequest,
) -> SquarefreeProfileResult:
    """Project a squarefree request into the canonical native operation."""
    try:
        return squarefree_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_divisor_count_profile(
    request: DivisorCountProfileRequest,
) -> DivisorCountProfileResult:
    """Project a divisor-count request into the canonical native operation."""
    try:
        return divisor_count_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_greatest_prime_factor_profile(
    request: GreatestPrimeFactorProfileRequest,
) -> GreatestPrimeFactorProfileResult:
    """Project a greatest-prime-factor request into the canonical operation."""
    try:
        return greatest_prime_factor_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_prime_gap_profile(request: PrimeGapProfileRequest) -> PrimeGapProfileResult:
    """Project a prime-gap request into the canonical native operation."""
    try:
        return prime_gap_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_least_prime_factor_profile(
    request: LeastPrimeFactorProfileRequest,
) -> LeastPrimeFactorProfileResult:
    """Project a least-prime-factor request into the canonical operation."""
    try:
        return least_prime_factor_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_euler_totient_profile(
    request: EulerTotientProfileRequest,
) -> EulerTotientProfileResult:
    """Project a totient request into the canonical native operation."""
    try:
        return euler_totient_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


def compute_divisor_sum_profile(
    request: DivisorSumProfileRequest,
) -> DivisorSumProfileResult:
    """Project a divisor-sum request into the canonical native operation."""
    try:
        return divisor_sum_profile(request.lower_bound, request.upper_bound)
    except IntervalAdmissionError as exc:
        _raise_interval_admission(exc)


INTERVAL_PROFILE_OPERATIONS = (
    MathTool(
        operation_id="number_theory.integer_interval.squarefree_profile.compute",
        title="Compute squarefree profile on a bounded interval",
        description=(
            "Partition a closed positive integer interval [L, U] into its "
            "exact squarefree and non-squarefree members, retaining ordered "
            "lists and counts for both classes."
        ),
        request_type=SquarefreeProfileRequest,
        result_type=SquarefreeProfileResult,
        run=compute_squarefree_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="squarefree_interval_1_to_12",
                description=(
                    "Partition [1, 12] into squarefree and non-squarefree "
                    "integers; coupled width and result-size limits are "
                    "published in the request schema."
                ),
                input={"lower_bound": 1, "upper_bound": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.divisor_count_profile.compute",
        title="Compute divisor-count profile on a bounded interval",
        description=(
            "Return the complete ordered table (n, tau(n)) for every integer "
            "n in a closed positive interval [L, U], where tau(n) is the "
            "number of positive divisors of n."
        ),
        request_type=DivisorCountProfileRequest,
        result_type=DivisorCountProfileResult,
        run=compute_divisor_count_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="divisor_count_interval_1_to_12",
                description=(
                    "Compute tau(n) for each n from 1 to 12; coupled width and "
                    "result-size limits are published in the request schema."
                ),
                input={"lower_bound": 1, "upper_bound": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.greatest_prime_factor_profile.compute",
        title="Compute greatest-prime-factor profile on a bounded interval",
        description=(
            "Return the complete ordered table (n, P+(n)) for every integer "
            "n in a closed positive interval [L, U], where P+(1) = 1 and "
            "P+(n) is the largest prime divisor of n for n >= 2."
        ),
        request_type=GreatestPrimeFactorProfileRequest,
        result_type=GreatestPrimeFactorProfileResult,
        run=compute_greatest_prime_factor_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="gpf_interval_1_to_10",
                description=(
                    "Compute P+(n) for each n from 1 to 10; coupled width and "
                    "result-size limits are published in the request schema."
                ),
                input={"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.prime_gap_profile.compute",
        title="Compute consecutive-prime gap profile on a bounded interval",
        description=(
            "Return every consecutive-prime pair (p, q, q - p) for which the "
            "lower endpoint p lies in a closed positive interval [L, U], "
            "including the successor prime beyond U when needed to complete "
            "the last gap."
        ),
        request_type=PrimeGapProfileRequest,
        result_type=PrimeGapProfileResult,
        run=compute_prime_gap_profile,
        tags=("number-theory", "prime", "interval-profile"),
        examples=(
            OperationExample(
                name="prime_gap_interval_3_to_5",
                description=(
                    "Compute consecutive-prime gaps for primes with lower "
                    "endpoint between 3 and 5; coupled width and result-size "
                    "limits are published in the request schema."
                ),
                input={"lower_bound": 3, "upper_bound": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.least_prime_factor_profile.compute",
        title="Compute least-prime-factor profile on a bounded interval",
        description="Return the complete ordered table (n, p(n)) for every n in [L, U], with p(1)=1.",
        request_type=LeastPrimeFactorProfileRequest,
        result_type=LeastPrimeFactorProfileResult,
        run=compute_least_prime_factor_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="lpf_1_10",
                description="Compute p(n) for each n from 1 to 10.",
                input={"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.euler_totient_profile.compute",
        title="Compute Euler-totient profile on a bounded interval",
        description="Return the complete ordered table (n, phi(n)) for every n in [L, U], with phi(1)=1.",
        request_type=EulerTotientProfileRequest,
        result_type=EulerTotientProfileResult,
        run=compute_euler_totient_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="totient_1_10",
                description="Compute phi(n) for each n from 1 to 10.",
                input={"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.divisor_sum_profile.compute",
        title="Compute divisor-sum profile on a bounded interval",
        description="Return the complete ordered table (n, sigma(n)) for every n in [L, U], with sigma(1)=1.",
        request_type=DivisorSumProfileRequest,
        result_type=DivisorSumProfileResult,
        run=compute_divisor_sum_profile,
        tags=("number-theory", "arithmetic-function", "interval-profile"),
        examples=(
            OperationExample(
                name="divisor_sum_1_6",
                description="Compute sigma(n) for each n from 1 to 6.",
                input={"lower_bound": 1, "upper_bound": 6},
            ),
        ),
    ),
)

__all__ = ["INTERVAL_PROFILE_OPERATIONS"]
