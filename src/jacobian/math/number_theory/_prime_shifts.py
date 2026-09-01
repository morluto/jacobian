"""Declarations for translated-prime representation profiles."""

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory._prime_shift_models import (
    MAX_SHIFT_INTERVAL_WIDTH,
    MAX_SHIFT_WORK,
    PrimeShiftProfileRequest,
    PrimeShiftProfileResult,
)
from jacobian.math.number_theory.operations import prime_shift_profile


def compute_prime_shift_profile(
    request: PrimeShiftProfileRequest,
) -> PrimeShiftProfileResult:
    """Project one wire request onto the native translated-prime operation."""

    try:
        return prime_shift_profile(request.lower_bound, request.upper_bound)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.translated_prime.admission",
            message=str(exc),
        ) from exc


PRIME_SHIFT_OPERATION = MathTool(
    operation_id="number_theory.translated_prime.representation_profile.compute",
    title="Compute translated-prime representation profile on a bounded interval",
    description=(
        "For each n in [L, U], count representations n = p + 2^k where p is "
        "prime and k >= 0. Returns the complete ordered profile for at most "
        f"{MAX_SHIFT_INTERVAL_WIDTH} materialized rows when its segmented-sieve "
        f"work is within {MAX_SHIFT_WORK} "
        "units. Endpoint size is admitted by that derived work envelope rather "
        "than by a fixed scalar cap."
    ),
    request_type=PrimeShiftProfileRequest,
    result_type=PrimeShiftProfileResult,
    run=compute_prime_shift_profile,
    tags=("number-theory", "prime", "interval-profile"),
    examples=(
        OperationExample(
            name="prime_shift_1_20",
            description="Compute translated-prime representations for n from 1 to 20.",
            input={"lower_bound": 1, "upper_bound": 20},
        ),
    ),
)

__all__ = ["PRIME_SHIFT_OPERATION", "compute_prime_shift_profile"]
