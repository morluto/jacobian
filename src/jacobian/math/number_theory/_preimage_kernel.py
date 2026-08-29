"""Exact kernels for multiplier divisor-sum fibers and p-adic profiles."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS
from jacobian.math.number_theory._preimage_models import (
    MAX_INTERVAL_PROFILE_RESULT_BYTES,
    MAX_INTERVAL_PROFILE_ROWS,
    MAX_INTERVAL_PROFILE_WORK,
    MAX_KSIGMA_MULTIPLIER,
    MAX_KSIGMA_SEARCH,
    MAX_KSIGMA_TARGET,
    PRIMALITY_WORK_DIGIT_EXPONENT,
)


@dataclass(frozen=True, slots=True)
class _PAdicIntervalProfilePlan:
    """One exact profile and its derived result data for the kernel."""

    rows: tuple[tuple[int, int], ...]
    total_valuation: int
    maximum_valuation: int


def _primality_work_units(prime: int) -> int:
    """Conservatively charge digit-dependent primality-test arithmetic."""

    # SymPy's pinned isprime backend uses a bounded number of modular
    # exponentiation and Lucas-test rounds for inputs above its small-prime
    # tables. Charging cubically in decimal digit count covers the
    # digit-dependent integer arithmetic while keeping this operation's
    # existing work-unit scale. The preflight charge is checked before the
    # backend call, so an oversized prime never reaches isprime.
    return int(pow(len(str(prime)), PRIMALITY_WORK_DIGIT_EXPONENT))


def _profile_result_payload(
    start: int,
    length: int,
    prime: int,
    *,
    rows: tuple[tuple[int, int], ...],
    total_valuation: int,
    maximum_valuation: int,
) -> dict[str, object]:
    """Build the canonical payload whose size is part of request admission."""

    return {
        "start": format_canonical_integer(start),
        "length": format_canonical_integer(length),
        "prime": format_canonical_integer(prime),
        "rows": [
            {
                "valuation": valuation,
                "count": format_canonical_integer(count),
            }
            for valuation, count in rows
        ],
        "total_valuation": format_canonical_integer(total_valuation),
        "maximum_valuation": maximum_valuation,
    }


def _admit_p_adic_interval_profile(
    start: int,
    length: int,
    prime: int,
) -> _PAdicIntervalProfilePlan:
    """Admit one interval and retain the derived profile for its result."""

    if start < 0:
        raise OperationDomainValidationError(
            location=("start",),
            code="number_theory.p_adic_interval_start_must_be_nonnegative",
            message="start must be nonnegative",
        )
    if length < 1:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_length_must_be_positive",
            message="length must be positive",
        )
    if prime < 2:
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.p_adic_interval_prime_must_be_at_least_two",
            message="prime must be at least two",
        )

    endpoint = start + length
    powers: list[int] = []
    power = 1
    while power <= endpoint:
        powers.append(power)
        if power > endpoint // prime:
            break
        power *= prime

    power_count = len(powers)
    # Two interval quotient differences and one power-step charge per visited
    # power, plus the digit-dependent primality charge, bound the exact
    # integer work used to build the retained profile.
    profile_work_units = 3 * power_count
    work_units = profile_work_units + _primality_work_units(prime)
    if (
        power_count > MAX_INTERVAL_PROFILE_ROWS
        or work_units > MAX_INTERVAL_PROFILE_WORK
    ):
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_profile_row_bound",
            message=(
                f"profile needs at most {MAX_INTERVAL_PROFILE_ROWS} visited powers "
                f"and {MAX_INTERVAL_PROFILE_WORK} combined arithmetic work units, "
                "including primality testing"
            ),
        )

    from sympy import isprime

    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.p_adic_interval_prime_must_be_prime",
            message="prime must be prime",
        )

    divisible_counts = [endpoint // power - start // power for power in powers]

    rows = tuple(
        (valuation, count)
        for valuation, (current, following) in enumerate(
            zip(
                divisible_counts,
                (*divisible_counts[1:], 0),
                strict=True,
            )
        )
        if (count := current - following)
    )
    total_valuation = sum(valuation * count for valuation, count in rows)
    if len(format_canonical_integer(total_valuation)) > MAX_INTEGER_DIGITS:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_total_valuation_digits",
            message=(
                "exact total valuation must fit the "
                f"{MAX_INTEGER_DIGITS}-digit canonical integer bound"
            ),
        )

    maximum_valuation = rows[-1][0] if rows else 0
    payload = _profile_result_payload(
        start,
        length,
        prime,
        rows=rows,
        total_valuation=total_valuation,
        maximum_valuation=maximum_valuation,
    )
    try:
        result_bytes = len(
            encode_strict_json(
                payload,
                limits=CanonicalLimits(
                    max_output_bytes=MAX_INTERVAL_PROFILE_RESULT_BYTES
                ),
            )
        )
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_result_bytes",
            message=(
                "exact profile result must fit the "
                f"{MAX_INTERVAL_PROFILE_RESULT_BYTES}-byte canonical output bound"
            ),
        ) from exc
    if result_bytes > MAX_INTERVAL_PROFILE_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("length",),
            code="number_theory.p_adic_interval_result_bytes",
            message=(
                "exact profile result must fit the "
                f"{MAX_INTERVAL_PROFILE_RESULT_BYTES}-byte canonical output bound"
            ),
        )

    return _PAdicIntervalProfilePlan(
        rows=rows,
        total_valuation=total_valuation,
        maximum_valuation=maximum_valuation,
    )


def ksigma_preimages(k: int, target: int) -> tuple[int, ...]:
    """Compute every positive ``n`` with ``k * sigma(n) == target``."""
    from sympy import divisor_sigma

    if not 1 <= target <= MAX_KSIGMA_TARGET:
        raise OperationDomainValidationError(
            location=("target_value",),
            code="number_theory.ksigma_target_bound",
            message=f"target_value must be between 1 and {MAX_KSIGMA_TARGET}",
        )
    if not 1 <= k <= MAX_KSIGMA_MULTIPLIER:
        raise OperationDomainValidationError(
            location=("k",),
            code="number_theory.ksigma_multiplier_bound",
            message=f"k must be between 1 and {MAX_KSIGMA_MULTIPLIER}",
        )
    if target % k:
        return ()

    sigma_target = target // k
    if sigma_target > MAX_KSIGMA_SEARCH:
        raise OperationDomainValidationError(
            location=("target_value",),
            code="number_theory.ksigma_search_bound",
            message=(
                "target_value / k must be at most "
                f"{MAX_KSIGMA_SEARCH} when k divides target_value"
            ),
        )

    # sigma(n) >= n + 1 for n > 1, while sigma(1) = 1, so this is a
    # complete search of the positive preimage rather than a heuristic cap.
    preimages = tuple(
        n for n in range(1, sigma_target + 1) if int(divisor_sigma(n)) == sigma_target
    )
    return preimages


def p_adic_interval_profile(
    start: int,
    length: int,
    prime: int,
) -> _PAdicIntervalProfilePlan:
    """Admit an interval and return its exact derived valuation profile."""

    return _admit_p_adic_interval_profile(start, length, prime)


__all__ = [
    "_PAdicIntervalProfilePlan",
    "ksigma_preimages",
    "p_adic_interval_profile",
]
