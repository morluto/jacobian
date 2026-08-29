"""Exact interval-profile kernels for arithmetic functions.

All four operations share a segmented-sieve kernel.  The public contracts
are:

- squarefree:       partition [L, U] into squarefree and non-squarefree sets
- divisor-count:    (n, tau(n)) for every L <= n <= U
- greatest-prime:   (n, P+(n)) for every L <= n <= U
- prime-gap:        (p, q, q-p) for every consecutive-prime pair with L <= p <= U

Each kernel is a direct bounded computation; no CAS, solver, or persistent
state is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from jacobian.math.number_theory._interval_profile_models import (
    MAX_INTERVAL_UPPER_BOUND,
    MAX_INTERVAL_WIDTH,
    MAX_PROFILE_RESULT_BYTES,
    MAX_SIEVE_WORK,
    DivisorCountProfileResult,
    DivisorCountProfileRow,
    DivisorSumProfileResult,
    DivisorSumProfileRow,
    EulerTotientProfileResult,
    EulerTotientProfileRow,
    GreatestPrimeFactorProfileResult,
    GreatestPrimeFactorProfileRow,
    IntervalAdmission,
    LeastPrimeFactorProfileResult,
    LeastPrimeFactorProfileRow,
    PrimeGapProfileResult,
    PrimeGapProfileRow,
    SquarefreeProfileResult,
    _estimate_divisor_count_result_bytes,
    _estimate_factor_profile_work,
    _estimate_greatest_prime_factor_result_bytes,
    _estimate_prime_gap_result_bytes,
    _estimate_prime_gap_work,
    _estimate_squarefree_result_bytes,
    _estimate_squarefree_work,
)


class IntervalAdmissionError(ValueError):
    """A canonical interval fails one mathematical admission bound."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _admit_interval(
    lower_bound: int,
    upper_bound: int,
    *,
    result_estimator: Callable[[int, int], int],
    work_estimator: Callable[[int, int], int],
    max_width: int | None,
) -> IntervalAdmission:
    """Build one operation-local execution envelope for canonical bounds."""
    if type(lower_bound) is not int or type(upper_bound) is not int:
        raise TypeError("interval bounds must be integers")
    if lower_bound < 1 or upper_bound < 1:
        raise IntervalAdmissionError(
            "domain_bound", "interval bounds must be positive integers"
        )
    if lower_bound > MAX_INTERVAL_UPPER_BOUND or upper_bound > MAX_INTERVAL_UPPER_BOUND:
        raise IntervalAdmissionError(
            "domain_bound",
            f"interval bounds must be at most {MAX_INTERVAL_UPPER_BOUND}",
        )
    if upper_bound < lower_bound:
        raise IntervalAdmissionError(
            "order_bound", "upper_bound must be >= lower_bound"
        )
    width = upper_bound - lower_bound + 1
    if max_width is not None and width > max_width:
        raise IntervalAdmissionError(
            "width_bound", "interval width exceeds maximum supported width"
        )
    estimated_result_bytes = result_estimator(lower_bound, upper_bound)
    if estimated_result_bytes > MAX_PROFILE_RESULT_BYTES:
        raise IntervalAdmissionError(
            "output_bound", "interval result exceeds the canonical output budget"
        )
    estimated_work = work_estimator(lower_bound, upper_bound)
    if estimated_work > MAX_SIEVE_WORK:
        raise IntervalAdmissionError(
            "work_bound",
            "interval exceeds the segmented-sieve work budget of "
            f"{MAX_SIEVE_WORK} steps",
        )
    return IntervalAdmission(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        width=width,
        estimated_work=estimated_work,
        estimated_result_bytes=estimated_result_bytes,
    )


def _simple_sieve(limit: int) -> list[int]:
    """Return all primes up to and including limit using a basic sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, limit + 1) if is_prime[i]]


def _segmented_primes(lower_bound: int, upper_bound: int) -> list[int]:
    """Return primes in a closed interval without sieving below its lower end."""
    width = upper_bound - lower_bound + 1
    is_prime = bytearray(b"\x01") * width
    if lower_bound == 1:
        is_prime[0] = 0

    for p in _simple_sieve(math.isqrt(upper_bound)):
        first_multiple = max(
            p * p,
            ((lower_bound + p - 1) // p) * p,
        )
        for multiple in range(first_multiple, upper_bound + 1, p):
            is_prime[multiple - lower_bound] = 0

    return [
        lower_bound + offset
        for offset, marked_prime in enumerate(is_prime)
        if marked_prime
    ]


def _segmented_factor_profile_data(
    lower_bound: int, upper_bound: int
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Factor every interval value for the additional arithmetic profiles."""
    width = upper_bound - lower_bound + 1
    remaining = list(range(lower_bound, upper_bound + 1))
    divisor_counts = [1] * width
    greatest_prime_factors = [1] * width
    least_prime_factors = [1] * width
    euler_totients = list(range(lower_bound, upper_bound + 1))
    divisor_sums = [1] * width

    for prime in _simple_sieve(math.isqrt(upper_bound)):
        first_multiple = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        for value in range(first_multiple, upper_bound + 1, prime):
            index = value - lower_bound
            if remaining[index] % prime != 0:
                continue
            exponent = 0
            prime_power = 1
            while remaining[index] % prime == 0:
                remaining[index] //= prime
                prime_power *= prime
                exponent += 1
            divisor_counts[index] *= exponent + 1
            greatest_prime_factors[index] = prime
            if least_prime_factors[index] == 1:
                least_prime_factors[index] = prime
            euler_totients[index] = euler_totients[index] // prime * (prime - 1)
            divisor_sums[index] *= (prime_power * prime - 1) // (prime - 1)

    for index, cofactor in enumerate(remaining):
        if cofactor <= 1:
            continue
        divisor_counts[index] *= 2
        greatest_prime_factors[index] = cofactor
        if least_prime_factors[index] == 1:
            least_prime_factors[index] = cofactor
        euler_totients[index] = euler_totients[index] // cofactor * (cofactor - 1)
        divisor_sums[index] *= cofactor + 1

    return (
        divisor_counts,
        greatest_prime_factors,
        least_prime_factors,
        euler_totients,
        divisor_sums,
    )


def squarefree_profile(lower_bound: int, upper_bound: int) -> SquarefreeProfileResult:
    """Partition [L, U] into squarefree and non-squarefree integers."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_squarefree_result_bytes,
        work_estimator=_estimate_squarefree_work,
        max_width=None,
    )
    return _squarefree_profile_kernel(admission)


def _squarefree_profile_kernel(admission: IntervalAdmission) -> SquarefreeProfileResult:
    """Run the squarefree kernel using the already validated envelope."""
    lo = admission.lower_bound
    hi = admission.upper_bound
    width = admission.width

    # Segmented square sieve: mark n in [lo, hi] as non-squarefree if p^2 | n
    # for some prime p.  We only need primes p with p^2 <= hi.
    sqrt_hi = math.isqrt(hi)
    primes = _simple_sieve(sqrt_hi)

    is_non_squarefree = bytearray(width)
    for p in primes:
        p_sq = p * p
        # First multiple of p_sq in [lo, hi]
        start = ((lo + p_sq - 1) // p_sq) * p_sq
        for j in range(start, hi + 1, p_sq):
            is_non_squarefree[j - lo] = 1

    squarefree_values: list[int] = []
    nonsquarefree_values: list[int] = []
    for i in range(width):
        n = lo + i
        if is_non_squarefree[i]:
            nonsquarefree_values.append(n)
        else:
            squarefree_values.append(n)

    return SquarefreeProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        squarefree_values=tuple(squarefree_values),
        nonsquarefree_values=tuple(nonsquarefree_values),
        squarefree_count=len(squarefree_values),
        nonsquarefree_count=len(nonsquarefree_values),
    )


def divisor_count_profile(
    lower_bound: int, upper_bound: int
) -> DivisorCountProfileResult:
    """Compute tau(n) for every n in [L, U]."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_divisor_count_result_bytes,
        work_estimator=_estimate_factor_profile_work,
        max_width=MAX_INTERVAL_WIDTH,
    )
    return _divisor_count_profile_kernel(admission)


def _divisor_count_profile_kernel(
    admission: IntervalAdmission,
) -> DivisorCountProfileResult:
    """Run the divisor-count kernel using the already validated envelope."""
    lo = admission.lower_bound
    hi = admission.upper_bound
    width = admission.width

    # Factor each interval value in place.  The base sieve is bounded by
    # sqrt(hi), while the mutable residuals are bounded by the interval width.
    primes = _simple_sieve(math.isqrt(hi))
    residuals = list(range(lo, hi + 1))
    divisor_counts = [1] * width

    for p in primes:
        first_multiple = max(p * p, ((lo + p - 1) // p) * p)
        for offset in range(first_multiple - lo, width, p):
            residual = residuals[offset]
            if residual % p != 0:
                continue
            exponent = 0
            while residual % p == 0:
                residual //= p
                exponent += 1
            residuals[offset] = residual
            divisor_counts[offset] *= exponent + 1

    rows: list[DivisorCountProfileRow] = []
    for i in range(width):
        n = lo + i
        if residuals[i] > 1:
            divisor_counts[i] *= 2
        rows.append(DivisorCountProfileRow(n=n, divisor_count=divisor_counts[i]))

    return DivisorCountProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def greatest_prime_factor_profile(
    lower_bound: int, upper_bound: int
) -> GreatestPrimeFactorProfileResult:
    """Compute P+(n) for every n in [L, U]."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_greatest_prime_factor_result_bytes,
        work_estimator=_estimate_factor_profile_work,
        max_width=MAX_INTERVAL_WIDTH,
    )
    return _greatest_prime_factor_profile_kernel(admission)


def _greatest_prime_factor_profile_kernel(
    admission: IntervalAdmission,
) -> GreatestPrimeFactorProfileResult:
    """Run the greatest-prime-factor kernel using the validated envelope."""
    lo = admission.lower_bound
    hi = admission.upper_bound
    width = admission.width

    primes = _simple_sieve(math.isqrt(hi))
    residuals = list(range(lo, hi + 1))
    greatest_prime_factors = [1] * width

    for p in primes:
        first_multiple = max(p * p, ((lo + p - 1) // p) * p)
        for offset in range(first_multiple - lo, width, p):
            residual = residuals[offset]
            if residual % p != 0:
                continue
            greatest_prime_factors[offset] = p
            while residual % p == 0:
                residual //= p
            residuals[offset] = residual

    rows: list[GreatestPrimeFactorProfileRow] = []
    for i in range(width):
        n = lo + i
        if n == 1:
            rows.append(GreatestPrimeFactorProfileRow(n=1, greatest_prime_factor=1))
            continue
        gpf = residuals[i] if residuals[i] > 1 else greatest_prime_factors[i]
        rows.append(GreatestPrimeFactorProfileRow(n=n, greatest_prime_factor=gpf))

    return GreatestPrimeFactorProfileResult(
        lower_bound=lo,
        upper_bound=hi,
        rows=tuple(rows),
    )


def prime_gap_profile(lower_bound: int, upper_bound: int) -> PrimeGapProfileResult:
    """Compute consecutive-prime gaps for primes p with L <= p <= U."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_prime_gap_result_bytes,
        work_estimator=_estimate_prime_gap_work,
        max_width=None,
    )
    return _prime_gap_profile_kernel(admission)


def _prime_gap_profile_kernel(admission: IntervalAdmission) -> PrimeGapProfileResult:
    """Run the prime-gap kernel using the already validated envelope."""
    lo = admission.lower_bound
    hi = admission.upper_bound

    # Mark only [lo, hi]; the successor beyond hi is queried separately.
    primes_in_interval = _segmented_primes(lo, hi)

    # We need at least 2 primes to form a gap.  If no primes in [lo, hi],
    # return empty.  If exactly one prime in [lo, hi], we need the next
    # prime after hi to form the gap.
    if not primes_in_interval:
        return PrimeGapProfileResult(lower_bound=lo, upper_bound=hi, rows=())

    from sympy import nextprime

    # Admission charges the proven span of this mandatory successor search.
    # Dusart's bound makes the backend call finite on every admitted request.
    primes_in_interval.append(int(nextprime(hi)))

    rows: list[PrimeGapProfileRow] = []
    for i in range(len(primes_in_interval) - 1):
        p = primes_in_interval[i]
        q = primes_in_interval[i + 1]
        if p >= lo and p <= hi:
            rows.append(PrimeGapProfileRow(lower_prime=p, upper_prime=q, gap=q - p))

    return PrimeGapProfileResult(lower_bound=lo, upper_bound=hi, rows=tuple(rows))


def least_prime_factor_profile(
    lower_bound: int, upper_bound: int
) -> LeastPrimeFactorProfileResult:
    """Compute p(n), the least prime factor, for every n in [L, U]."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_divisor_count_result_bytes,
        work_estimator=_estimate_factor_profile_work,
        max_width=MAX_INTERVAL_WIDTH,
    )
    lo, hi = admission.lower_bound, admission.upper_bound
    _, _, least_prime_factors, _, _ = _segmented_factor_profile_data(lo, hi)
    rows = tuple(
        LeastPrimeFactorProfileRow(n=lo + i, least_prime_factor=value)
        for i, value in enumerate(least_prime_factors)
    )
    return LeastPrimeFactorProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


def euler_totient_profile(
    lower_bound: int, upper_bound: int
) -> EulerTotientProfileResult:
    """Compute phi(n), Euler's totient, for every n in [L, U]."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_divisor_count_result_bytes,
        work_estimator=_estimate_factor_profile_work,
        max_width=MAX_INTERVAL_WIDTH,
    )
    lo, hi = admission.lower_bound, admission.upper_bound
    _, _, _, euler_totients, _ = _segmented_factor_profile_data(lo, hi)
    rows = tuple(
        EulerTotientProfileRow(n=lo + i, euler_totient=value)
        for i, value in enumerate(euler_totients)
    )
    return EulerTotientProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


def divisor_sum_profile(lower_bound: int, upper_bound: int) -> DivisorSumProfileResult:
    """Compute sigma(n), the divisor sum, for every n in [L, U]."""
    admission = _admit_interval(
        lower_bound,
        upper_bound,
        result_estimator=_estimate_divisor_count_result_bytes,
        work_estimator=_estimate_factor_profile_work,
        max_width=MAX_INTERVAL_WIDTH,
    )
    lo, hi = admission.lower_bound, admission.upper_bound
    _, _, _, _, divisor_sums = _segmented_factor_profile_data(lo, hi)
    rows = tuple(
        DivisorSumProfileRow(n=lo + i, divisor_sum=value)
        for i, value in enumerate(divisor_sums)
    )
    return DivisorSumProfileResult(lower_bound=lo, upper_bound=hi, rows=rows)


__all__ = [
    "divisor_count_profile",
    "divisor_sum_profile",
    "euler_totient_profile",
    "greatest_prime_factor_profile",
    "least_prime_factor_profile",
    "prime_gap_profile",
    "squarefree_profile",
]
