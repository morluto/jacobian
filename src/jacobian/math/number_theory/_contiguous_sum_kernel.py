"""Bounded contiguous-sum profile execution kernel."""

from __future__ import annotations

import os
import re
from math import isqrt, prod
from time import monotonic
from typing import Literal

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._contiguous_sum_admission import (
    ContiguousSumProfileAdmission,
)
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileResult,
    ContiguousSumWorkerDiagnostic,
)
from jacobian.math.number_theory._factorization_kernels import (
    BoundedFactorizationFailure,
    _bounded_direct_factorization,
)

_CONTIGUOUS_SUM_OPERATION_VERSION: Literal["1"] = "1"
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _repository_revision() -> str:
    """Read the immutable build revision when the runtime provides one."""

    revision = os.environ.get("JACOBIAN_REVISION", "unknown")
    return revision if _GIT_SHA_PATTERN.fullmatch(revision) else "unknown"


def _worker_diagnostic(
    admission: ContiguousSumProfileAdmission,
    failure: BoundedFactorizationFailure,
    *,
    elapsed_seconds: float | None = None,
) -> ContiguousSumWorkerDiagnostic:
    """Project bounded worker evidence into the public UNKNOWN diagnostic."""

    budget_seconds = admission.factorization_budget_seconds
    assert budget_seconds is not None
    return ContiguousSumWorkerDiagnostic(
        failure=failure.kind,
        timeout_layer=failure.timeout_layer,
        elapsed_ms=round(
            (failure.elapsed_seconds if elapsed_seconds is None else elapsed_seconds)
            * 1_000
        ),
        worker_timeout_ms=round(failure.timeout_seconds * 1_000),
        budget_seconds=budget_seconds,
        returncode=failure.returncode,
        operation_version=_CONTIGUOUS_SUM_OPERATION_VERSION,
        repository_revision=_repository_revision(),
    )


def _odd_primes_up_to(limit: int) -> list[int]:
    """Return odd primes up to ``limit`` with a bounded segmented regime."""

    sieve = bytearray(b"\x01") * (limit + 1)
    if limit >= 0:
        sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for candidate in range(3, isqrt(limit) + 1, 2):
        if sieve[candidate]:
            for composite in range(candidate * candidate, limit + 1, 2 * candidate):
                sieve[composite] = 0
    return [candidate for candidate in range(3, limit + 1, 2) if sieve[candidate]]


def _segmented_odd_divisor_counts(
    admission: ContiguousSumProfileAdmission,
) -> tuple[int, ...]:
    """Count odd divisors over a dense interval without prefix allocation."""

    residuals = list(range(admission.lower_bound, admission.upper_bound + 1))
    width = admission.width
    counts = [1] * width
    for prime in _odd_primes_up_to(isqrt(admission.upper_bound)):
        first_multiple = ((admission.lower_bound + prime - 1) // prime) * prime
        for multiple in range(first_multiple, admission.upper_bound + 1, prime):
            index = multiple - admission.lower_bound
            residual = residuals[index]
            exponent = 0
            while residual % prime == 0:
                residual //= prime
                exponent += 1
            if exponent:
                residuals[index] = residual
                counts[index] *= exponent + 1
    for index, residual in enumerate(residuals):
        while residual % 2 == 0:
            residual //= 2
        if residual > 1:
            counts[index] *= 2
    return tuple(counts)


def _factored_odd_divisor_count(
    value: int,
    *,
    timeout_seconds: float,
    failure: list[BoundedFactorizationFailure],
) -> int | None:
    """Count odd divisors through the bounded factorization worker."""

    factors = _bounded_direct_factorization(
        value, timeout_seconds=timeout_seconds, failure=failure
    )
    if factors is None:
        return None
    return prod(
        factor.power + 1
        for factor in factors
        if parse_canonical_integer(factor.prime) % 2
    )


def run_contiguous_sum_profile(
    admission: ContiguousSumProfileAdmission,
    *,
    profile_started: float,
) -> ContiguousSumProfileResult:
    """For each n in [L, U], count representations as a sum of consecutive positive integers.

    A contiguous-sum representation of n is: n = a + (a+1) + ... + (a+k-1)
    for some a >= 1 and k >= 1 (k=1 gives the trivial representation n=n).

    The number of such representations equals the number of odd divisors of n
    that are greater than 1 (or equivalently, the number of ways to factor
    n as (a+b)*(b-a+1)/2 with appropriate constraints).

    A known result: the number of ways to write n as a sum of consecutive
    positive integers equals the number of odd divisors of n (including 1).
    Dense intervals use a segmented odd-factor sieve, while high-magnitude
    narrow intervals use the maintained SymPy factorization backend.
    """
    if admission.regime == "SEGMENTED":
        counts = _segmented_odd_divisor_counts(admission)
    else:
        direct_counts: list[int] = []
        assert admission.factorization_budget_seconds is not None
        assert admission.execution_deadline is not None
        factorization_deadline = admission.execution_deadline
        for n in range(admission.lower_bound, admission.upper_bound + 1):
            remaining = factorization_deadline - monotonic()
            failures: list[BoundedFactorizationFailure] = []
            if remaining <= 0:
                count = None
                failures.append(
                    BoundedFactorizationFailure(
                        kind="REQUEST_DEADLINE_EXPIRED",
                        timeout_layer="REQUEST_DEADLINE",
                        elapsed_seconds=max(0.0, monotonic() - profile_started),
                        timeout_seconds=0.0,
                    )
                )
            else:
                count = _factored_odd_divisor_count(
                    n, timeout_seconds=remaining, failure=failures
                )
            if count is None:
                assert failures
                failure = failures[0]
                return ContiguousSumProfileResult._unknown_from_kernel(
                    admission=admission,
                    detail=(
                        "the bounded factorization worker did not establish "
                        "the complete profile"
                    ),
                    diagnostic=_worker_diagnostic(
                        admission,
                        failure,
                        elapsed_seconds=max(
                            failure.elapsed_seconds,
                            monotonic() - profile_started,
                        ),
                    ),
                )
            direct_counts.append(count)
        counts = tuple(direct_counts)
        if monotonic() >= factorization_deadline:
            failure = BoundedFactorizationFailure(
                kind="REQUEST_DEADLINE_EXPIRED",
                timeout_layer="REQUEST_DEADLINE",
                elapsed_seconds=monotonic() - profile_started,
                timeout_seconds=0.0,
            )
            return ContiguousSumProfileResult._unknown_from_kernel(
                admission=admission,
                detail="the request deadline expired before complete-result construction",
                diagnostic=_worker_diagnostic(admission, failure),
            )
    if (
        admission.execution_deadline is not None
        and monotonic() >= admission.execution_deadline
    ):
        failure = BoundedFactorizationFailure(
            kind="REQUEST_DEADLINE_EXPIRED",
            timeout_layer="REQUEST_DEADLINE",
            elapsed_seconds=monotonic() - profile_started,
            timeout_seconds=0.0,
        )
        return ContiguousSumProfileResult._unknown_from_kernel(
            admission=admission,
            detail="the request deadline expired before result construction",
            diagnostic=_worker_diagnostic(admission, failure),
        )
    result = ContiguousSumProfileResult._complete_from_kernel(
        admission=admission,
        counts=tuple(counts),
    )
    if (
        admission.execution_deadline is not None
        and monotonic() >= admission.execution_deadline
    ):
        failure = BoundedFactorizationFailure(
            kind="REQUEST_DEADLINE_EXPIRED",
            timeout_layer="REQUEST_DEADLINE",
            elapsed_seconds=monotonic() - profile_started,
            timeout_seconds=0.0,
        )
        return ContiguousSumProfileResult._unknown_from_kernel(
            admission=admission,
            detail="the request deadline expired after result construction",
            diagnostic=_worker_diagnostic(admission, failure),
        )
    return result


__all__ = ["run_contiguous_sum_profile"]
