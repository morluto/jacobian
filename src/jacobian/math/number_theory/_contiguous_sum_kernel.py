"""Bounded contiguous-sum profile execution kernel."""

from __future__ import annotations

from math import isqrt, prod
from time import monotonic
from typing import NoReturn

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.math.number_theory._contiguous_sum_admission import (
    ContiguousSumProfileAdmission,
)
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileResult,
)
from jacobian.math.number_theory._factorization_kernels import (
    BoundedFactorizationFailure,
    _bounded_direct_factorization,
)


def _raise_factorization_failure(failure: BoundedFactorizationFailure) -> NoReturn:
    if failure.kind == "WORKER_CANCELLED":
        raise OperationExecutionCancelledError("contiguous-sum factorization cancelled")
    if failure.kind in ("WORKER_TIMEOUT", "REQUEST_DEADLINE_EXPIRED"):
        raise OperationExecutionTimeoutError(
            "contiguous-sum factorization deadline expired",
            configured_seconds=failure.timeout_seconds,
            elapsed_seconds=failure.elapsed_seconds,
        )
    if failure.kind in ("STDOUT_LIMIT_EXCEEDED", "STDERR_LIMIT_EXCEEDED"):
        raise OperationResourceExhaustedError(ExecutionResource.OUTPUT)
    if failure.kind == "WORKER_RESOURCE_LIMIT":
        raise OperationResourceExhaustedError(ExecutionResource.MEMORY)
    reason = {
        "WORKER_START_FAILED": BackendFailureReason.STARTUP,
        "WORKER_EXITED": BackendFailureReason.ABNORMAL_EXIT,
        "MALFORMED_OUTPUT": BackendFailureReason.MALFORMED_RESPONSE,
    }.get(failure.kind, BackendFailureReason.INVALID_OUTPUT)
    raise OperationBackendError(reason)


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
    return prod(factor.power + 1 for factor in factors if factor.prime % 2)


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
                _raise_factorization_failure(failures[0])
            direct_counts.append(count)
        counts = tuple(direct_counts)
        if monotonic() >= factorization_deadline:
            raise OperationExecutionTimeoutError(
                "contiguous-sum deadline expired before result construction",
                elapsed_seconds=monotonic() - profile_started,
            )
    if (
        admission.execution_deadline is not None
        and monotonic() >= admission.execution_deadline
    ):
        raise OperationExecutionTimeoutError(
            "contiguous-sum deadline expired before result construction",
            elapsed_seconds=monotonic() - profile_started,
        )
    result = ContiguousSumProfileResult._complete_from_kernel(
        admission=admission,
        counts=tuple(counts),
    )
    if (
        admission.execution_deadline is not None
        and monotonic() >= admission.execution_deadline
    ):
        raise OperationExecutionTimeoutError(
            "contiguous-sum deadline expired after result construction",
            elapsed_seconds=monotonic() - profile_started,
        )
    return result


__all__ = ["run_contiguous_sum_profile"]
