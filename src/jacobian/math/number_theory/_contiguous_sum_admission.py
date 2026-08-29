"""One request-scoped admission plan for contiguous-sum profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Literal

from jacobian._execution import bind_request_deadline
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._contiguous_sum_models import (
    MAX_FACTORING_INTERVAL_WIDTH,
    MAX_FACTORING_WORK_SECONDS,
    MAX_INTERVAL_RESULT_BYTES,
    MAX_INTERVAL_WIDTH,
    MAX_INTERVAL_WORK,
    MAX_SEGMENTED_SIEVE_UPPER,
)

ContiguousSumRegime = Literal["SEGMENTED", "DIRECT_FACTORIZATION"]


@dataclass(frozen=True, slots=True)
class ContiguousSumProfileAdmission:
    """The complete execution envelope for one contiguous-sum request."""

    lower_bound: int
    upper_bound: int
    width: int
    regime: ContiguousSumRegime
    estimated_work: int
    estimated_result_bytes: int
    factorization_budget_seconds: int | None
    execution_deadline: float | None


def require_contiguous_sum_profile_admission(
    lower: int,
    upper: int,
    *,
    started_at: float | None = None,
) -> ContiguousSumProfileAdmission:
    """Validate one interval and select the complete bounded kernel regime."""

    if lower < 1 or upper < 1:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.contiguous_sum_positive_interval",
            message="interval endpoints must be positive",
        )
    if upper < lower:
        raise OperationDomainValidationError(
            location=("upper_bound",),
            code="number_theory.contiguous_sum_ordered_interval",
            message="upper_bound must be >= lower_bound",
        )

    width = upper - lower + 1
    if width > MAX_INTERVAL_WIDTH:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.contiguous_sum_interval_width",
            message="interval width exceeds maximum supported width",
        )

    upper_digits = len(str(upper))
    if upper > MAX_SEGMENTED_SIEVE_UPPER:
        if width > MAX_FACTORING_INTERVAL_WIDTH:
            raise OperationDomainValidationError(
                location=("lower_bound", "upper_bound"),
                code="number_theory.contiguous_sum_factoring_width",
                message=(
                    "high-magnitude intervals exceed the direct-factorization "
                    "width bound"
                ),
            )
        regime: ContiguousSumRegime = "DIRECT_FACTORIZATION"
        estimated_work = width * upper_digits * 1_000
        factorization_budget_seconds: int | None = MAX_FACTORING_WORK_SECONDS
        execution_deadline = (
            None if started_at is None else started_at + MAX_FACTORING_WORK_SECONDS
        )
        if execution_deadline is not None:
            bind_request_deadline(execution_deadline)
    else:
        regime = "SEGMENTED"
        estimated_work = 3 * isqrt(upper) + width * (upper_digits + 1)
        factorization_budget_seconds = None
        execution_deadline = None

    if estimated_work > MAX_INTERVAL_WORK:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.contiguous_sum_interval_work",
            message="interval work exceeds the maximum supported budget",
        )
    estimated_result_bytes = width * (upper_digits + 32)
    if estimated_result_bytes > MAX_INTERVAL_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.contiguous_sum_interval_result_size",
            message="interval result exceeds the maximum supported size",
        )
    return ContiguousSumProfileAdmission(
        lower_bound=lower,
        upper_bound=upper,
        width=width,
        regime=regime,
        estimated_work=estimated_work,
        estimated_result_bytes=estimated_result_bytes,
        factorization_budget_seconds=factorization_budget_seconds,
        execution_deadline=execution_deadline,
    )


__all__ = [
    "ContiguousSumProfileAdmission",
    "ContiguousSumRegime",
    "require_contiguous_sum_profile_admission",
]
