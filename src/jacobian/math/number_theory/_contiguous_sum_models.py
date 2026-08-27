"""Typed contracts for contiguous-sum representation profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

if TYPE_CHECKING:
    from jacobian.math.number_theory._contiguous_sum_admission import (
        ContiguousSumProfileAdmission,
    )

# The segmented regime stores one residual and one divisor count per requested
# integer. The high-magnitude regime factors each requested integer directly,
# so its width is deliberately narrower than the dense interval regime.
MAX_INTERVAL_WIDTH: int = 100_000
MAX_PROFILE_INTEGER_DIGITS: int = 20
MAX_FACTORING_INTERVAL_WIDTH: int = 128
MAX_INTERVAL_WORK: int = 6_000_000
MAX_INTERVAL_RESULT_BYTES: int = 8_000_000
MAX_SEGMENTED_SIEVE_UPPER: int = 10**12
MAX_FACTORING_WORK_SECONDS: int = 60

ContiguousSumInteger = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_PROFILE_INTEGER_DIGITS, strict=True),
]

ContiguousSumFailureKind = Literal[
    "WORKER_START_FAILED",
    "WORKER_TIMEOUT",
    "WORKER_CANCELLED",
    "STDOUT_LIMIT_EXCEEDED",
    "STDERR_LIMIT_EXCEEDED",
    "WORKER_RESOURCE_LIMIT",
    "WORKER_EXITED",
    "MALFORMED_OUTPUT",
    "REQUEST_DEADLINE_EXPIRED",
]
ContiguousSumTimeoutLayer = Literal[
    "WORKER_START",
    "WORKER_WALL",
    "REQUEST_CANCELLATION",
    "OUTPUT_LIMIT",
    "PROCESS_RESOURCE",
    "WORKER_EXIT",
    "RESULT_VALIDATION",
    "REQUEST_DEADLINE",
]


class ContiguousSumWorkerDiagnostic(StrictModel):
    """Bounded evidence explaining why a factorization profile is UNKNOWN."""

    failure: ContiguousSumFailureKind
    timeout_layer: ContiguousSumTimeoutLayer
    elapsed_ms: StrictInt = Field(ge=0)
    worker_timeout_ms: StrictInt = Field(ge=0, le=MAX_FACTORING_WORK_SECONDS * 1000)
    budget_seconds: StrictInt = Field(ge=1, le=MAX_FACTORING_WORK_SECONDS)
    returncode: StrictInt | None = Field(default=None, ge=-(2**31), le=(2**32) - 1)
    operation_version: Literal["1"] = "1"
    repository_revision: Annotated[
        str,
        StringConstraints(
            pattern=r"^(?:unknown|[0-9a-f]{40})$", max_length=40, strict=True
        ),
    ]


class ContiguousSumProfileRequest(StrictModel):
    """A bounded closed positive interval [L, U] for contiguous-sum profiling.

    Endpoints are strict positive integers of at most 20 decimal digits. The
    interval contains at most 100,000 integers; intervals above 10**12 use
    direct factorization and therefore contain at most 128 integers.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A closed positive integer interval [lower_bound, upper_bound]. "
                f"Endpoints must be strict integers with at most "
                f"{MAX_PROFILE_INTEGER_DIGITS} decimal digits. The interval "
                f"contains at most {MAX_INTERVAL_WIDTH:,} integers; intervals "
                f"above {MAX_SEGMENTED_SIEVE_UPPER:,} use direct factorization "
                f"and contain at most {MAX_FACTORING_INTERVAL_WIDTH} integers."
            ),
            "x-jacobian-bounds": {
                "max_interval_width": MAX_INTERVAL_WIDTH,
                "max_profile_integer_digits": MAX_PROFILE_INTEGER_DIGITS,
                "max_factoring_interval_width": MAX_FACTORING_INTERVAL_WIDTH,
                "max_interval_work": MAX_INTERVAL_WORK,
                "max_interval_result_bytes": MAX_INTERVAL_RESULT_BYTES,
                "segmented_sieve_upper": MAX_SEGMENTED_SIEVE_UPPER,
                "max_factoring_work_seconds": MAX_FACTORING_WORK_SECONDS,
            },
        }
    )

    lower_bound: ContiguousSumInteger = Field(
        description=(
            "Inclusive lower endpoint as a canonical positive decimal integer "
            f"with at most {MAX_PROFILE_INTEGER_DIGITS} digits."
        ),
    )
    upper_bound: ContiguousSumInteger = Field(
        description=(
            "Inclusive upper endpoint as a canonical positive decimal integer "
            f"with at most {MAX_PROFILE_INTEGER_DIGITS} digits."
        ),
    )


class ContiguousSumProfileRow(StrictModel):
    """One (n, count) pair where count is the number of contiguous-sum representations."""

    n: ContiguousSumInteger
    representation_count: StrictInt = Field(ge=1)


class ContiguousSumProfileResult(StrictModel):
    """Complete or operationally incomplete profile over a closed interval."""

    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    lower_bound: ContiguousSumInteger
    upper_bound: ContiguousSumInteger
    rows: tuple[ContiguousSumProfileRow, ...] = Field(
        min_length=0, max_length=MAX_INTERVAL_WIDTH
    )
    detail: str | None = None
    diagnostic: ContiguousSumWorkerDiagnostic | None = None

    @classmethod
    def _unknown_from_kernel(
        cls,
        *,
        admission: ContiguousSumProfileAdmission,
        detail: str,
        diagnostic: ContiguousSumWorkerDiagnostic,
    ) -> Self:
        """Build the typed non-conclusion from the admitted execution plan."""

        return cls.model_construct(
            status="UNKNOWN",
            lower_bound=format_canonical_integer(admission.lower_bound),
            upper_bound=format_canonical_integer(admission.upper_bound),
            rows=(),
            detail=detail,
            diagnostic=diagnostic,
        )

    @classmethod
    def _complete_from_kernel(
        cls,
        *,
        admission: ContiguousSumProfileAdmission,
        counts: tuple[int, ...],
    ) -> Self:
        """Build a complete profile using the plan that bounded its kernel."""

        if len(counts) != admission.width:
            raise RuntimeError("contiguous-sum kernel returned the wrong row count")
        rows = tuple(
            ContiguousSumProfileRow(
                n=format_canonical_integer(n), representation_count=count
            )
            for n, count in zip(
                range(admission.lower_bound, admission.upper_bound + 1),
                counts,
                strict=True,
            )
        )
        return cls.model_construct(
            status="COMPLETE",
            lower_bound=format_canonical_integer(admission.lower_bound),
            upper_bound=format_canonical_integer(admission.upper_bound),
            rows=rows,
            detail=None,
        )

    @model_validator(mode="after")
    def require_ordered_interval_rows(self) -> Self:
        lower = parse_canonical_integer(self.lower_bound)
        upper = parse_canonical_integer(self.upper_bound)
        if lower < 1 or upper < lower:
            raise ValueError("result endpoints must form a positive interval")
        if self.status == "UNKNOWN":
            if self.rows or not self.detail or self.diagnostic is None:
                raise ValueError(
                    "an unknown profile has no rows, detail, and diagnostic"
                )
            return self
        if self.detail is not None or self.diagnostic is not None:
            raise ValueError("a complete profile cannot include diagnostics")
        expected_width = upper - lower + 1
        if len(self.rows) != expected_width:
            raise ValueError("a complete profile has one row per interval integer")
        for expected, row in zip(range(lower, upper + 1), self.rows, strict=True):
            if parse_canonical_integer(row.n) != expected:
                raise ValueError("profile rows must be ordered over the interval")
        return self


__all__ = [
    "MAX_FACTORING_INTERVAL_WIDTH",
    "MAX_FACTORING_WORK_SECONDS",
    "MAX_INTERVAL_RESULT_BYTES",
    "MAX_INTERVAL_WIDTH",
    "MAX_INTERVAL_WORK",
    "MAX_PROFILE_INTEGER_DIGITS",
    "MAX_SEGMENTED_SIEVE_UPPER",
    "ContiguousSumInteger",
    "ContiguousSumProfileRequest",
    "ContiguousSumProfileResult",
    "ContiguousSumProfileRow",
    "ContiguousSumWorkerDiagnostic",
]
