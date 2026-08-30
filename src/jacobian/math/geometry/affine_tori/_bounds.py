"""One admission plan, deadline, and work ledger for affine-torus fixed loci."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Literal, NoReturn

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.affine_tori.values import RationalAffineTorusMap

# The dense upper-triangular 16-by-16, 32-digit boundary fixture completes in
# well under one second on an ordinary development host.  Two minutes leaves a
# generous platform margin while retaining one finite owner deadline.
AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS = 120.0
MAX_AFFINE_TORUS_WORK_UNITS = 40_000_000
MAX_AFFINE_TORUS_RESULT_BYTES = 1_048_576

# Hadamard bounds at n=16 and 33 input digits give at most 538 decimal
# digits for every rank minor.  A common denominator of sixteen 32-digit
# translation coordinates has at most 512 digits.  Solving the admitted
# systems then gives a conservative 1,050-digit base-point component bound.
MAX_AFFINE_TORUS_MINOR_DIGITS = 538
MAX_AFFINE_TORUS_COMMON_DENOMINATOR_DIGITS = 512
MAX_AFFINE_TORUS_BASE_POINT_DIGITS = 1_050

type AffineTorusWorkCategory = Literal[
    "source_conversion",
    "hnf",
    "rank_minor_selection",
    "rational_solves",
    "snf",
    "normalization",
    "serialization",
]

_WORK_CATEGORIES: tuple[AffineTorusWorkCategory, ...] = (
    "source_conversion",
    "hnf",
    "rank_minor_selection",
    "rational_solves",
    "snf",
    "normalization",
    "serialization",
)


@dataclass(frozen=True, slots=True)
class AffineTorusFixedLocusPlan:
    """The conservative envelope reused by every mandatory kernel phase."""

    dimension: int
    deadline: float
    result_bytes_upper_bound: int
    work_units_by_category: tuple[tuple[AffineTorusWorkCategory, int], ...]

    @property
    def work_units(self) -> int:
        return sum(amount for _, amount in self.work_units_by_category)


class _Ledger:
    """Admission-only accounting in the same units used for all kernel phases."""

    def __init__(self, *, deadline: float) -> None:
        self.deadline = deadline
        self.total = 0
        self.by_category: dict[AffineTorusWorkCategory, int] = dict.fromkeys(
            _WORK_CATEGORIES, 0
        )

    def charge(self, category: AffineTorusWorkCategory, amount: int) -> None:
        if amount < 0:
            raise AssertionError("affine-torus work charges must be nonnegative")
        require_affine_torus_deadline(self.deadline, f"while charging {category} work")
        self.total += amount
        self.by_category[category] += amount
        if self.total > MAX_AFFINE_TORUS_WORK_UNITS:
            _reject(
                "work_budget",
                "affine-torus fixed-locus arithmetic exceeds the "
                f"{MAX_AFFINE_TORUS_WORK_UNITS}-unit work budget",
            )

    def freeze(self) -> tuple[tuple[AffineTorusWorkCategory, int], ...]:
        return tuple(
            (category, self.by_category[category]) for category in _WORK_CATEGORIES
        )


def _reject(reason: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("affine_map",),
        code=f"affine_torus.fixed_locus.{reason}",
        message=message,
    )


def begin_affine_torus_deadline() -> float:
    """Bind the one deadline covering admission, FLINT, result, and projection."""

    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    require_affine_torus_deadline(deadline, "before semantic admission")
    return deadline


def require_affine_torus_deadline(deadline: float, stage: str) -> None:
    """Stop without a mathematical conclusion when the request envelope expires."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"affine-torus fixed-locus computation cancelled {stage}"
        )
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"affine-torus fixed-locus deadline expired {stage}"
        )


def _result_bytes_upper_bound(dimension: int) -> int:
    """Bound the full retained source and the larger NONEMPTY result branch."""

    input_scalar_bytes = 2 * 32 + 32
    minor_integer_bytes = MAX_AFFINE_TORUS_MINOR_DIGITS + 4
    generator_rational_bytes = 2 * MAX_AFFINE_TORUS_MINOR_DIGITS + 40
    base_rational_bytes = 2 * MAX_AFFINE_TORUS_BASE_POINT_DIGITS + 40
    source_bytes = (dimension * dimension + dimension) * input_scalar_bytes + 8_192
    integer_matrix_bytes = 2 * dimension * dimension * minor_integer_bytes
    generator_bytes = dimension * dimension * generator_rational_bytes
    base_bytes = dimension * base_rational_bytes
    repeated_parent_and_json_bytes = 96_000 + 1_024 * dimension
    return (
        source_bytes
        + integer_matrix_bytes
        + generator_bytes
        + base_bytes
        + repeated_parent_and_json_bytes
    )


def build_affine_torus_plan(
    source: RationalAffineTorusMap, *, deadline: float
) -> AffineTorusFixedLocusPlan:
    """Precharge every possible phase before the first FLINT backend call."""

    require_affine_torus_deadline(deadline, "before work accounting")
    dimension = source.torus.dimension
    # One unit is one 32-decimal-digit scalar operation.  HNF and solve costs
    # are deliberately charged at the proved worst-case output heights, not at
    # the unusually small values in common examples.
    minor_chunks = (MAX_AFFINE_TORUS_MINOR_DIGITS + 31) // 32
    common_denominator_chunks = (MAX_AFFINE_TORUS_COMMON_DENOMINATOR_DIGITS + 31) // 32
    base_chunks = (MAX_AFFINE_TORUS_BASE_POINT_DIGITS + 31) // 32
    cube = dimension**3
    result_bytes = _result_bytes_upper_bound(dimension)
    if result_bytes > MAX_AFFINE_TORUS_RESULT_BYTES:
        _reject(
            "result_size",
            f"predicted exact result of {result_bytes} bytes exceeds the "
            f"{MAX_AFFINE_TORUS_RESULT_BYTES}-byte owner result bound",
        )
    ledger = _Ledger(deadline=deadline)
    ledger.charge("source_conversion", dimension * dimension + 2 * dimension)
    # Three saturated kernels each use HNF with transformation, followed by a
    # canonical row-HNF of the extracted lattice basis.
    ledger.charge("hnf", 6 * cube * minor_chunks**2)
    # Greedy first-rank-increasing rows and columns inspect at most 2n minors;
    # the three kernel ranks, their basis checks, and rank(M) add seven calls.
    ledger.charge("rank_minor_selection", (2 * dimension + 7) * cube * minor_chunks)
    # Wz=h, R, E, C^{-1}, and the base point are five exact rational solves.
    ledger.charge("rational_solves", 5 * cube * base_chunks**2)
    ledger.charge("snf", cube * minor_chunks**2)
    ledger.charge(
        "normalization",
        cube * base_chunks + dimension * dimension * common_denominator_chunks**2,
    )
    ledger.charge("serialization", (result_bytes + 31) // 32)
    require_affine_torus_deadline(deadline, "after semantic admission")
    return AffineTorusFixedLocusPlan(
        dimension=dimension,
        deadline=deadline,
        result_bytes_upper_bound=result_bytes,
        work_units_by_category=ledger.freeze(),
    )


__all__ = [
    "AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS",
    "MAX_AFFINE_TORUS_BASE_POINT_DIGITS",
    "MAX_AFFINE_TORUS_COMMON_DENOMINATOR_DIGITS",
    "MAX_AFFINE_TORUS_MINOR_DIGITS",
    "MAX_AFFINE_TORUS_RESULT_BYTES",
    "MAX_AFFINE_TORUS_WORK_UNITS",
    "AffineTorusFixedLocusPlan",
    "begin_affine_torus_deadline",
    "build_affine_torus_plan",
    "require_affine_torus_deadline",
]
