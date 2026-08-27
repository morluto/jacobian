"""Typed wire contracts for finite set-system discrepancy operations."""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_GROUND_SET = 64
MAX_SETS = 1_000
# The optimum operation pairs two maintained backends: a bounded
# scipy.optimize.milp (HiGHS) search produces the incumbent coloring, and the
# minimality of any positive claimed optimum is re-established exactly by one
# Z3 pseudo-boolean feasibility check at D-1 before OPTIMAL may carry it.
# Exhausted budgets return BUDGET_EXCEEDED without any mathematical claim.
MAX_OPTIMUM_SOLVER_MILLISECONDS = 30_000
# The node limit bounds HiGHS' branch-and-bound tree retention independently
# of wall clock: an admitted hard 64-variable instance can otherwise expand
# and retain an arbitrary portion of its ~2^64 search tree before the timer
# fires. Exhausting either limit yields the claim-free outcome; neither limit
# stands in for a mathematical bound on the optimum itself.
MAX_OPTIMUM_SOLVER_NODES = 1_000_000
# The exact proof check runs under its own explicit budget so one request's
# total solver work stays bounded; exhaustion reports unknown, which the
# producing path maps to BUDGET_EXCEEDED and replay rejects fail-closed.
MAX_OPTIMUM_PROOF_MILLISECONDS = 10_000

MAX_ROUNDING_COORDINATES = 512
MAX_ROUNDING_ROWS = 512
MAX_MONITORED_COLUMNS = 512
MAX_COLUMN_INCIDENCES = 32_768
MAX_ROUNDING_RATIONAL_DIGITS = 256
MAX_ROUNDING_WORK = 8_000_000
MAX_ROUNDING_INTERMEDIATE_DIGITS = 8_192
MAX_ROUNDING_RESULT_RATIONAL_DIGITS = 100_000

RoundingIndexLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, strict=True),
]
RoundingCoordinateIndex = Annotated[int, Field(ge=0, strict=True)]
RoundedBit = Annotated[int, Field(ge=0, le=1, strict=True)]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Return a stable structured reason for an owner-model validation failure."""

    return PydanticCustomError(f"discrepancy_theory.{reason}", message)


def _fraction_wire_size(value: Fraction) -> int:
    return len(str(abs(value.numerator))) + len(str(value.denominator))


def _sum_selected_fractions(
    values: list[Fraction], coordinates: tuple[int, ...]
) -> Fraction:
    """Aggregate one admitted row or column after cheap source preflight."""

    return sum((values[index] for index in coordinates), Fraction())


def _as_canonical_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _require_canonical_subset(
    elements: tuple[int, ...], *, coordinate_count: int, owner: str
) -> None:
    if any(type(index) is not int for index in elements):
        raise _validation_error(
            "coordinate_indices_not_strict_integers",
            f"{owner} coordinate indices must be strict integers",
        )
    if any(not 0 <= index < coordinate_count for index in elements):
        raise _validation_error(
            "coordinate_indices_out_of_range",
            f"{owner} coordinate indices must be in 0..coordinate_count-1",
        )
    if any(left >= right for left, right in itertools.pairwise(elements)):
        raise _validation_error(
            "coordinate_indices_not_strictly_increasing",
            f"{owner} coordinate indices must be strictly increasing",
        )


class HardConstraintRow(StrictModel):
    """One named nonempty block of the hard row partition."""

    label: RoundingIndexLabel
    coordinates: tuple[RoundingCoordinateIndex, ...] = Field(
        min_length=1,
        max_length=MAX_ROUNDING_COORDINATES,
    )


class MonitoredColumn(StrictModel):
    """One named monitored zero-one column, represented by its support."""

    label: RoundingIndexLabel
    coordinates: tuple[RoundingCoordinateIndex, ...] = Field(
        max_length=MAX_ROUNDING_COORDINATES
    )


class HardConstraintRoundingSource(StrictModel):
    """A materialized rational vector, hard row partition, and monitored columns.

    ``coordinate_labels`` is the ordered coordinate axis. Every row is a
    nonempty, strictly increasing tuple of coordinate indices, and the rows
    partition that axis exactly. Monitored columns are strictly increasing
    index tuples; distinct column labels may have identical supports.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A materialized exact vector with an ordered coordinate axis, a "
                "named nonempty-row partition of that axis, and named monitored "
                "zero-one columns. Row and column supports use strictly increasing "
                "zero-based coordinate indices. Rows must partition every "
                "coordinate exactly once; duplicate monitored supports are allowed."
            )
        }
    )

    coordinate_labels: tuple[RoundingIndexLabel, ...] = Field(
        max_length=MAX_ROUNDING_COORDINATES,
    )
    values: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_ROUNDING_COORDINATES,
    )
    rows: tuple[HardConstraintRow, ...] = Field(max_length=MAX_ROUNDING_ROWS)
    columns: tuple[MonitoredColumn, ...] = Field(max_length=MAX_MONITORED_COLUMNS)

    @model_validator(mode="after")
    def require_admitted_source(self) -> Self:
        coordinate_count = len(self.coordinate_labels)
        if len(set(self.coordinate_labels)) != coordinate_count:
            raise _validation_error(
                "coordinate_labels_not_unique", "coordinate labels must be unique"
            )
        if len(self.values) != coordinate_count:
            raise _validation_error(
                "values_axis_length_mismatch",
                "values must align with the coordinate axis",
            )
        if len({row.label for row in self.rows}) != len(self.rows):
            raise _validation_error(
                "row_labels_not_unique", "row labels must be unique"
            )
        if len({column.label for column in self.columns}) != len(self.columns):
            raise _validation_error(
                "column_labels_not_unique", "column labels must be unique"
            )

        fractions: list[Fraction] = []
        for value in self.values:
            require_bounded_rational(
                value,
                max_digits=MAX_ROUNDING_RATIONAL_DIGITS,
                label="rounding source rational",
            )
            fraction = value.as_fraction()
            if not 0 <= fraction <= 1:
                raise _validation_error(
                    "source_values_out_of_range",
                    "rounding source values must lie in [0, 1]",
                )
            fractions.append(fraction)

        partition: list[int] = []
        for row in self.rows:
            _require_canonical_subset(
                row.coordinates,
                coordinate_count=coordinate_count,
                owner="row",
            )
            partition.extend(row.coordinates)
        if sorted(partition) != list(range(coordinate_count)):
            raise _validation_error(
                "rows_not_a_partition",
                "hard rows must partition every coordinate exactly once",
            )

        total_incidences = 0
        for column in self.columns:
            _require_canonical_subset(
                column.coordinates,
                coordinate_count=coordinate_count,
                owner="column",
            )
            total_incidences += len(column.coordinates)
            if total_incidences > MAX_COLUMN_INCIDENCES:
                raise _validation_error(
                    "column_incidences_over_budget",
                    f"monitored column incidences exceed {MAX_COLUMN_INCIDENCES}",
                )

        fractional_count = sum(0 < value < 1 for value in fractions)
        scan_work = fractional_count * (coordinate_count + total_incidences)
        elimination_work = (fractional_count * (fractional_count + 1) // 2) ** 2
        if scan_work + elimination_work > MAX_ROUNDING_WORK:
            raise _validation_error(
                "rounding_work_over_budget",
                "rounding work bound exceeded by fractional support and incidences",
            )

        common_denominator_digits = 1 + sum(
            len(value.den) for value in self.values if value.den != "1"
        )
        # For an f-column zero-one retained system, fraction-free nullspace
        # coordinates are minors. Hadamard bounds each minor by f^(f/2);
        # f*digits(f)+1 is a simple conservative decimal bound. A common
        # denominator therefore grows by at most this factor per endpoint move.
        direction_growth_digits = sum(
            support * len(str(support)) + 1
            for support in range(2, fractional_count + 1)
        )
        if (
            common_denominator_digits + direction_growth_digits
            > MAX_ROUNDING_INTERMEDIATE_DIGITS
        ):
            raise _validation_error(
                "intermediate_rational_height_over_budget",
                "rounding intermediate rational-height bound exceeded",
            )

        row_sums = [
            _sum_selected_fractions(fractions, row.coordinates) for row in self.rows
        ]
        if any(row_sum.denominator != 1 for row_sum in row_sums):
            raise _validation_error(
                "row_sum_not_integral",
                "every hard row must have an integral source sum",
            )
        column_sums = [
            _sum_selected_fractions(fractions, column.coordinates)
            for column in self.columns
        ]

        input_rational_digits = sum(
            len(value.num.lstrip("-")) + len(value.den) for value in self.values
        )
        row_digits = sum(_fraction_wire_size(row_sum) for row_sum in row_sums)
        coordinate_digits = len(str(max(1, coordinate_count)))
        column_digits = sum(
            3 * (_fraction_wire_size(value) + coordinate_digits + 1)
            for value in column_sums
        )
        if (
            input_rational_digits + row_digits + column_digits
            > MAX_ROUNDING_RESULT_RATIONAL_DIGITS
        ):
            raise _validation_error(
                "result_rational_height_over_budget",
                "rounding exact result-size bound exceeded",
            )
        return self


class HardConstraintRoundingRequest(StrictModel):
    """Compute one exact binary rounding of an admitted materialized source."""

    source: HardConstraintRoundingSource


class HardConstraintRowLedger(StrictModel):
    """Exact source and rounded sums for one hard row."""

    row_label: RoundingIndexLabel
    source_sum: CanonicalRational
    rounded_sum: int = Field(ge=0, strict=True)


class MonitoredColumnLedger(StrictModel):
    """Exact source sum and rounded-minus-source error for one column."""

    column_label: RoundingIndexLabel
    source_sum: CanonicalRational
    rounded_sum: int = Field(ge=0, strict=True)
    signed_error: CanonicalRational
    absolute_error: CanonicalRational


class HardConstraintRoundingResult(StrictModel):
    """A source-bound binary rounding with hard-row and error ledgers.

    Parsing checks the result's wire shape only.  The floating-rounding kernel
    establishes preservation and error claims when it produces this value;
    deliberate validation of an independently supplied claim belongs to the
    owner-private :func:`_verify_hard_constraint_rounding_result` helper.
    """

    source: HardConstraintRoundingSource
    rounded_values: tuple[RoundedBit, ...] = Field(max_length=MAX_ROUNDING_COORDINATES)
    maximum_column_incidence: int = Field(ge=0, strict=True)
    column_error_bound: int = Field(ge=0, strict=True)
    row_ledger: tuple[HardConstraintRowLedger, ...] = Field(
        max_length=MAX_ROUNDING_ROWS
    )
    column_ledger: tuple[MonitoredColumnLedger, ...] = Field(
        max_length=MAX_MONITORED_COLUMNS
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        coordinate_count = len(self.source.coordinate_labels)
        if len(self.rounded_values) != coordinate_count:
            raise _validation_error(
                "rounded_values_axis_length_mismatch",
                "rounded values must align with the coordinate axis",
            )
        if any(
            type(value) is not int or value not in (0, 1)
            for value in self.rounded_values
        ):
            raise _validation_error(
                "rounded_values_not_binary",
                "rounded values must be strict binary integers",
            )
        if len(self.row_ledger) != len(self.source.rows):
            raise _validation_error(
                "row_ledger_length_mismatch",
                "row ledger must contain one entry for every hard row",
            )
        if tuple(item.row_label for item in self.row_ledger) != tuple(
            row.label for row in self.source.rows
        ):
            raise _validation_error(
                "row_ledger_labels_mismatch",
                "row ledger labels must align with the hard-row order",
            )
        if len(self.column_ledger) != len(self.source.columns):
            raise _validation_error(
                "column_ledger_length_mismatch",
                "column ledger must contain one entry for every monitored column",
            )
        if tuple(item.column_label for item in self.column_ledger) != tuple(
            column.label for column in self.source.columns
        ):
            raise _validation_error(
                "column_ledger_labels_mismatch",
                "column ledger labels must align with the monitored-column order",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build a result after the admitted floating-rounding kernel proved it."""

        return cls.model_construct(**values)


def _verify_hard_constraint_rounding_result(
    result: HardConstraintRoundingResult,
) -> bool:
    """Check one independently supplied rounding claim under request admission.

    This deliberately replays the source-derived ledgers and guarantees.  It
    is private because ordinary result deserialization is not proof checking.
    """

    try:
        admitted = HardConstraintRoundingRequest.model_validate(
            {"source": result.source.model_dump()}
        )
    except Exception:  # request admission is fail-closed for supplied claims
        return False
    source = admitted.source
    if len(result.rounded_values) != len(source.coordinate_labels):
        return False
    if any(
        type(value) is not int or value not in (0, 1) for value in result.rounded_values
    ):
        return False
    source_values = tuple(value.as_fraction() for value in source.values)
    expected_rows = tuple(
        HardConstraintRowLedger(
            row_label=row.label,
            source_sum=_as_canonical_rational(
                _sum_selected_fractions(list(source_values), row.coordinates)
            ),
            rounded_sum=sum(result.rounded_values[index] for index in row.coordinates),
        )
        for row in source.rows
    )
    if (
        any(row.source_sum.as_fraction() != row.rounded_sum for row in expected_rows)
        or result.row_ledger != expected_rows
    ):
        return False
    incidences = [0] * len(source_values)
    for column in source.columns:
        for index in column.coordinates:
            incidences[index] += 1
    maximum_incidence = max(incidences, default=0)
    error_bound = 4 * maximum_incidence
    if (
        result.maximum_column_incidence != maximum_incidence
        or result.column_error_bound != error_bound
    ):
        return False
    expected_columns = tuple(
        MonitoredColumnLedger(
            column_label=column.label,
            source_sum=_as_canonical_rational(
                _sum_selected_fractions(list(source_values), column.coordinates)
            ),
            rounded_sum=sum(
                result.rounded_values[index] for index in column.coordinates
            ),
            signed_error=_as_canonical_rational(
                Fraction(
                    sum(result.rounded_values[index] for index in column.coordinates)
                )
                - _sum_selected_fractions(list(source_values), column.coordinates)
            ),
            absolute_error=_as_canonical_rational(
                abs(
                    Fraction(
                        sum(
                            result.rounded_values[index] for index in column.coordinates
                        )
                    )
                    - _sum_selected_fractions(list(source_values), column.coordinates)
                )
            ),
        )
        for column in source.columns
    )
    return result.column_ledger == expected_columns and all(
        ledger.absolute_error.as_fraction() <= error_bound
        for ledger in expected_columns
    )


class FiniteSetSystem(StrictModel):
    """A finite ground set [n] and a family of subsets over it.

    Each subset is a tuple of distinct element indices in 0..n-1. The
    ground set size ``n`` bounds the indices that may appear in any
    subset. An empty family is permitted.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A finite ground set of size `ground_set_size` and up to "
                f"{MAX_SETS} subsets given as strictly increasing index tuples. "
                f"The optimum operation admits ground sets up to {MAX_GROUND_SET} "
                "elements and pairs a bounded HiGHS MILP incumbent search with "
                "an exact pseudo-boolean optimality proof; the combined solver "
                "budget bounds each request."
            )
        }
    )

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET, strict=True)
    sets: tuple[tuple[int, ...], ...] = Field(max_length=MAX_SETS)

    @model_validator(mode="after")
    def require_valid_sets(self) -> Self:
        for subset in self.sets:
            seen: set[int] = set()
            for element in subset:
                if not (0 <= element < self.ground_set_size):
                    raise _validation_error(
                        "subset_element_out_of_range",
                        "subset element must be in 0..ground_set_size-1",
                    )
                if element in seen:
                    raise _validation_error(
                        "subset_elements_not_distinct",
                        "subset elements must be distinct",
                    )
                seen.add(element)
        return self


class DiscrepancyEvalRequest(StrictModel):
    """Evaluate the signed sums and maximum imbalance of a coloring."""

    set_system: FiniteSetSystem
    coloring: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_coloring(self) -> Self:
        if len(self.coloring) != self.set_system.ground_set_size:
            raise _validation_error(
                "coloring_length_mismatch",
                "coloring length must equal ground_set_size",
            )
        for value in self.coloring:
            if value not in (-1, 1):
                raise _validation_error(
                    "coloring_not_signed_binary", "coloring values must be +1 or -1"
                )
        return self


class DiscrepancyEvalResult(StrictModel):
    """The signed sum on every set and the maximum absolute imbalance."""

    signed_sums: tuple[int, ...]
    max_absolute_imbalance: int = Field(ge=0, strict=True)


class DiscrepancyOptimumRequest(StrictModel):
    """Search for a coloring minimizing the maximum discrepancy."""

    set_system: FiniteSetSystem


def _feasibility_outcome(
    set_system: FiniteSetSystem, allowed: int
) -> Literal["sat", "unsat", "unknown"]:
    """Decide exactly whether a coloring with imbalance at most ``allowed`` exists.

    One pseudo-boolean satisfiability check over the binary color bits with
    the maintained Z3 backend: ``unsat`` is an exact infeasibility proof and
    is the only outcome that may support a lower bound; ``sat`` supplies a
    witness; an exhausted proof budget, unavailable backend, or backend
    failure reports ``unknown`` so validation fails closed rather than
    accepting an unproven claim.
    """

    n = set_system.ground_set_size
    if n == 0:
        return "sat" if allowed >= 0 else "unsat"
    if allowed < 0:
        return "unsat"

    try:
        import z3  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return "unknown"
    try:
        solver = z3.Solver()
        solver.set(timeout=max(1, MAX_OPTIMUM_PROOF_MILLISECONDS))
        bits = [z3.Bool(f"b_{index}") for index in range(n)]
        signed_sums = [
            z3.Sum([z3.If(bits[element], 1, -1) for element in subset])
            for subset in set_system.sets
        ]
        for signed_sum in signed_sums:
            solver.add(signed_sum <= allowed)
            solver.add(signed_sum >= -allowed)
        status = solver.check()
    except z3.Z3Exception:
        return "unknown"
    if status == z3.sat:
        return "sat"
    if status == z3.unsat:
        return "unsat"
    return "unknown"


class DiscrepancyOptimumResult(StrictModel):
    """A proven-minimum coloring or an exhausted solver budget.

    Source-bound on its set system: ``OPTIMAL`` carries the exact minimum
    discrepancy and one witnessing coloring. Deserialization validates only
    the retained source and the witness's attained discrepancy. Independently
    supplied optimality claims are replayed by
    :func:`verify_discrepancy_optimum_result` under its explicit proof budget.
    ``BUDGET_EXCEEDED`` makes no mathematical claim: it carries neither a
    coloring nor a discrepancy value.
    """

    set_system: FiniteSetSystem
    status: Literal["OPTIMAL", "BUDGET_EXCEEDED", "EXECUTION_FAILED"]
    optimal_coloring: tuple[int, ...] = Field(default=())
    optimal_discrepancy: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def bind_optimal_coloring(self) -> Self:
        if self.status in ("BUDGET_EXCEEDED", "EXECUTION_FAILED"):
            if self.optimal_coloring or self.optimal_discrepancy is not None:
                raise _validation_error(
                    "incomplete_result_carries_claim",
                    f"a {self.status} result must not carry a coloring or optimum",
                )
            return self
        if self.optimal_discrepancy is None:
            raise _validation_error(
                "optimal_discrepancy_missing",
                "an OPTIMAL result requires its discrepancy value",
            )
        if len(self.optimal_coloring) != self.set_system.ground_set_size:
            raise _validation_error(
                "optimal_coloring_length_mismatch",
                "coloring length must equal the ground-set size",
            )
        if any(value not in (-1, 1) for value in self.optimal_coloring):
            raise _validation_error(
                "optimal_coloring_not_signed_binary", "coloring values must be +1 or -1"
            )
        maximum = max(
            (
                abs(sum(self.optimal_coloring[element] for element in subset))
                for subset in self.set_system.sets
            ),
            default=0,
        )
        if maximum != self.optimal_discrepancy:
            raise _validation_error(
                "optimal_discrepancy_mismatch",
                "the reported discrepancy must be the exact maximum imbalance "
                "of the returned coloring",
            )
        return self


def verify_discrepancy_optimum_result(result: DiscrepancyOptimumResult) -> bool:
    """Replay one independently supplied positive optimum within the proof cap."""

    if result.status != "OPTIMAL":
        return True
    if result.optimal_discrepancy is None:
        return False
    if result.optimal_discrepancy == 0:
        return True
    return (
        _feasibility_outcome(result.set_system, result.optimal_discrepancy - 1)
        == "unsat"
    )


def _proven_optimal_result(
    set_system: FiniteSetSystem,
    optimal_coloring: tuple[int, ...],
    optimal_discrepancy: int,
) -> DiscrepancyOptimumResult:
    """Build a proven-optimal result after one producing incumbent solve.

    Direct construction is permitted after the owner kernel has established
    witness feasibility and the exact lower-bound proof. Independently
    supplied results use :func:`verify_discrepancy_optimum_result`.
    """

    return DiscrepancyOptimumResult.model_construct(
        set_system=set_system,
        status="OPTIMAL",
        optimal_coloring=optimal_coloring,
        optimal_discrepancy=optimal_discrepancy,
    )


def _budget_exceeded_result(set_system: FiniteSetSystem) -> DiscrepancyOptimumResult:
    """Build the typed incomplete outcome from one exhausted producing solve.

    As with ``_proven_optimal_result``, the producing solve's own answer is
    carried unclaimed; replay stays reserved for independently supplied
    results via ``bind_optimal_coloring``.
    """

    return DiscrepancyOptimumResult.model_construct(
        set_system=set_system,
        status="BUDGET_EXCEEDED",
        optimal_coloring=(),
        optimal_discrepancy=None,
    )


def _execution_failed_result(set_system: FiniteSetSystem) -> DiscrepancyOptimumResult:
    """Build the typed non-mathematical outcome from a backend failure.

    Same claim-free shape as ``_budget_exceeded_result``: the producing
    solve's answer is carried unclaimed and replay stays reserved for
    independently supplied results via ``bind_optimal_coloring``.
    """

    return DiscrepancyOptimumResult.model_construct(
        set_system=set_system,
        status="EXECUTION_FAILED",
        optimal_coloring=(),
        optimal_discrepancy=None,
    )


__all__ = [
    "MAX_COLUMN_INCIDENCES",
    "MAX_GROUND_SET",
    "MAX_MONITORED_COLUMNS",
    "MAX_ROUNDING_COORDINATES",
    "MAX_ROUNDING_INTERMEDIATE_DIGITS",
    "MAX_ROUNDING_RATIONAL_DIGITS",
    "MAX_ROUNDING_RESULT_RATIONAL_DIGITS",
    "MAX_ROUNDING_ROWS",
    "MAX_ROUNDING_WORK",
    "MAX_SETS",
    "DiscrepancyEvalRequest",
    "DiscrepancyEvalResult",
    "DiscrepancyOptimumRequest",
    "DiscrepancyOptimumResult",
    "FiniteSetSystem",
    "HardConstraintRoundingRequest",
    "HardConstraintRoundingResult",
    "HardConstraintRoundingSource",
    "HardConstraintRow",
    "HardConstraintRowLedger",
    "MonitoredColumn",
    "MonitoredColumnLedger",
    "verify_discrepancy_optimum_result",
]
