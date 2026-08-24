"""Typed wire contracts for finite set-system discrepancy operations."""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_GROUND_SET = 64
MAX_SETS = 1_000
# The optimum operation encodes minimum-discrepancy search as an exact
# integer program (Z3 Optimize over {±1} variables with one shared
# objective D); a sat answer is a proven optimum, and an exhausted budget
# returns BUDGET_EXCEEDED without any mathematical claim.
MAX_OPTIMUM_SOLVER_MILLISECONDS = 30_000

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
        raise ValueError(f"{owner} coordinate indices must be strict integers")
    if any(not 0 <= index < coordinate_count for index in elements):
        raise ValueError(f"{owner} coordinate indices must be in 0..coordinate_count-1")
    if any(left >= right for left, right in itertools.pairwise(elements)):
        raise ValueError(f"{owner} coordinate indices must be strictly increasing")


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
            raise ValueError("coordinate labels must be unique")
        if len(self.values) != coordinate_count:
            raise ValueError("values must align with the coordinate axis")
        if len({row.label for row in self.rows}) != len(self.rows):
            raise ValueError("row labels must be unique")
        if len({column.label for column in self.columns}) != len(self.columns):
            raise ValueError("column labels must be unique")

        fractions: list[Fraction] = []
        for value in self.values:
            require_bounded_rational(
                value,
                max_digits=MAX_ROUNDING_RATIONAL_DIGITS,
                label="rounding source rational",
            )
            fraction = value.as_fraction()
            if not 0 <= fraction <= 1:
                raise ValueError("rounding source values must lie in [0, 1]")
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
            raise ValueError("hard rows must partition every coordinate exactly once")

        total_incidences = 0
        for column in self.columns:
            _require_canonical_subset(
                column.coordinates,
                coordinate_count=coordinate_count,
                owner="column",
            )
            total_incidences += len(column.coordinates)
            if total_incidences > MAX_COLUMN_INCIDENCES:
                raise ValueError(
                    f"monitored column incidences exceed {MAX_COLUMN_INCIDENCES}"
                )

        fractional_count = sum(0 < value < 1 for value in fractions)
        scan_work = fractional_count * (coordinate_count + total_incidences)
        elimination_work = (fractional_count * (fractional_count + 1) // 2) ** 2
        if scan_work + elimination_work > MAX_ROUNDING_WORK:
            raise ValueError(
                "rounding work bound exceeded by fractional support and incidences"
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
            raise ValueError("rounding intermediate rational-height bound exceeded")

        row_sums = [
            _sum_selected_fractions(fractions, row.coordinates) for row in self.rows
        ]
        if any(row_sum.denominator != 1 for row_sum in row_sums):
            raise ValueError("every hard row must have an integral source sum")
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
            raise ValueError("rounding exact result-size bound exceeded")
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
    """A source-bound binary rounding with replayable hard-row and error ledgers."""

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
    def replay_rounding_invariants(self) -> Self:
        coordinate_count = len(self.source.coordinate_labels)
        if len(self.rounded_values) != coordinate_count:
            raise ValueError("rounded values must align with the coordinate axis")
        if any(
            type(value) is not int or value not in (0, 1)
            for value in self.rounded_values
        ):
            raise ValueError("rounded values must be strict binary integers")

        source_values = tuple(value.as_fraction() for value in self.source.values)
        expected_rows: list[HardConstraintRowLedger] = []
        for row in self.source.rows:
            source_sum = sum(
                (source_values[index] for index in row.coordinates), Fraction()
            )
            rounded_sum = sum(self.rounded_values[index] for index in row.coordinates)
            if source_sum != rounded_sum:
                raise ValueError("rounded values must preserve every hard row sum")
            expected_rows.append(
                HardConstraintRowLedger(
                    row_label=row.label,
                    source_sum=_as_canonical_rational(source_sum),
                    rounded_sum=rounded_sum,
                )
            )
        if tuple(expected_rows) != self.row_ledger:
            raise ValueError("row ledger must replay exactly from source and rounding")

        incidences = [0] * coordinate_count
        for column in self.source.columns:
            for index in column.coordinates:
                incidences[index] += 1
        expected_incidence = max(incidences, default=0)
        expected_bound = 4 * expected_incidence
        if self.maximum_column_incidence != expected_incidence:
            raise ValueError("maximum column incidence must be derived from the source")
        if self.column_error_bound != expected_bound:
            raise ValueError("column error bound must equal four times the incidence")

        expected_columns: list[MonitoredColumnLedger] = []
        for column in self.source.columns:
            source_sum = sum(
                (source_values[index] for index in column.coordinates), Fraction()
            )
            rounded_sum = sum(
                self.rounded_values[index] for index in column.coordinates
            )
            signed_error = Fraction(rounded_sum) - source_sum
            absolute_error = abs(signed_error)
            if absolute_error > expected_bound:
                raise ValueError("monitored column error exceeds the derived 4d bound")
            expected_columns.append(
                MonitoredColumnLedger(
                    column_label=column.label,
                    source_sum=_as_canonical_rational(source_sum),
                    rounded_sum=rounded_sum,
                    signed_error=_as_canonical_rational(signed_error),
                    absolute_error=_as_canonical_rational(absolute_error),
                )
            )
        if tuple(expected_columns) != self.column_ledger:
            raise ValueError(
                "column ledger must replay exactly from source and rounding"
            )
        return self


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
                "elements and encodes the minimum-discrepancy search as an exact "
                "integer program; the solver budget bounds each request."
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
                    raise ValueError("subset element must be in 0..ground_set_size-1")
                if element in seen:
                    raise ValueError("subset elements must be distinct")
                seen.add(element)
        return self


class DiscrepancyEvalRequest(StrictModel):
    """Evaluate the signed sums and maximum imbalance of a coloring."""

    set_system: FiniteSetSystem
    coloring: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_coloring(self) -> Self:
        if len(self.coloring) != self.set_system.ground_set_size:
            raise ValueError(
                "coloring length must equal ground_set_size",
            )
        for value in self.coloring:
            if value not in (-1, 1):
                raise ValueError("coloring values must be +1 or -1")
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
    """Run one bounded exact feasibility check for imbalance at most ``allowed``.

    Mirrors the optimum program's constraints with a fixed target instead of
    a minimized objective. Only an explicit ``unsat`` may support a lower
    bound; an exhausted budget reports ``unknown`` so validation fails
    closed rather than accepting an unproven claim.
    """

    import z3  # type: ignore[import-untyped]

    solver = z3.Solver()
    solver.set("timeout", MAX_OPTIMUM_SOLVER_MILLISECONDS)
    variables = [z3.Int(f"x_{index}") for index in range(set_system.ground_set_size)]
    solver.add(*(z3.Or(value == 1, value == -1) for value in variables))
    for subset in set_system.sets:
        signed_sum = (
            z3.Sum([variables[element] for element in subset])
            if subset
            else z3.IntVal(0)
        )
        solver.add(signed_sum <= allowed, -signed_sum <= allowed)
    outcome = solver.check()
    if outcome == z3.sat:
        return "sat"
    if outcome == z3.unsat:
        return "unsat"
    return "unknown"


class DiscrepancyOptimumResult(StrictModel):
    """A proven-minimum coloring or an exhausted solver budget.

    Source-bound on its set system: ``OPTIMAL`` carries the exact minimum
    discrepancy and one witnessing coloring. Deserialization replays the
    witness against the retained system and independently re-establishes
    the lower bound: zero is definitional, and any positive claimed optimum
    must be backed by an explicit unsat of the exact feasibility program
    that asks for a coloring of imbalance at most one less.
    ``BUDGET_EXCEEDED`` makes no mathematical claim: it carries neither a
    coloring nor a discrepancy value.
    """

    set_system: FiniteSetSystem
    status: Literal["OPTIMAL", "BUDGET_EXCEEDED"]
    optimal_coloring: tuple[int, ...] = Field(default=())
    optimal_discrepancy: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def bind_optimal_coloring(self) -> Self:
        if self.status == "BUDGET_EXCEEDED":
            if self.optimal_coloring or self.optimal_discrepancy is not None:
                raise ValueError(
                    "a BUDGET_EXCEEDED result must not carry a coloring or optimum"
                )
            return self
        if self.optimal_discrepancy is None:
            raise ValueError("an OPTIMAL result requires its discrepancy value")
        if len(self.optimal_coloring) != self.set_system.ground_set_size:
            raise ValueError("coloring length must equal the ground-set size")
        if any(value not in (-1, 1) for value in self.optimal_coloring):
            raise ValueError("coloring values must be +1 or -1")
        maximum = max(
            (
                abs(sum(self.optimal_coloring[element] for element in subset))
                for subset in self.set_system.sets
            ),
            default=0,
        )
        if maximum != self.optimal_discrepancy:
            raise ValueError(
                "the reported discrepancy must be the exact maximum imbalance "
                "of the returned coloring"
            )
        self._require_lower_bound()
        return self

    def _require_lower_bound(self) -> None:
        """Re-establish minimality; only a proven lower bound may pass."""

        assert self.optimal_discrepancy is not None
        if self.optimal_discrepancy == 0:
            return
        outcome = _feasibility_outcome(self.set_system, self.optimal_discrepancy - 1)
        if outcome == "unsat":
            return
        if outcome == "sat":
            raise ValueError(
                "a coloring with smaller imbalance exists; the claimed "
                "optimum is not minimal"
            )
        raise ValueError(
            "claimed optimality was not established within the replay budget"
        )


def _proven_optimal_result(
    set_system: FiniteSetSystem,
    optimal_coloring: tuple[int, ...],
    optimal_discrepancy: int,
) -> DiscrepancyOptimumResult:
    """Build a proven-optimal result from one producing Optimize solve.

    Direct construction from the producing solve skips result replay so one
    declared budget covers all solver work; independently supplied results
    always validate through ``bind_optimal_coloring``, which replays the
    witness and re-establishes the lower bound.
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
]
