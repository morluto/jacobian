"""Admission and request execution for exact rational linear optimization."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from math import comb
from time import monotonic

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_BACKEND_STATES,
    StandardFormRationalLinearProgram,
    _active_equations,
    _result_digit_bound,
)

LINEAR_PROGRAM_WALL_SECONDS = 600


@contextmanager
def linear_execution() -> Iterator[None]:
    """Share one deadline across normalization, worker execution, and mapping."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(monotonic()), linear_execution():
            yield
        return
    deadline = execution.started_at + LINEAR_PROGRAM_WALL_SECONDS
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("linear-program admission")
    yield
    request_checkpoint("linear-program result construction")


@dataclass(frozen=True)
class LinearAdmission:
    columns: tuple[int, ...]
    result_digits: int
    components: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()


def _constraint_components(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    row_columns = [
        tuple(j for j, value in enumerate(row) if value.num != 0)
        for row in program.coefficients
    ]
    column_rows: dict[int, list[int]] = {}
    for row_index, source_columns in enumerate(row_columns):
        for column in source_columns:
            column_rows.setdefault(column, []).append(row_index)
    remaining = set(column_rows)
    components = []
    while remaining:
        pending = [min(remaining)]
        rows: set[int] = set()
        columns: set[int] = set()
        while pending:
            column = pending.pop()
            if column not in remaining:
                continue
            remaining.remove(column)
            columns.add(column)
            for row_index in column_rows[column]:
                if row_index not in rows:
                    rows.add(row_index)
                    pending.extend(row_columns[row_index])
        components.append((tuple(sorted(rows)), tuple(sorted(columns))))
    return tuple(components)


def _component_state_bound(rows: int, columns: int) -> int:
    maximum_rank = min(columns, rows)
    return max(
        (comb(columns + 1, rank) for rank in range(maximum_rank + 1)),
        default=1,
    )


def admit_linear_program(program: StandardFormRationalLinearProgram) -> LinearAdmission:
    """Admit canonical input and complete exact certificate representation.

    The model owns fixed dimension, source-height, and matrix-cardinality limits.
    PPL does not enumerate bases, but its primal, dual, and Farkas polyhedra still
    have a finite combinatorial state envelope. Admission bounds that envelope
    before constructing backend polyhedra; the worker deadline is only a safety
    limit inside already-admitted work.
    """
    columns = tuple(
        j
        for j in range(len(program.variables))
        if any(row[j].num != 0 for row in program.coefficients)
    )
    digits = _result_digit_bound(program)
    rows = len(_active_equations(program))
    components = _constraint_components(program)
    backend_states = sum(
        _component_state_bound(len(component_rows), len(component_columns))
        for component_rows, component_columns in components
    )
    quantities = (
        f"normalized_columns={len(program.variables)}, "
        f"active_columns={len(columns)}, "
        f"normalized_rows={len(program.rhs)}, active_rows={rows}, "
        f"backend_state_estimate={backend_states}, "
        f"backend_state_limit={MAX_LINEAR_PROGRAM_BACKEND_STATES}, "
        f"result_digits={digits}, "
        f"result_digit_limit={MAX_CANONICAL_RATIONAL_DIGITS}"
    )
    for reason, measured, limit in (
        ("result_height", digits, MAX_CANONICAL_RATIONAL_DIGITS),
        ("backend_state_bound", backend_states, MAX_LINEAR_PROGRAM_BACKEND_STATES),
    ):
        if measured > limit:
            raise OperationResourceAdmissionError(
                location=("program",),
                code=f"optimization.linear.{reason}",
                message=f"Exact LP {reason} exceeded: {quantities}.",
            )
    return LinearAdmission(
        columns=columns,
        result_digits=digits,
        components=components,
    )
