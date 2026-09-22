"""Admission and request execution for exact rational linear optimization."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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


def admit_linear_program(program: StandardFormRationalLinearProgram) -> LinearAdmission:
    """Admit canonical input and complete exact certificate representation.

    The model owns fixed dimension, source-height, and matrix-cardinality limits.
    PPL replaces combinatorial basis enumeration, so basis-family cardinality is
    no longer a measure of executed work or a valid reason to reject a request.
    """
    columns = tuple(
        j
        for j in range(len(program.variables))
        if any(row[j].num != 0 for row in program.coefficients)
    )
    digits = _result_digit_bound(program)
    if digits > MAX_CANONICAL_RATIONAL_DIGITS:
        rows = len(_active_equations(program))
        quantities = (
            f"normalized_columns={len(program.variables)}, "
            f"active_columns={len(columns)}, "
            f"normalized_rows={len(program.rhs)}, active_rows={rows}, "
            f"result_digits={digits}, "
            f"result_digit_limit={MAX_CANONICAL_RATIONAL_DIGITS}"
        )
        raise OperationResourceAdmissionError(
            location=("program",),
            code="optimization.linear.result_height",
            message=f"Exact LP result_height exceeded: {quantities}.",
        )
    return LinearAdmission(columns=columns, result_digits=digits)
