"""Exact checking of supplied rational primal-dual pairs, without search."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization._arithmetic import rational_dot
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    RationalLinearConstraint,
    RationalLinearObjective,
    RationalLinearProgramVariable,
)
from jacobian.math.optimization._models import (
    StandardFormRationalLinearProgram,
    _prepare_raw_rational_vector,
)

type LinearProgram = (
    GeneralFormRationalLinearProgram | StandardFormRationalLinearProgram
)


class RationalLinearOptimalityCandidate(StrictModel):
    """Candidates in the source axes and source objective sign convention.

    For minimization, GE/lower multipliers are nonnegative and LE/upper
    multipliers nonpositive; signs reverse for maximization. EQ multipliers
    are free. Absent bounds require zero multipliers. Standard form means
    minimization with equality rows and nonnegative variables.
    """

    program: LinearProgram
    primal_candidate: tuple[CanonicalRational, ...] = Field(max_length=32)
    constraint_dual: tuple[CanonicalRational, ...] = Field(max_length=64)
    lower_bound_dual: tuple[CanonicalRational, ...] = Field(max_length=32)
    upper_bound_dual: tuple[CanonicalRational, ...] = Field(max_length=32)

    @model_validator(mode="before")
    @classmethod
    def bound_candidate_encoding(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        prepared = dict(value)
        for field, limit in (
            ("primal_candidate", 32),
            ("constraint_dual", 64),
            ("lower_bound_dual", 32),
            ("upper_bound_dual", 32),
        ):
            prepared[field] = _prepare_raw_rational_vector(
                prepared.get(field),
                maximum_length=limit,
                label=field,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            )
        return canonicalize_json_containers(prepared)

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        n = len(self.program.variables)
        m = (
            len(self.program.constraints)
            if isinstance(self.program, GeneralFormRationalLinearProgram)
            else len(self.program.rhs)
        )
        if len(self.constraint_dual) != m or any(
            len(vector) != n
            for vector in (
                self.primal_candidate,
                self.lower_bound_dual,
                self.upper_bound_dual,
            )
        ):
            raise ValueError(
                "candidate vectors must match the source variable and constraint axes"
            )
        return self


class RationalLinearOptimalityRequest(RationalLinearOptimalityCandidate):
    """Wire projection of a supplied source-coordinate primal-dual pair."""


class RationalLinearOptimalityResult(StrictModel):
    """Whether this supplied pair establishes optimality of its retained LP.

    A false result rejects the candidate, not the source program. Parsing
    retains the authored claim structurally; the checking operation owns
    feasibility and objective arithmetic.
    """

    candidate: RationalLinearOptimalityCandidate
    is_optimal: bool
    primal_feasible: bool
    dual_feasible: bool
    objectives_equal: bool
    primal_objective: CanonicalRational
    dual_objective: CanonicalRational
    objective_gap: CanonicalRational
    primal_residuals: tuple[CanonicalRational, ...] = Field(max_length=64)
    stationarity_residuals: tuple[CanonicalRational, ...] = Field(max_length=32)
    failed_conditions: tuple[str, ...] = Field(max_length=8)


def _general_source(program: LinearProgram) -> GeneralFormRationalLinearProgram:
    if isinstance(program, GeneralFormRationalLinearProgram):
        return program
    zero = CanonicalRational(num=0, den=1)
    return GeneralFormRationalLinearProgram(
        variables=tuple(
            RationalLinearProgramVariable(name=v, lower_bound=zero)
            for v in program.variables
        ),
        objective=RationalLinearObjective(
            sense="MINIMIZE", coefficients=program.objective
        ),
        constraints=tuple(
            RationalLinearConstraint(
                label=f"row_{i}", coefficients=row, relation="EQ", rhs=rhs
            )
            for i, (row, rhs) in enumerate(
                zip(program.coefficients, program.rhs, strict=True)
            )
        ),
    )


def _admit(
    candidate: RationalLinearOptimalityCandidate,
    source: GeneralFormRationalLinearProgram,
) -> None:
    scalars = (
        *source.objective.coefficients,
        *(v for row in source.constraints for v in (*row.coefficients, row.rhs)),
        *(
            b
            for var in source.variables
            for b in (var.lower_bound, var.upper_bound)
            if b is not None
        ),
        *candidate.primal_candidate,
        *candidate.constraint_dual,
        *candidate.lower_bound_dual,
        *candidate.upper_bound_dual,
    )
    digits = max(
        max(
            len(format_canonical_integer(abs(v.num))),
            len(format_canonical_integer(v.den)),
        )
        for v in scalars
    )
    n, m = len(source.variables), len(source.constraints)
    # The objective gap combines n primal and m+2*n dual products.
    # Multiplying all denominators bounds unreduced intermediate height;
    # numerator height adds one product's height and summation carries.
    result_digits = 2 * digits * (m + 3 * n + 4) + 8
    # Shared denominators, especially integral data, should not pay for
    # unrelated denominators at every term. Every product denominator
    # divides the square of the product of distinct input denominators.
    denominator_digits = sum(
        len(format_canonical_integer(den))
        for den in {v.den for v in scalars}
        if den != 1
    )
    result_digits = min(
        result_digits, 2 * denominator_digits + 2 * digits + len(str(m + 3 * n + 1)) + 4
    )
    scalar_updates = 8 * (m * n + m + n + 1)
    retained_digits = 2 * result_digits * (m + n + 3) + 2 * digits * len(scalars)
    if (
        result_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or retained_digits > 8 * 1024 * 1024
        or scalar_updates > 20_000
    ):
        raise OperationResourceAdmissionError(
            location=("program",),
            code="optimization.linear.optimality_check_bound",
            message=f"candidate check predicts {scalar_updates} scalar updates, {result_digits} rational digits and {retained_digits} retained scalar digits; limits 20000, {MAX_CANONICAL_RATIONAL_DIGITS}, 8388608",
        )


def _check(
    candidate: RationalLinearOptimalityCandidate,
    source: GeneralFormRationalLinearProgram,
) -> RationalLinearOptimalityResult:
    x, y, lower, upper = (
        tuple(v.as_fraction() for v in vector)
        for vector in (
            candidate.primal_candidate,
            candidate.constraint_dual,
            candidate.lower_bound_dual,
            candidate.upper_bound_dual,
        )
    )
    c = tuple(v.as_fraction() for v in source.objective.coefficients)
    rows = tuple(
        tuple(v.as_fraction() for v in row.coefficients) for row in source.constraints
    )
    rhs = tuple(row.rhs.as_fraction() for row in source.constraints)
    residuals = tuple(
        rational_dot(row, x) - b for row, b in zip(rows, rhs, strict=True)
    )
    row_feasible = all(
        r == 0 if row.relation == "EQ" else r <= 0 if row.relation == "LE" else r >= 0
        for row, r in zip(source.constraints, residuals, strict=True)
    )
    bound_feasible = all(
        (var.lower_bound is None or value >= var.lower_bound.as_fraction())
        and (var.upper_bound is None or value <= var.upper_bound.as_fraction())
        for var, value in zip(source.variables, x, strict=True)
    )
    sign = 1 if source.objective.sense == "MINIMIZE" else -1
    row_signs = all(
        row.relation == "EQ"
        or (sign * value <= 0 if row.relation == "LE" else sign * value >= 0)
        for row, value in zip(source.constraints, y, strict=True)
    )
    bound_signs = all(
        (lo == 0 if var.lower_bound is None else sign * lo >= 0)
        and (hi == 0 if var.upper_bound is None else sign * hi <= 0)
        for var, lo, hi in zip(source.variables, lower, upper, strict=True)
    )
    stationarity = tuple(
        coefficient
        - sum((row[j] * value for row, value in zip(rows, y, strict=True)), Fraction())
        - lower[j]
        - upper[j]
        for j, coefficient in enumerate(c)
    )
    primal_objective = rational_dot(c, x)
    dual_objective = rational_dot(rhs, y) + sum(
        (
            multiplier * bound.as_fraction()
            for var, lo, hi in zip(source.variables, lower, upper, strict=True)
            for bound, multiplier in ((var.lower_bound, lo), (var.upper_bound, hi))
            if bound is not None
        ),
        Fraction(),
    )
    conditions = {
        "primal_constraints": row_feasible,
        "primal_bounds": bound_feasible,
        "dual_constraint_signs": row_signs,
        "dual_bound_signs": bound_signs,
        "stationarity": all(value == 0 for value in stationarity),
        "objective_equality": primal_objective == dual_objective,
    }
    request_checkpoint("after rational primal-dual checking")
    return RationalLinearOptimalityResult.model_construct(
        candidate=candidate,
        is_optimal=all(conditions.values()),
        primal_feasible=row_feasible and bound_feasible,
        dual_feasible=row_signs and bound_signs and conditions["stationarity"],
        objectives_equal=conditions["objective_equality"],
        primal_objective=CanonicalRational.from_fraction(primal_objective),
        dual_objective=CanonicalRational.from_fraction(dual_objective),
        objective_gap=CanonicalRational.from_fraction(
            primal_objective - dual_objective
        ),
        primal_residuals=tuple(CanonicalRational.from_fraction(v) for v in residuals),
        stationarity_residuals=tuple(
            CanonicalRational.from_fraction(v) for v in stationarity
        ),
        failed_conditions=tuple(
            name for name, passed in conditions.items() if not passed
        ),
    )


def check_linear_optimality(
    candidate: RationalLinearOptimalityCandidate,
) -> RationalLinearOptimalityResult:
    """Check feasibility, multiplier signs, stationarity and objective equality.

    This accepts the same original coordinates as the solvers but performs
    only matrix-vector arithmetic. It never normalizes slack/sign-split
    columns or invokes basis search. Weak duality establishes optimality
    exactly when the supplied feasible primal and dual objectives coincide.
    """
    source = _general_source(candidate.program)
    _admit(candidate, source)
    execution = current_request_execution()
    with request_execution(time.monotonic()) if execution is None else nullcontext():
        execution = current_request_execution()
        assert execution is not None
        deadline = execution.started_at + 60.0
        if execution.deadline is not None:
            deadline = min(deadline, execution.deadline)
        bind_request_deadline(deadline)
        request_checkpoint("before rational primal-dual checking")
        result = _check(candidate, source)
        request_checkpoint("after rational primal-dual result construction")
        return result


__all__ = [
    "RationalLinearOptimalityCandidate",
    "RationalLinearOptimalityRequest",
    "RationalLinearOptimalityResult",
    "check_linear_optimality",
]
