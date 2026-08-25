"""Canonical source values and replayed outcomes for general rational LPs."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.optimization._arithmetic import rational_dot
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_RESULT_BYTES,
    RationalLinearProgramStatus,
    _bound_raw_rational,
    _prepare_raw_rational_vector,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"general_linear_program.{reason}", message)


type RationalLinearRelation = Literal["LE", "EQ", "GE"]
type RationalObjectiveSense = Literal["MINIMIZE", "MAXIMIZE"]

MAX_GENERAL_LINEAR_PROGRAM_VARIABLES = 32
MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS = 64


def _prepare_raw_general_program(value: object) -> object:
    """Bound and tuple-normalize a general LP before nested rational parsing."""

    if not isinstance(value, Mapping):
        return value
    prepared = dict(value)
    variables = prepared.get("variables")
    if isinstance(variables, (list, tuple)):
        if len(variables) > MAX_GENERAL_LINEAR_PROGRAM_VARIABLES:
            raise ValueError(
                "general linear-program variables exceed the "
                f"{MAX_GENERAL_LINEAR_PROGRAM_VARIABLES}-entry bound"
            )
        for variable in variables:
            if not isinstance(variable, Mapping):
                continue
            name = variable.get("name")
            if isinstance(name, str) and len(name) > 64:
                raise ValueError(
                    "general linear-program variable name exceeds the "
                    "64-character bound"
                )
            _bound_raw_rational(
                variable.get("lower_bound"),
                label="general linear-program lower bound",
            )
            _bound_raw_rational(
                variable.get("upper_bound"),
                label="general linear-program upper bound",
            )
        prepared["variables"] = tuple(variables)
    objective = prepared.get("objective")
    if isinstance(objective, Mapping):
        objective_prepared = dict(objective)
        objective_prepared["coefficients"] = _prepare_raw_rational_vector(
            objective_prepared.get("coefficients"),
            maximum_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
            label="general linear-program objective",
        )
        prepared["objective"] = objective_prepared
    constraints = prepared.get("constraints")
    if isinstance(constraints, (list, tuple)):
        if len(constraints) > MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS:
            raise ValueError(
                "general linear-program constraints exceed the "
                f"{MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS}-entry bound"
            )
        normalized_constraints: list[object] = []
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                normalized_constraints.append(constraint)
                continue
            constraint_prepared = dict(constraint)
            label = constraint_prepared.get("label")
            if isinstance(label, str) and len(label) > 64:
                raise ValueError(
                    "general linear-program constraint label exceeds the "
                    "64-character bound"
                )
            constraint_prepared["coefficients"] = _prepare_raw_rational_vector(
                constraint_prepared.get("coefficients"),
                maximum_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                label="general linear-program constraint coefficients",
            )
            _bound_raw_rational(
                constraint_prepared.get("rhs"),
                label="general linear-program constraint rhs",
            )
            normalized_constraints.append(constraint_prepared)
        prepared["constraints"] = tuple(normalized_constraints)
    return prepared


class RationalLinearProgramVariable(StrictModel):
    """One named original-coordinate LP variable and its closed bounds."""

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Unique identifier of this original-coordinate variable.",
    )
    lower_bound: CanonicalRational | None = Field(
        default=None,
        description="Finite closed lower bound, or null when the variable is free below.",
    )
    upper_bound: CanonicalRational | None = Field(
        default=None,
        description="Finite closed upper bound, or null when the variable is free above.",
    )

    @model_validator(mode="after")
    def require_canonical_bounds(self) -> Self:
        if not (self.name[0].isalpha() or self.name[0] == "_") or any(
            not (character.isalnum() or character == "_") for character in self.name
        ):
            raise _validation_error(
                "variable_identifier",
                "general linear-program variable names must be identifiers",
            )
        for bound in (self.lower_bound, self.upper_bound):
            if bound is not None:
                try:
                    require_bounded_rational(
                        bound,
                        max_digits=128,
                        label="general linear-program input",
                    )
                except ValueError as error:
                    raise _validation_error("bounded_rational", str(error)) from error
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound.as_fraction() > self.upper_bound.as_fraction()
        ):
            raise _validation_error(
                "bound_order", "a variable lower bound cannot exceed its upper bound"
            )
        return self


class RationalLinearObjective(StrictModel):
    """A rational linear objective in the declared original variable order."""

    sense: RationalObjectiveSense = Field(
        description="MINIMIZE or MAXIMIZE the exact linear objective.",
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
        description="Exact coefficients in the enclosing program variable order.",
    )

    @model_validator(mode="after")
    def require_bounded_coefficients(self) -> Self:
        for coefficient in self.coefficients:
            try:
                require_bounded_rational(
                    coefficient,
                    max_digits=128,
                    label="general linear-program input",
                )
            except ValueError as error:
                raise _validation_error("bounded_rational", str(error)) from error
        return self


class RationalLinearConstraint(StrictModel):
    """One labeled exact rational affine relation in source coordinates."""

    label: str = Field(
        min_length=1,
        max_length=64,
        description="Unique source identity for this relation, not display-only text.",
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
        description="Exact row coefficients in the enclosing program variable order.",
    )
    relation: RationalLinearRelation = Field(
        description="LE means a*x <= rhs, EQ means equality, and GE means a*x >= rhs.",
    )
    rhs: CanonicalRational = Field(description="Exact right-hand-side rational.")

    @model_validator(mode="after")
    def require_bounded_coefficients(self) -> Self:
        for value in (*self.coefficients, self.rhs):
            try:
                require_bounded_rational(
                    value,
                    max_digits=128,
                    label="general linear-program input",
                )
            except ValueError as error:
                raise _validation_error("bounded_rational", str(error)) from error
        return self


class GeneralFormRationalLinearProgram(StrictModel):
    """Optimize an exact rational objective with rows and finite bounds.

    A null bound is absent; two null bounds make a variable free.  All rows and
    bounds are closed, so strict feasibility is expressed with a caller-owned
    explicit margin variable rather than an implicit strict relation.
    """

    variables: tuple[RationalLinearProgramVariable, ...] = Field(
        min_length=1,
        max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
        description="Ordered original-coordinate variables with explicit closed bounds.",
    )
    objective: RationalLinearObjective
    constraints: tuple[RationalLinearConstraint, ...] = Field(
        default=(),
        max_length=MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
        description="Ordered labeled LE, EQ, or GE rows in original coordinates.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_program(cls, value: object) -> object:
        try:
            return _prepare_raw_general_program(value)
        except ValueError as error:
            raise _validation_error("raw_input_bound", str(error)) from error

    @model_validator(mode="after")
    def require_dimensions_and_admission(self) -> Self:
        if len({variable.name for variable in self.variables}) != len(self.variables):
            raise _validation_error(
                "duplicate_variable",
                "general linear-program variable names must be unique",
            )
        if len({constraint.label for constraint in self.constraints}) != len(
            self.constraints
        ):
            raise _validation_error(
                "duplicate_constraint_label",
                "general linear-program constraint labels must be unique",
            )
        if len(self.objective.coefficients) != len(self.variables):
            raise _validation_error(
                "objective_length",
                "objective length must equal the variable count",
            )
        if any(
            len(constraint.coefficients) != len(self.variables)
            for constraint in self.constraints
        ):
            raise _validation_error(
                "constraint_row_length",
                "every constraint row must match the variable count",
            )
        # Import lazily: normalizing imports these public source types, while
        # this model owns the pre-backend work admission.
        from jacobian.math.optimization._general_normalization import (
            require_admitted_general_normalization,
        )

        try:
            require_admitted_general_normalization(self)
        except ValueError as error:
            raise _validation_error("normalization_admission", str(error)) from error
        return self


class GeneralRationalLinearProgramRequest(StrictModel):
    program: GeneralFormRationalLinearProgram


class GeneralRationalLinearProgramResult(StrictModel):
    """A source-bound exact outcome for a general-form rational LP.

    Dual multipliers use the source objective sense.  For minimization, LE
    row and upper-bound multipliers are nonpositive while GE and lower-bound
    multipliers are nonnegative; the signs reverse for maximization.  An
    infeasibility certificate instead has LE/nonnegative and GE/nonpositive
    row multipliers, and satisfies its source Farkas balance directly.
    """

    program: GeneralFormRationalLinearProgram
    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    primal_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS
    )
    constraint_slacks: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS
    )
    lower_bound_slacks: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    upper_bound_slacks: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    constraint_dual: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS
    )
    lower_bound_dual: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    upper_bound_dual: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    dual_objective: CanonicalRational | None = None
    stationarity_residuals: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    farkas_constraints: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS
    )
    farkas_lower_bounds: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    farkas_upper_bounds: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )
    recession_direction: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        try:
            prepared = dict(value)
            prepared["program"] = _prepare_raw_general_program(prepared.get("program"))
            vector_bounds = {
                "primal_candidate": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "primal_residuals": MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
                "constraint_slacks": MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
                "lower_bound_slacks": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "upper_bound_slacks": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "constraint_dual": MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
                "lower_bound_dual": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "upper_bound_dual": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "stationarity_residuals": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "farkas_constraints": MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
                "farkas_lower_bounds": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "farkas_upper_bounds": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
                "recession_direction": MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
            }
            for name, maximum_length in vector_bounds.items():
                prepared[name] = _prepare_raw_rational_vector(
                    prepared.get(name),
                    maximum_length=maximum_length,
                    maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                    label=f"general linear-program {name}",
                )
            for name in ("primal_objective", "dual_objective"):
                _bound_raw_rational(
                    prepared.get(name),
                    maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                    label=f"general linear-program {name}",
                )
            return prepared
        except ValueError as error:
            raise _validation_error("raw_result_bound", str(error)) from error

    @model_validator(mode="after")
    def bind_result_to_source(self) -> Self:
        try:
            _require_general_result_shape(self)
            _require_general_result_heights(self)
            primal_objective = _replay_general_primal(self)
            _replay_general_dual(self, primal_objective)
            _replay_general_farkas(self)
            _replay_general_recession(self)
        except ValueError as error:
            raise _validation_error("result_replay", str(error)) from error
        return self


def _fractions(values: tuple[CanonicalRational, ...]) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in values)


def _source_arrays(
    program: GeneralFormRationalLinearProgram,
) -> tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]:
    return (
        _fractions(program.objective.coefficients),
        tuple(_fractions(row.coefficients) for row in program.constraints),
        tuple(row.rhs.as_fraction() for row in program.constraints),
    )


def _require_general_result_shape(result: GeneralRationalLinearProgramResult) -> None:
    primal = (
        result.primal_candidate,
        result.primal_objective,
        result.primal_residuals,
        result.constraint_slacks,
        result.lower_bound_slacks,
        result.upper_bound_slacks,
    )
    dual = (
        result.constraint_dual,
        result.lower_bound_dual,
        result.upper_bound_dual,
        result.dual_objective,
        result.stationarity_residuals,
    )
    farkas = (
        result.farkas_constraints,
        result.farkas_lower_bounds,
        result.farkas_upper_bounds,
    )
    has_primal = result.status in {"OPTIMAL", "PRIMAL_FEASIBLE", "UNBOUNDED"}
    if has_primal and not all(value is not None for value in primal):
        raise ValueError(
            "general LP primal outcomes require exactly one replayable point"
        )
    if not has_primal and any(value is not None for value in primal):
        raise ValueError(
            "infeasible and unknown general LP results cannot carry primal data"
        )
    if (result.status == "OPTIMAL") != all(value is not None for value in dual):
        raise ValueError("an optimal general LP result requires source dual data")
    if result.status != "OPTIMAL" and any(value is not None for value in dual):
        raise ValueError("only an optimal general LP result can carry dual data")
    if (result.status == "INFEASIBLE") != all(value is not None for value in farkas):
        raise ValueError("an infeasible general LP result requires source Farkas data")
    if result.status != "INFEASIBLE" and any(value is not None for value in farkas):
        raise ValueError("only an infeasible general LP result can carry Farkas data")
    if (result.status == "UNBOUNDED") != (result.recession_direction is not None):
        raise ValueError("an unbounded general LP result requires one source ray")


def _require_general_result_heights(result: GeneralRationalLinearProgramResult) -> None:
    from jacobian.math.optimization._general_normalization import (
        estimated_mapped_result_bytes,
        normalized_certificate_digit_bound,
        normalized_point_digit_bound,
        normalized_residual_digit_bound,
    )

    point_digits = normalized_point_digit_bound(result.program)
    residual_digits = normalized_residual_digit_bound(result.program)
    certificate_digits = normalized_certificate_digit_bound(result.program)
    grouped = (
        (point_digits, result.primal_candidate),
        (point_digits, result.lower_bound_slacks),
        (point_digits, result.upper_bound_slacks),
        (point_digits, result.recession_direction),
        (residual_digits, result.primal_residuals),
        (residual_digits, result.constraint_slacks),
        (certificate_digits, result.constraint_dual),
        (certificate_digits, result.lower_bound_dual),
        (certificate_digits, result.upper_bound_dual),
        (certificate_digits, result.stationarity_residuals),
        (certificate_digits, result.farkas_constraints),
        (certificate_digits, result.farkas_lower_bounds),
        (certificate_digits, result.farkas_upper_bounds),
    )
    for maximum_digits, field in grouped:
        if field is None:
            continue
        for value in field:
            require_bounded_rational(
                value,
                max_digits=maximum_digits,
                label="general linear-program result",
            )
    for maximum_digits, scalar in (
        (residual_digits, result.primal_objective),
        (certificate_digits, result.dual_objective),
    ):
        if scalar is None:
            continue
        require_bounded_rational(
            scalar,
            max_digits=maximum_digits,
            label="general linear-program result",
        )
    if estimated_mapped_result_bytes(result.program) > MAX_LINEAR_PROGRAM_RESULT_BYTES:
        raise ValueError(
            "general linear-program result can exceed the result byte bound"
        )


def _replay_general_primal(
    result: GeneralRationalLinearProgramResult,
) -> Fraction | None:
    if result.primal_candidate is None:
        return None
    program = result.program
    if len(result.primal_candidate) != len(program.variables):
        raise ValueError("general LP primal candidate length must match the source")
    primal = _fractions(result.primal_candidate)
    for value, variable in zip(primal, program.variables, strict=True):
        if (
            variable.lower_bound is not None
            and value < variable.lower_bound.as_fraction()
        ):
            raise ValueError("general LP primal point violates a lower bound")
        if (
            variable.upper_bound is not None
            and value > variable.upper_bound.as_fraction()
        ):
            raise ValueError("general LP primal point violates an upper bound")
    objective, coefficients, rhs = _source_arrays(program)
    residuals = tuple(
        rational_dot(row, primal) - bound
        for row, bound in zip(coefficients, rhs, strict=True)
    )
    for residual, row in zip(residuals, program.constraints, strict=True):
        if (
            (row.relation == "LE" and residual > 0)
            or (row.relation == "EQ" and residual != 0)
            or (row.relation == "GE" and residual < 0)
        ):
            raise ValueError("general LP primal point violates a source relation")
    assert result.primal_objective is not None
    assert result.primal_residuals is not None
    assert result.constraint_slacks is not None
    assert result.lower_bound_slacks is not None
    assert result.upper_bound_slacks is not None
    source_objective = rational_dot(objective, primal)
    if result.primal_objective.as_fraction() != source_objective:
        raise ValueError(
            "general LP primal objective must be recomputed from the source"
        )
    if _fractions(result.primal_residuals) != residuals:
        raise ValueError(
            "general LP primal residuals must be recomputed from the source"
        )
    slacks = tuple(
        -residual if row.relation == "LE" else residual
        for residual, row in zip(residuals, program.constraints, strict=True)
    )
    lower_slacks = tuple(
        value - variable.lower_bound.as_fraction()
        if variable.lower_bound is not None
        else Fraction()
        for value, variable in zip(primal, program.variables, strict=True)
    )
    upper_slacks = tuple(
        variable.upper_bound.as_fraction() - value
        if variable.upper_bound is not None
        else Fraction()
        for value, variable in zip(primal, program.variables, strict=True)
    )
    if (
        len(result.constraint_slacks) != len(program.constraints)
        or len(result.lower_bound_slacks) != len(program.variables)
        or len(result.upper_bound_slacks) != len(program.variables)
        or _fractions(result.constraint_slacks) != slacks
        or _fractions(result.lower_bound_slacks) != lower_slacks
        or _fractions(result.upper_bound_slacks) != upper_slacks
        or any(value < 0 for value in (*slacks, *lower_slacks, *upper_slacks))
    ):
        raise ValueError("general LP source slacks must replay exactly")
    return source_objective


def _replay_general_dual(
    result: GeneralRationalLinearProgramResult,
    primal_objective: Fraction | None,
) -> None:
    if result.constraint_dual is None:
        return
    program = result.program
    variables = program.variables
    if (
        len(result.constraint_dual) != len(program.constraints)
        or result.lower_bound_dual is None
        or len(result.lower_bound_dual) != len(variables)
        or result.upper_bound_dual is None
        or len(result.upper_bound_dual) != len(variables)
        or result.stationarity_residuals is None
        or len(result.stationarity_residuals) != len(variables)
    ):
        raise ValueError("general LP dual dimensions must match the source")
    multipliers = _fractions(result.constraint_dual)
    lower = _fractions(result.lower_bound_dual)
    upper = _fractions(result.upper_bound_dual)
    sign = Fraction(1 if program.objective.sense == "MINIMIZE" else -1)
    for multiplier, row in zip(multipliers, program.constraints, strict=True):
        if row.relation == "LE" and sign * multiplier > 0:
            raise ValueError("general LP dual LE multiplier has the wrong sign")
        if row.relation == "GE" and sign * multiplier < 0:
            raise ValueError("general LP dual GE multiplier has the wrong sign")
    for lower_multiplier, upper_multiplier, variable in zip(
        lower, upper, variables, strict=True
    ):
        if variable.lower_bound is None and lower_multiplier != 0:
            raise ValueError("an absent lower bound has zero dual multiplier")
        if variable.upper_bound is None and upper_multiplier != 0:
            raise ValueError("an absent upper bound has zero dual multiplier")
        if variable.lower_bound is not None and sign * lower_multiplier < 0:
            raise ValueError("general LP lower-bound multiplier has the wrong sign")
        if variable.upper_bound is not None and sign * upper_multiplier > 0:
            raise ValueError("general LP upper-bound multiplier has the wrong sign")
    objective, coefficients, rhs = _source_arrays(program)
    stationarity = tuple(
        objective[column]
        - sum(
            (
                row[column] * multiplier
                for row, multiplier in zip(coefficients, multipliers, strict=True)
            ),
            Fraction(),
        )
        - lower[column]
        - upper[column]
        for column in range(len(variables))
    )
    assert result.stationarity_residuals is not None
    if _fractions(result.stationarity_residuals) != stationarity or any(stationarity):
        raise ValueError("general LP dual stationarity must replay exactly")
    dual_objective = (
        rational_dot(rhs, multipliers)
        + sum(
            (
                lower_value * variable.lower_bound.as_fraction()
                for lower_value, variable in zip(lower, variables, strict=True)
                if variable.lower_bound is not None
            ),
            Fraction(),
        )
        + sum(
            (
                upper_value * variable.upper_bound.as_fraction()
                for upper_value, variable in zip(upper, variables, strict=True)
                if variable.upper_bound is not None
            ),
            Fraction(),
        )
    )
    assert result.dual_objective is not None
    if result.dual_objective.as_fraction() != dual_objective:
        raise ValueError("general LP dual objective must be recomputed from the source")
    if primal_objective != dual_objective:
        raise ValueError("general LP optimal primal and dual objectives must agree")


def _replay_general_farkas(result: GeneralRationalLinearProgramResult) -> None:
    if result.farkas_constraints is None:
        return
    program = result.program
    if (
        len(result.farkas_constraints) != len(program.constraints)
        or result.farkas_lower_bounds is None
        or len(result.farkas_lower_bounds) != len(program.variables)
        or result.farkas_upper_bounds is None
        or len(result.farkas_upper_bounds) != len(program.variables)
    ):
        raise ValueError("general LP Farkas dimensions must match the source")
    multipliers = _fractions(result.farkas_constraints)
    lower = _fractions(result.farkas_lower_bounds)
    upper = _fractions(result.farkas_upper_bounds)
    for multiplier, row in zip(multipliers, program.constraints, strict=True):
        if (row.relation == "LE" and multiplier < 0) or (
            row.relation == "GE" and multiplier > 0
        ):
            raise ValueError("general LP Farkas row multiplier has the wrong sign")
    for lower_multiplier, upper_multiplier, variable in zip(
        lower, upper, program.variables, strict=True
    ):
        if variable.lower_bound is None and lower_multiplier != 0:
            raise ValueError("an absent lower bound has zero Farkas multiplier")
        if variable.upper_bound is None and upper_multiplier != 0:
            raise ValueError("an absent upper bound has zero Farkas multiplier")
        if lower_multiplier > 0 or upper_multiplier < 0:
            raise ValueError("general LP Farkas bound multiplier has the wrong sign")
    _, coefficients, rhs = _source_arrays(program)
    balance = tuple(
        sum(
            (
                row[column] * multiplier
                for row, multiplier in zip(coefficients, multipliers, strict=True)
            ),
            Fraction(),
        )
        + lower[column]
        + upper[column]
        for column in range(len(program.variables))
    )
    if any(balance):
        raise ValueError("general LP Farkas multipliers must balance every variable")
    pairing = (
        rational_dot(rhs, multipliers)
        + sum(
            (
                lower_value * variable.lower_bound.as_fraction()
                for lower_value, variable in zip(lower, program.variables, strict=True)
                if variable.lower_bound is not None
            ),
            Fraction(),
        )
        + sum(
            (
                upper_value * variable.upper_bound.as_fraction()
                for upper_value, variable in zip(upper, program.variables, strict=True)
                if variable.upper_bound is not None
            ),
            Fraction(),
        )
    )
    if pairing >= 0:
        raise ValueError("general LP Farkas rhs pairing must be strictly negative")


def _replay_general_recession(result: GeneralRationalLinearProgramResult) -> None:
    if result.recession_direction is None:
        return
    program = result.program
    if len(result.recession_direction) != len(program.variables):
        raise ValueError("general LP recession direction length must match the source")
    direction = _fractions(result.recession_direction)
    for value, variable in zip(direction, program.variables, strict=True):
        if variable.lower_bound is not None and value < 0:
            raise ValueError(
                "general LP lower-bounded ray coordinate must be nonnegative"
            )
        if variable.upper_bound is not None and value > 0:
            raise ValueError(
                "general LP upper-bounded ray coordinate must be nonpositive"
            )
    _, coefficients, _ = _source_arrays(program)
    for row, constraint in zip(coefficients, program.constraints, strict=True):
        derivative = rational_dot(row, direction)
        if (
            (constraint.relation == "EQ" and derivative != 0)
            or (constraint.relation == "LE" and derivative > 0)
            or (constraint.relation == "GE" and derivative < 0)
        ):
            raise ValueError("general LP recession direction violates a source row")
    objective, _, _ = _source_arrays(program)
    improvement = rational_dot(objective, direction)
    if (program.objective.sense == "MINIMIZE" and improvement >= 0) or (
        program.objective.sense == "MAXIMIZE" and improvement <= 0
    ):
        raise ValueError(
            "general LP recession direction must improve the source objective"
        )


__all__ = [
    "MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS",
    "MAX_GENERAL_LINEAR_PROGRAM_VARIABLES",
    "GeneralFormRationalLinearProgram",
    "GeneralRationalLinearProgramRequest",
    "GeneralRationalLinearProgramResult",
    "RationalLinearConstraint",
    "RationalLinearObjective",
    "RationalLinearProgramVariable",
]
