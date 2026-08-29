"""Canonical source values and replayed outcomes for general rational LPs."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.optimization._models import (
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
MAX_GENERAL_RATIONAL_INPUT_DIGITS = 128


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
                        max_digits=MAX_GENERAL_RATIONAL_INPUT_DIGITS,
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
                    max_digits=MAX_GENERAL_RATIONAL_INPUT_DIGITS,
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
                    max_digits=MAX_GENERAL_RATIONAL_INPUT_DIGITS,
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
        value = canonicalize_json_containers(value)
        try:
            return _prepare_raw_general_program(value)
        except ValueError as error:
            raise _validation_error("raw_input_bound", str(error)) from error

    @model_validator(mode="after")
    def require_dimensions(self) -> Self:
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
        return self


class GeneralRationalLinearProgramRequest(StrictModel):
    program: GeneralFormRationalLinearProgram


class GeneralRationalLinearProgramResult(StrictModel):
    """A canonical exact outcome for a general-form rational LP.

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

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build an outcome whose source-derived facts the kernel established.

        Parsing checks representation and field presence only.  It does not
        rerun normalization or certificate arithmetic.
        """

        return cls.model_construct(**values)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
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
        except ValueError as error:
            raise _validation_error("result_shape", str(error)) from error
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
