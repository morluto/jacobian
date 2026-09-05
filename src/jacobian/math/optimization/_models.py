"""Private wire models for bounded rational linear optimization."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import factorial
from typing import Any, Literal, Self

from pydantic import Field, ValidationInfo, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.optimization._arithmetic import rational_dot

MAX_RATIONAL_DIGITS = 128
MAX_LINEAR_PROGRAM_VARIABLES = 32
MAX_LINEAR_PROGRAM_CONSTRAINTS = 64
MAX_LINEAR_PROGRAM_VARIABLE_NAME_LENGTH = 64
MAX_LINEAR_PROGRAM_BASES = 1_000_000
MAX_LINEAR_PROGRAM_SCALAR_UPDATES = 50_000_000
_INTERMEDIATE_SCALAR_DIGITS = "standard_intermediate_scalar_digits"


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"linear_program.{reason}", message)


def _scalar_digit_cap(info: ValidationInfo) -> int:
    """Read a derived intermediate scalar cap, defaulting to the source bound."""

    context = info.context
    if isinstance(context, Mapping):
        digits = context.get(_INTERMEDIATE_SCALAR_DIGITS)
        if isinstance(digits, int) and 0 < digits <= MAX_CANONICAL_RATIONAL_DIGITS:
            return digits
    return MAX_RATIONAL_DIGITS


type RationalLinearProgramStatus = Literal[
    "OPTIMAL",
    "PRIMAL_FEASIBLE",
    "INFEASIBLE",
    "UNBOUNDED",
]


def _bound_raw_rational(
    value: object,
    *,
    maximum_digits: int = MAX_RATIONAL_DIGITS,
    label: str,
) -> None:
    """Apply the LP scalar limit before generic rational construction."""

    if isinstance(value, CanonicalRational):
        components: tuple[object, object] = (value.num, value.den)
    elif isinstance(value, Mapping):
        components = (value.get("num"), value.get("den"))
    else:
        return
    for component in components:
        if (
            isinstance(component, str)
            and len(component) - component.startswith("-") > maximum_digits
        ):
            raise ValueError(f"{label} exceeds the {maximum_digits}-digit bound")


def _bound_raw_rational_vector(
    value: object,
    *,
    maximum_length: int,
    maximum_digits: int = MAX_RATIONAL_DIGITS,
    label: str,
) -> None:
    """Bound one raw LP vector before recursive model construction."""

    if not isinstance(value, (list, tuple)):
        return
    if len(value) > maximum_length:
        raise ValueError(f"{label} exceeds the {maximum_length}-entry bound")
    for item in value:
        _bound_raw_rational(item, maximum_digits=maximum_digits, label=label)


def _prepare_raw_rational_vector(
    value: object,
    *,
    maximum_length: int,
    maximum_digits: int = MAX_RATIONAL_DIGITS,
    label: str,
) -> object:
    """Bound and normalize one raw LP vector before nested parsing."""

    _bound_raw_rational_vector(
        value,
        maximum_length=maximum_length,
        maximum_digits=maximum_digits,
        label=label,
    )
    return tuple(value) if isinstance(value, list) else value


def _prepare_raw_program(
    value: object,
    *,
    maximum_digits: int = MAX_RATIONAL_DIGITS,
) -> object:
    """Bound and normalize an LP source before nested model construction."""

    if not isinstance(value, Mapping):
        return value
    variables = value.get("variables")
    if isinstance(variables, (list, tuple)):
        if len(variables) > MAX_LINEAR_PROGRAM_VARIABLES:
            raise ValueError(
                "linear-program variables exceed the "
                f"{MAX_LINEAR_PROGRAM_VARIABLES}-entry bound"
            )
        if any(
            isinstance(name, str)
            and len(name) > MAX_LINEAR_PROGRAM_VARIABLE_NAME_LENGTH
            for name in variables
        ):
            raise ValueError(
                "linear-program variable name exceeds the "
                f"{MAX_LINEAR_PROGRAM_VARIABLE_NAME_LENGTH}-character bound"
            )
    objective = _prepare_raw_rational_vector(
        value.get("objective"),
        maximum_length=MAX_LINEAR_PROGRAM_VARIABLES,
        maximum_digits=maximum_digits,
        label="rational linear-program objective",
    )
    rhs = _prepare_raw_rational_vector(
        value.get("rhs"),
        maximum_length=MAX_LINEAR_PROGRAM_CONSTRAINTS,
        maximum_digits=maximum_digits,
        label="rational linear-program rhs",
    )
    rows = value.get("coefficients")
    if not isinstance(rows, (list, tuple)):
        return value
    if len(rows) > MAX_LINEAR_PROGRAM_CONSTRAINTS:
        raise ValueError(
            "linear-program coefficient rows exceed the "
            f"{MAX_LINEAR_PROGRAM_CONSTRAINTS}-row bound"
        )
    prepared_rows = tuple(
        _prepare_raw_rational_vector(
            row,
            maximum_length=MAX_LINEAR_PROGRAM_VARIABLES,
            maximum_digits=maximum_digits,
            label="rational linear-program coefficient row",
        )
        for row in rows
    )
    prepared = dict(value)
    if isinstance(variables, list):
        prepared["variables"] = tuple(variables)
    prepared["objective"] = objective
    prepared["coefficients"] = prepared_rows
    prepared["rhs"] = rhs
    return prepared


def _cleared_row_digit_bound(
    values: tuple[CanonicalRational, ...],
) -> int:
    """Bound one integer row after clearing every rational denominator."""

    if not values:
        return 1
    denominator_digits = sum(
        len(denominator) for denominator in {v.den for v in values}
    )
    return max(
        len(value.num.lstrip("-")) + denominator_digits - len(value.den)
        for value in values
    )


def _determinant_digit_bound(
    row_bounds: tuple[int, ...],
    *,
    variable_columns: int,
) -> int:
    """Leibniz-bound every basis minor and one diagnostic row."""

    order = min(variable_columns + 1, len(row_bounds))
    if order == 0:
        return 1
    largest_rows = sorted(row_bounds, reverse=True)[:order]
    return sum(largest_rows) + len(str(factorial(order))) + 2


def _active_equations(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[tuple[CanonicalRational, ...], CanonicalRational], ...]:
    return tuple(
        (row, rhs)
        for row, rhs in zip(program.coefficients, program.rhs, strict=True)
        if any(value.num != "0" for value in row) or rhs.num != "0"
    )


def _has_trivial_inconsistent_row(
    program: StandardFormRationalLinearProgram,
) -> bool:
    return any(
        not any(value.num != "0" for value in row) and rhs.num != "0"
        for row, rhs in zip(program.coefficients, program.rhs, strict=True)
    )


def _result_digit_bound(program: StandardFormRationalLinearProgram) -> int:
    """Conservatively bound basis witnesses and source-derived diagnostics.

    Clearing denominators row by row turns each exact LP used by the operation
    into an integer matrix. Every basic coordinate, reduced cost, objective,
    and Farkas or recession coordinate is a ratio of minors. Both source-row
    and transposed-row denominator clearings are covered below. Duplicated rows
    and normalization rows deliberately overestimate the minors needed by the
    single-artificial-column basis algorithm; the additional diagnostic rows
    and final two digits cover exact dot products and differences.
    """

    if _has_trivial_inconsistent_row(program):
        return max(
            (
                max(len(value.num.lstrip("-")), len(value.den))
                for value in (
                    *program.objective,
                    *program.rhs,
                    *(value for row in program.coefficients for value in row),
                )
            ),
            default=1,
        )

    active_equations = _active_equations(program)
    active_coefficients = tuple(row for row, _ in active_equations)
    active_rhs = tuple(rhs for _, rhs in active_equations)
    zero = CanonicalRational(num="0", den="1")
    one = CanonicalRational(num="1", den="1")
    objective_row = _cleared_row_digit_bound((*program.objective, zero))
    unit_primal_objective = _cleared_row_digit_bound(
        (*(one for _ in program.variables), zero)
    )
    primal_rows = tuple(
        _cleared_row_digit_bound((*row, rhs)) for row, rhs in active_equations
    )
    dual_rows = tuple(
        _cleared_row_digit_bound(
            (
                *(row[column] for row in active_coefficients),
                program.objective[column],
            )
        )
        for column in range(len(program.variables))
    )
    dual_objective_row = _cleared_row_digit_bound((*active_rhs, zero))
    farkas_rows = tuple(
        _cleared_row_digit_bound((*(row[column] for row in active_coefficients), zero))
        for column in range(len(program.variables))
    )
    farkas_normalization = _cleared_row_digit_bound((*active_rhs, one))
    ray_rows = tuple(
        _cleared_row_digit_bound((*row, zero)) for row in active_coefficients
    )
    ray_normalization = _cleared_row_digit_bound((*program.objective, one))

    variables = len(program.variables)
    equations = len(active_equations)
    return max(
        _determinant_digit_bound(
            (*primal_rows, *primal_rows, objective_row, unit_primal_objective),
            variable_columns=2 * variables,
        ),
        _determinant_digit_bound(
            (*dual_rows, dual_objective_row),
            variable_columns=2 * equations,
        ),
        _determinant_digit_bound(
            (*farkas_rows, farkas_normalization, farkas_normalization, 1),
            variable_columns=2 * equations,
        ),
        _determinant_digit_bound(
            (
                *ray_rows,
                *ray_rows,
                ray_normalization,
                ray_normalization,
                unit_primal_objective,
            ),
            variable_columns=2 * variables,
        ),
    )


def _program_fractions(
    program: StandardFormRationalLinearProgram,
) -> tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]:
    return (
        tuple(value.as_fraction() for value in program.objective),
        tuple(
            tuple(value.as_fraction() for value in row) for row in program.coefficients
        ),
        tuple(value.as_fraction() for value in program.rhs),
    )


def _primal_diagnostics(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[Fraction, ...],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    objective, coefficients, rhs = _program_fractions(program)
    return rational_dot(objective, candidate), tuple(
        rational_dot(row, candidate) - expected
        for row, expected in zip(coefficients, rhs, strict=True)
    )


def _dual_diagnostics(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[Fraction, ...],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    objective, coefficients, rhs = _program_fractions(program)
    slacks = tuple(
        objective[column]
        - sum(
            (
                coefficients[row][column] * candidate[row]
                for row in range(len(coefficients))
            ),
            Fraction(),
        )
        for column in range(len(objective))
    )
    return rational_dot(rhs, candidate), slacks


class StandardFormRationalLinearProgram(StrictModel):
    """Minimize ``cᵀx`` subject to ``Ax=b`` and ``x>=0``.

    An empty coefficient/RHS family is the unconstrained nonnegative orthant.
    """

    variables: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_PROGRAM_VARIABLES
    )
    objective: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_LINEAR_PROGRAM_VARIABLES
    )
    coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_LINEAR_PROGRAM_CONSTRAINTS
    )
    rhs: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_LINEAR_PROGRAM_CONSTRAINTS
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_program(cls, value: object, info: ValidationInfo) -> object:
        value = canonicalize_json_containers(value)
        try:
            return _prepare_raw_program(value, maximum_digits=_scalar_digit_cap(info))
        except ValueError as error:
            raise _validation_error("raw_input_bound", str(error)) from error

    @model_validator(mode="after")
    def require_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "duplicate_variable", "linear-program variable names must be unique"
            )
        if any(
            not name
            or len(name) > MAX_LINEAR_PROGRAM_VARIABLE_NAME_LENGTH
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (char.isalnum() or char == "_") for char in name)
            for name in self.variables
        ):
            raise _validation_error(
                "variable_identifier",
                "linear-program variable names must be identifiers",
            )
        width = len(self.variables)
        if len(self.objective) != width:
            raise _validation_error(
                "objective_length", "objective length must equal the variable count"
            )
        if len(self.coefficients) != len(self.rhs):
            raise _validation_error(
                "row_count", "coefficient row count must equal the rhs length"
            )
        if any(len(row) != width for row in self.coefficients):
            raise _validation_error(
                "row_width", "every coefficient row must match the variable count"
            )

        return self

    @classmethod
    def admit_derived_intermediate(cls, value: object, *, maximum_digits: int) -> Self:
        """Construct one privately derived program under a proven scalar envelope.

        Public requests keep the ``MAX_RATIONAL_DIGITS`` source-scalar cap.
        A normalization step that derives its intermediates from admitted
        inputs may pass the envelope its own derivation proves; every derived
        shape checks above still run; execution owns resource admission.
        """

        return cls.model_validate(
            value, context={_INTERMEDIATE_SCALAR_DIGITS: maximum_digits}
        )


class RationalLinearProgramRequest(StrictModel):
    program: StandardFormRationalLinearProgram


class RationalLinearProgramResult(StrictModel):
    """A source-bound outcome for one standard-form rational LP.

    ``INFEASIBLE`` uses the Farkas convention ``Aᵀy>=0, bᵀy<0``.
    ``UNBOUNDED`` carries feasible ``x0`` and nonnegative ``d`` satisfying
    ``Ad=0, cᵀd<0``.
    """

    program: StandardFormRationalLinearProgram
    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_VARIABLES
    )
    dual_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_CONSTRAINTS
    )
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_CONSTRAINTS
    )
    dual_slacks: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_VARIABLES
    )
    farkas_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_CONSTRAINTS
    )
    recession_direction: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=MAX_LINEAR_PROGRAM_VARIABLES
    )

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build an outcome whose source-derived facts the kernel established.

        Result parsing intentionally does not rerun simplex certificates.  The
        optimization kernel computes every diagnostic and witness before using
        this trusted construction path.
        """

        return cls.model_construct(**values)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        try:
            prepared: dict[str, object] = dict(value)
            prepared["program"] = _prepare_raw_program(prepared.get("program"))
            prepared["primal_candidate"] = _prepare_raw_rational_vector(
                prepared.get("primal_candidate"),
                maximum_length=MAX_LINEAR_PROGRAM_VARIABLES,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program primal candidate",
            )
            prepared["dual_candidate"] = _prepare_raw_rational_vector(
                prepared.get("dual_candidate"),
                maximum_length=MAX_LINEAR_PROGRAM_CONSTRAINTS,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program dual candidate",
            )
            prepared["primal_residuals"] = _prepare_raw_rational_vector(
                prepared.get("primal_residuals"),
                maximum_length=MAX_LINEAR_PROGRAM_CONSTRAINTS,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program primal residuals",
            )
            prepared["dual_slacks"] = _prepare_raw_rational_vector(
                prepared.get("dual_slacks"),
                maximum_length=MAX_LINEAR_PROGRAM_VARIABLES,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program dual slacks",
            )
            prepared["farkas_candidate"] = _prepare_raw_rational_vector(
                prepared.get("farkas_candidate"),
                maximum_length=MAX_LINEAR_PROGRAM_CONSTRAINTS,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program Farkas candidate",
            )
            prepared["recession_direction"] = _prepare_raw_rational_vector(
                prepared.get("recession_direction"),
                maximum_length=MAX_LINEAR_PROGRAM_VARIABLES,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program recession direction",
            )
            _bound_raw_rational(
                prepared.get("primal_objective"),
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program primal objective",
            )
            _bound_raw_rational(
                prepared.get("dual_objective"),
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program dual objective",
            )
            return prepared
        except ValueError as error:
            raise _validation_error("raw_result_bound", str(error)) from error

    @model_validator(mode="after")
    def bind_result_to_source(self) -> Self:
        try:
            _require_result_shape(self)
        except ValueError as error:
            raise _validation_error("result_shape", str(error)) from error
        return self


def _require_result_shape(result: RationalLinearProgramResult) -> None:
    primal_fields = (
        result.primal_candidate,
        result.primal_objective,
        result.primal_residuals,
    )
    dual_fields = (
        result.dual_candidate,
        result.dual_objective,
        result.dual_slacks,
    )
    has_primal = result.status in {"OPTIMAL", "PRIMAL_FEASIBLE", "UNBOUNDED"}
    if has_primal and not all(value is not None for value in primal_fields):
        raise ValueError(
            "optimal, primal-feasible, and unbounded results require exactly "
            "one feasible primal point with its source-derived diagnostics"
        )
    if not has_primal and any(value is not None for value in primal_fields):
        raise ValueError("infeasible results cannot carry primal data")
    if result.status == "OPTIMAL" and not all(
        value is not None for value in dual_fields
    ):
        raise ValueError(
            "an optimal result requires exactly one dual point with its "
            "source-derived diagnostics"
        )
    if result.status != "OPTIMAL" and any(value is not None for value in dual_fields):
        raise ValueError("only an optimal result can carry dual data")
    if (result.status == "INFEASIBLE") != (result.farkas_candidate is not None):
        raise ValueError("an infeasible result requires exactly one Farkas candidate")
    if (result.status == "UNBOUNDED") != (result.recession_direction is not None):
        raise ValueError("an unbounded result requires exactly one recession direction")


__all__ = [
    "MAX_LINEAR_PROGRAM_BASES",
    "MAX_LINEAR_PROGRAM_SCALAR_UPDATES",
    "RationalLinearProgramRequest",
    "RationalLinearProgramResult",
    "RationalLinearProgramStatus",
    "StandardFormRationalLinearProgram",
]
