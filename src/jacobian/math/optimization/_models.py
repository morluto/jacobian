"""Private wire models for bounded rational linear optimization."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import comb, factorial
from typing import Literal, Self

from pydantic import Field, ValidationInfo, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.optimization._arithmetic import rational_dot

MAX_RATIONAL_DIGITS = 128
MAX_LINEAR_PROGRAM_RESULT_BYTES = 10 * 1024 * 1024
MAX_LINEAR_PROGRAM_SIMPLEX_BASES = 1_000_000
MAX_LINEAR_PROGRAM_SIMPLEX_SCALAR_UPDATES = 50_000_000
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
    "UNKNOWN",
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
        if len(variables) > 32:
            raise ValueError("linear-program variables exceed the 32-entry bound")
        if any(isinstance(name, str) and len(name) > 64 for name in variables):
            raise ValueError(
                "linear-program variable name exceeds the 64-character bound"
            )
    objective = _prepare_raw_rational_vector(
        value.get("objective"),
        maximum_length=32,
        maximum_digits=maximum_digits,
        label="rational linear-program objective",
    )
    rhs = _prepare_raw_rational_vector(
        value.get("rhs"),
        maximum_length=64,
        maximum_digits=maximum_digits,
        label="rational linear-program rhs",
    )
    rows = value.get("coefficients")
    if not isinstance(rows, (list, tuple)):
        return value
    if len(rows) > 64:
        raise ValueError("linear-program coefficient rows exceed the 64-row bound")
    prepared_rows = tuple(
        _prepare_raw_rational_vector(
            row,
            maximum_length=32,
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
    """Bound simplex tableaux, witnesses, and source-derived diagnostics.

    Clearing denominators row by row turns each exact LP used by the operation
    into an integer tableau. Every basic coordinate, reduced cost, objective,
    and normalized Farkas or recession coordinate is a ratio of minors. The
    Leibniz bounds below include one additional row for a derived diagnostic.
    Free dual/Farkas variables are represented as a difference of two
    nonnegative basic variables; the final two digits cover that subtraction.
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


def _simplex_candidate_and_update_bounds(
    *,
    variables: int,
    equations: int,
) -> tuple[int, int]:
    """Bound Bland pivots by all possible bases in both simplex phases."""

    def solve(rows: int, columns: int) -> tuple[int, int]:
        # SymPy's pinned exact simplex uses Bland's rule. Each phase therefore
        # visits no basis twice; the factor two covers feasibility and optimum.
        candidates = 2 * comb(rows + columns, rows)
        scalar_updates = candidates * (rows + 2) * (columns + 2)
        return candidates, scalar_updates

    # ``lpmin`` may introduce one shifted nonnegative auxiliary for each
    # source variable whose univariate constraints form a finite interval.
    primal = solve(2 * equations, 2 * variables)
    dual = solve(variables, 2 * equations)
    farkas = solve(variables + 2, 2 * equations)
    feasible_point = primal
    recession = solve(2 * (equations + 1), 2 * variables)
    branches = (
        (primal, dual),
        (primal, farkas),
        (primal, feasible_point, recession),
    )
    return (
        max(sum(item[0] for item in branch) for branch in branches),
        max(sum(item[1] for item in branch) for branch in branches),
    )


def _source_wire_bytes(program: StandardFormRationalLinearProgram) -> int:
    rationals = (
        *program.objective,
        *program.rhs,
        *(value for row in program.coefficients for value in row),
    )
    return (
        2_048
        + sum(len(value.num) + len(value.den) + 32 for value in rationals)
        + sum(len(variable) + 4 for variable in program.variables)
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


def _recession_diagnostics(
    program: StandardFormRationalLinearProgram,
    direction: tuple[Fraction, ...],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    objective, coefficients, _ = _program_fractions(program)
    return rational_dot(objective, direction), tuple(
        rational_dot(row, direction) for row in coefficients
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

    variables: tuple[str, ...] = Field(min_length=1, max_length=32)
    objective: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)
    coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(max_length=64)
    rhs: tuple[CanonicalRational, ...] = Field(max_length=64)

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
            or len(name) > 64
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

        result_digits = _result_digit_bound(self)
        if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _validation_error(
                "result_height",
                "linear-program rational height can exceed the exact "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound",
            )
        direct_certificate = _has_trivial_inconsistent_row(self)
        active_equations = 0 if direct_certificate else len(_active_equations(self))
        candidates, scalar_updates = (0, 0)
        if not direct_certificate:
            candidates, scalar_updates = _simplex_candidate_and_update_bounds(
                variables=len(self.variables),
                equations=active_equations,
            )
        if candidates > MAX_LINEAR_PROGRAM_SIMPLEX_BASES:
            raise _validation_error(
                "simplex_basis_bound",
                "linear-program possible simplex bases exceed the "
                f"{MAX_LINEAR_PROGRAM_SIMPLEX_BASES}-candidate bound",
            )
        if scalar_updates > MAX_LINEAR_PROGRAM_SIMPLEX_SCALAR_UPDATES:
            raise _validation_error(
                "simplex_work_bound",
                "linear-program exact simplex pivot work exceeds the "
                f"{MAX_LINEAR_PROGRAM_SIMPLEX_SCALAR_UPDATES}-scalar-update bound",
            )
        derived_rationals = 2 * len(self.variables) + 2 * len(self.rhs) + 2
        estimated_result_bytes = (
            _source_wire_bytes(self)
            + derived_rationals * (2 * result_digits + 32)
            + 4_096
        )
        if estimated_result_bytes > MAX_LINEAR_PROGRAM_RESULT_BYTES:
            raise _validation_error(
                "result_size_bound",
                "linear-program exact result can exceed the "
                f"{MAX_LINEAR_PROGRAM_RESULT_BYTES}-byte result bound",
            )
        return self

    @classmethod
    def admit_derived_intermediate(cls, value: object, *, maximum_digits: int) -> Self:
        """Construct one privately derived program under a proven scalar envelope.

        Public requests keep the ``MAX_RATIONAL_DIGITS`` source-scalar cap.
        A normalization step that derives its intermediates from admitted
        inputs may pass the envelope its own derivation proves; every derived
        result-height, simplex-work, and byte check above still runs unchanged.
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
    ``Ad=0, cᵀd<0``. ``UNKNOWN`` makes no mathematical claim.
    """

    program: StandardFormRationalLinearProgram
    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=32
    )
    dual_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=64
    )
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=64
    )
    dual_slacks: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=32
    )
    farkas_candidate: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=64
    )
    recession_direction: tuple[CanonicalRational, ...] | None = Field(
        default=None, max_length=32
    )

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
                maximum_length=32,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program primal candidate",
            )
            prepared["dual_candidate"] = _prepare_raw_rational_vector(
                prepared.get("dual_candidate"),
                maximum_length=64,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program dual candidate",
            )
            prepared["primal_residuals"] = _prepare_raw_rational_vector(
                prepared.get("primal_residuals"),
                maximum_length=64,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program primal residuals",
            )
            prepared["dual_slacks"] = _prepare_raw_rational_vector(
                prepared.get("dual_slacks"),
                maximum_length=32,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program dual slacks",
            )
            prepared["farkas_candidate"] = _prepare_raw_rational_vector(
                prepared.get("farkas_candidate"),
                maximum_length=64,
                maximum_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="rational linear-program Farkas candidate",
            )
            prepared["recession_direction"] = _prepare_raw_rational_vector(
                prepared.get("recession_direction"),
                maximum_length=32,
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
            _require_result_heights(self)
            primal_objective = _replay_primal(self)
            _replay_dual(self, primal_objective)
            _replay_farkas(self)
            _replay_recession(self)
        except ValueError as error:
            raise _validation_error("result_replay", str(error)) from error
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
        raise ValueError("infeasible and unknown results cannot carry primal data")
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


def _require_result_heights(result: RationalLinearProgramResult) -> None:
    result_bound = _result_digit_bound(result.program)
    vector_fields = (
        result.primal_candidate,
        result.dual_candidate,
        result.primal_residuals,
        result.dual_slacks,
        result.farkas_candidate,
        result.recession_direction,
    )
    scalar_fields = (result.primal_objective, result.dual_objective)
    values = tuple(
        value for field in vector_fields if field is not None for value in field
    ) + tuple(value for value in scalar_fields if value is not None)
    for value in values:
        require_bounded_rational(
            value,
            max_digits=result_bound,
            label="rational linear-program result",
        )


def _replay_primal(result: RationalLinearProgramResult) -> Fraction | None:
    if result.primal_candidate is None:
        return None
    if len(result.primal_candidate) != len(result.program.variables):
        raise ValueError("primal candidate length must match the source")
    primal = tuple(value.as_fraction() for value in result.primal_candidate)
    if any(value < 0 for value in primal):
        raise ValueError("primal candidate must be nonnegative")
    objective, residuals = _primal_diagnostics(result.program, primal)
    assert result.primal_objective is not None
    assert result.primal_residuals is not None
    if result.primal_objective.as_fraction() != objective:
        raise ValueError("primal objective must be recomputed from the source")
    if tuple(value.as_fraction() for value in result.primal_residuals) != residuals:
        raise ValueError("primal residuals must be recomputed from the source")
    if len(residuals) != len(result.program.rhs) or any(residuals):
        raise ValueError("primal candidate must satisfy the source equalities")
    return objective


def _replay_dual(
    result: RationalLinearProgramResult,
    primal_objective: Fraction | None,
) -> None:
    if result.dual_candidate is None:
        return
    if len(result.dual_candidate) != len(result.program.rhs):
        raise ValueError("dual candidate length must match the source")
    dual = tuple(value.as_fraction() for value in result.dual_candidate)
    objective, slacks = _dual_diagnostics(result.program, dual)
    assert result.dual_objective is not None
    assert result.dual_slacks is not None
    if result.dual_objective.as_fraction() != objective:
        raise ValueError("dual objective must be recomputed from the source")
    if tuple(value.as_fraction() for value in result.dual_slacks) != slacks:
        raise ValueError("dual slacks must be recomputed from the source")
    if len(slacks) != len(result.program.variables) or any(
        value < 0 for value in slacks
    ):
        raise ValueError("dual candidate must satisfy the source inequalities")
    if primal_objective != objective:
        raise ValueError("optimal primal and dual objectives must agree")


def _replay_farkas(result: RationalLinearProgramResult) -> None:
    if result.farkas_candidate is None:
        return
    if len(result.farkas_candidate) != len(result.program.rhs):
        raise ValueError("Farkas candidate length must match the source")
    farkas = tuple(value.as_fraction() for value in result.farkas_candidate)
    _, slacks = _dual_diagnostics(result.program, farkas)
    # _dual_diagnostics returns c-Aᵀy; remove c to recover Aᵀy.
    objective = tuple(value.as_fraction() for value in result.program.objective)
    pairings = tuple(
        objective[column] - slacks[column] for column in range(len(objective))
    )
    rhs = tuple(value.as_fraction() for value in result.program.rhs)
    if any(value < 0 for value in pairings) or rational_dot(rhs, farkas) >= 0:
        raise ValueError(
            "Farkas candidate must satisfy Aᵀy>=0 and bᵀy<0 for the source"
        )


def _replay_recession(result: RationalLinearProgramResult) -> None:
    if result.recession_direction is None:
        return
    if len(result.recession_direction) != len(result.program.variables):
        raise ValueError("recession direction length must match the source")
    direction = tuple(value.as_fraction() for value in result.recession_direction)
    if any(value < 0 for value in direction):
        raise ValueError("recession direction must be nonnegative")
    objective, residuals = _recession_diagnostics(result.program, direction)
    if any(residuals) or objective >= 0:
        raise ValueError(
            "recession direction must satisfy Ad=0 and cᵀd<0 for the source"
        )


__all__ = [
    "MAX_LINEAR_PROGRAM_RESULT_BYTES",
    "MAX_LINEAR_PROGRAM_SIMPLEX_BASES",
    "MAX_LINEAR_PROGRAM_SIMPLEX_SCALAR_UPDATES",
    "RationalLinearProgramRequest",
    "RationalLinearProgramResult",
    "RationalLinearProgramStatus",
    "StandardFormRationalLinearProgram",
]
