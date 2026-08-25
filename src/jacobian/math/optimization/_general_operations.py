"""General-form exact rational LP operation built on the private standard kernel."""

from __future__ import annotations

from fractions import Fraction

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.optimization._arithmetic import rational_dot
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramRequest,
    GeneralRationalLinearProgramResult,
    _source_arrays,
)
from jacobian.math.optimization._general_normalization import (
    GeneralLinearNormalization,
    _mapped_certificate_digit_bound,
    _mapped_point_digit_bound,
    _mapped_residual_digit_bound,
    normalize_general_program,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)


def _wire(value: Fraction, *, max_digits: int) -> CanonicalRational | None:
    if (
        len(str(abs(value.numerator))) > max_digits
        or len(str(value.denominator)) > max_digits
    ):
        return None
    return CanonicalRational.from_fraction(value)


def _wire_vector(
    values: tuple[Fraction, ...], *, max_digits: int
) -> tuple[CanonicalRational, ...] | None:
    converted = tuple(_wire(value, max_digits=max_digits) for value in values)
    if any(value is None for value in converted):
        return None
    return tuple(value for value in converted if value is not None)


def _source_point(
    normalization: GeneralLinearNormalization,
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    standard = tuple(value.as_fraction() for value in values)
    return tuple(
        offset
        + sum(
            (multiplier * standard[column] for column, multiplier in mapping),
            Fraction(),
        )
        for offset, mapping in zip(
            normalization.offsets, normalization.columns, strict=True
        )
    )


def _source_direction(
    normalization: GeneralLinearNormalization,
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    standard = tuple(value.as_fraction() for value in values)
    return tuple(
        sum(
            (multiplier * standard[column] for column, multiplier in mapping),
            Fraction(),
        )
        for mapping in normalization.columns
    )


def _primal_data(
    program: GeneralFormRationalLinearProgram,
    point: tuple[Fraction, ...],
) -> tuple[
    Fraction,
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    objective, coefficients, rhs = _source_arrays(program)
    residuals = tuple(
        rational_dot(row, point) - expected
        for row, expected in zip(coefficients, rhs, strict=True)
    )
    constraint_slacks = tuple(
        -residual if row.relation == "LE" else residual
        for residual, row in zip(residuals, program.constraints, strict=True)
    )
    lower_slacks = tuple(
        value - variable.lower_bound.as_fraction()
        if variable.lower_bound is not None
        else Fraction()
        for value, variable in zip(point, program.variables, strict=True)
    )
    upper_slacks = tuple(
        variable.upper_bound.as_fraction() - value
        if variable.upper_bound is not None
        else Fraction()
        for value, variable in zip(point, program.variables, strict=True)
    )
    return (
        rational_dot(objective, point),
        residuals,
        constraint_slacks,
        lower_slacks,
        upper_slacks,
    )


def _effective_constraint_values(
    normalization: GeneralLinearNormalization,
    standard_values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    values = tuple(value.as_fraction() for value in standard_values)
    return tuple(
        values[standard_row] * sign for standard_row, sign in normalization.source_rows
    )


def _effective_bound_values(
    program: GeneralFormRationalLinearProgram,
    normalization: GeneralLinearNormalization,
    constraint_values: tuple[Fraction, ...],
    standard_values: tuple[CanonicalRational, ...],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    source_objective, coefficients, _ = _source_arrays(program)
    sense = Fraction(1 if program.objective.sense == "MINIMIZE" else -1)
    effective_objective = tuple(sense * value for value in source_objective)
    standard = tuple(value.as_fraction() for value in standard_values)
    lower: list[Fraction] = []
    upper: list[Fraction] = []
    for index, variable in enumerate(program.variables):
        gradient = sum(
            (
                row[index] * multiplier
                for row, multiplier in zip(coefficients, constraint_values, strict=True)
            ),
            Fraction(),
        )
        if variable.lower_bound is not None and variable.upper_bound is not None:
            row = normalization.upper_rows[index]
            assert row is not None
            upper_value = standard[row]
            lower_value = effective_objective[index] - gradient - upper_value
        elif variable.lower_bound is not None:
            upper_value = Fraction()
            lower_value = effective_objective[index] - gradient
        elif variable.upper_bound is not None:
            lower_value = Fraction()
            upper_value = effective_objective[index] - gradient
        else:
            lower_value = Fraction()
            upper_value = Fraction()
        lower.append(lower_value)
        upper.append(upper_value)
    return tuple(lower), tuple(upper)


def _effective_farkas_bounds(
    program: GeneralFormRationalLinearProgram,
    normalization: GeneralLinearNormalization,
    constraints: tuple[Fraction, ...],
    standard_values: tuple[CanonicalRational, ...],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    _, coefficients, _ = _source_arrays(program)
    standard = tuple(value.as_fraction() for value in standard_values)
    lower: list[Fraction] = []
    upper: list[Fraction] = []
    for index, variable in enumerate(program.variables):
        gradient = sum(
            (
                row[index] * multiplier
                for row, multiplier in zip(coefficients, constraints, strict=True)
            ),
            Fraction(),
        )
        if variable.lower_bound is not None and variable.upper_bound is not None:
            row = normalization.upper_rows[index]
            assert row is not None
            upper_value = standard[row]
            lower_value = -gradient - upper_value
        elif variable.lower_bound is not None:
            upper_value = Fraction()
            lower_value = -gradient
        elif variable.upper_bound is not None:
            lower_value = Fraction()
            upper_value = -gradient
        else:
            lower_value = Fraction()
            upper_value = Fraction()
        lower.append(lower_value)
        upper.append(upper_value)
    return tuple(lower), tuple(upper)


def _unknown(
    program: GeneralFormRationalLinearProgram,
) -> GeneralRationalLinearProgramResult:
    return GeneralRationalLinearProgramResult(program=program, status="UNKNOWN")


def _primal_feasible(
    program: GeneralFormRationalLinearProgram,
    point: tuple[CanonicalRational, ...],
    objective: CanonicalRational,
    residuals: tuple[CanonicalRational, ...],
    constraint_slacks: tuple[CanonicalRational, ...],
    lower_slacks: tuple[CanonicalRational, ...],
    upper_slacks: tuple[CanonicalRational, ...],
) -> GeneralRationalLinearProgramResult:
    """Return only the replayed source point when no dual proof is available."""

    try:
        return GeneralRationalLinearProgramResult(
            program=program,
            status="PRIMAL_FEASIBLE",
            primal_candidate=point,
            primal_objective=objective,
            primal_residuals=residuals,
            constraint_slacks=constraint_slacks,
            lower_bound_slacks=lower_slacks,
            upper_bound_slacks=upper_slacks,
        )
    except ValidationError:
        return _unknown(program)


def _mapped_primal_fields(
    program: GeneralFormRationalLinearProgram,
    normalization: GeneralLinearNormalization,
    standard_values: tuple[CanonicalRational, ...],
    *,
    point_max_digits: int,
    residual_max_digits: int,
) -> (
    tuple[
        tuple[CanonicalRational, ...],
        CanonicalRational,
        tuple[CanonicalRational, ...],
        tuple[CanonicalRational, ...],
        tuple[CanonicalRational, ...],
        tuple[CanonicalRational, ...],
    ]
    | None
):
    point = _source_point(normalization, standard_values)
    objective, residuals, constraint_slacks, lower_slacks, upper_slacks = _primal_data(
        program, point
    )
    wire_point = _wire_vector(point, max_digits=point_max_digits)
    wire_objective = _wire(objective, max_digits=residual_max_digits)
    wire_residuals = _wire_vector(residuals, max_digits=residual_max_digits)
    wire_constraint_slacks = _wire_vector(
        constraint_slacks, max_digits=residual_max_digits
    )
    wire_lower_slacks = _wire_vector(lower_slacks, max_digits=point_max_digits)
    wire_upper_slacks = _wire_vector(upper_slacks, max_digits=point_max_digits)
    if (
        wire_point is None
        or wire_objective is None
        or wire_residuals is None
        or wire_constraint_slacks is None
        or wire_lower_slacks is None
        or wire_upper_slacks is None
    ):
        return None
    return (
        wire_point,
        wire_objective,
        wire_residuals,
        wire_constraint_slacks,
        wire_lower_slacks,
        wire_upper_slacks,
    )


def _map_standard_result(
    program: GeneralFormRationalLinearProgram,
    normalization: GeneralLinearNormalization,
    standard_result: RationalLinearProgramResult,
    *,
    point_max_digits: int,
    residual_max_digits: int,
    certificate_max_digits: int,
) -> GeneralRationalLinearProgramResult:
    if standard_result.status == "UNKNOWN":
        return _unknown(program)
    if standard_result.status == "INFEASIBLE":
        assert standard_result.farkas_candidate is not None
        constraints = _effective_constraint_values(
            normalization, standard_result.farkas_candidate
        )
        lower, upper = _effective_farkas_bounds(
            program,
            normalization,
            constraints,
            standard_result.farkas_candidate,
        )
        wires = tuple(
            _wire_vector(values, max_digits=certificate_max_digits)
            for values in (constraints, lower, upper)
        )
        if any(value is None for value in wires):
            return _unknown(program)
        constraint_wire, lower_wire, upper_wire = wires
        assert (
            constraint_wire is not None
            and lower_wire is not None
            and upper_wire is not None
        )
        try:
            return GeneralRationalLinearProgramResult(
                program=program,
                status="INFEASIBLE",
                farkas_constraints=constraint_wire,
                farkas_lower_bounds=lower_wire,
                farkas_upper_bounds=upper_wire,
            )
        except ValidationError:
            return _unknown(program)

    assert standard_result.primal_candidate is not None
    primal = _mapped_primal_fields(
        program,
        normalization,
        standard_result.primal_candidate,
        point_max_digits=point_max_digits,
        residual_max_digits=residual_max_digits,
    )
    if primal is None:
        return _unknown(program)
    point, objective, residuals, constraint_slacks, lower_slacks, upper_slacks = primal
    if standard_result.status == "UNBOUNDED":
        assert standard_result.recession_direction is not None
        direction = _source_direction(
            normalization, standard_result.recession_direction
        )
        wire_direction = _wire_vector(direction, max_digits=point_max_digits)
        if wire_direction is None:
            return _unknown(program)
        try:
            return GeneralRationalLinearProgramResult(
                program=program,
                status="UNBOUNDED",
                primal_candidate=point,
                primal_objective=objective,
                primal_residuals=residuals,
                constraint_slacks=constraint_slacks,
                lower_bound_slacks=lower_slacks,
                upper_bound_slacks=upper_slacks,
                recession_direction=wire_direction,
            )
        except ValidationError:
            return _unknown(program)
    if standard_result.status == "PRIMAL_FEASIBLE":
        return _primal_feasible(
            program,
            point,
            objective,
            residuals,
            constraint_slacks,
            lower_slacks,
            upper_slacks,
        )

    assert standard_result.status == "OPTIMAL"
    assert standard_result.dual_candidate is not None
    effective_constraints = _effective_constraint_values(
        normalization, standard_result.dual_candidate
    )
    effective_lower, effective_upper = _effective_bound_values(
        program,
        normalization,
        effective_constraints,
        standard_result.dual_candidate,
    )
    sense = Fraction(1 if program.objective.sense == "MINIMIZE" else -1)
    constraints = tuple(sense * value for value in effective_constraints)
    lower = tuple(sense * value for value in effective_lower)
    upper = tuple(sense * value for value in effective_upper)
    _, coefficient_rows, rhs = _source_arrays(program)
    dual_objective = (
        rational_dot(rhs, constraints)
        + sum(
            (
                value * variable.lower_bound.as_fraction()
                for value, variable in zip(lower, program.variables, strict=True)
                if variable.lower_bound is not None
            ),
            Fraction(),
        )
        + sum(
            (
                value * variable.upper_bound.as_fraction()
                for value, variable in zip(upper, program.variables, strict=True)
                if variable.upper_bound is not None
            ),
            Fraction(),
        )
    )
    source_objective, _, _ = _source_arrays(program)
    stationarity = tuple(
        source_objective[column]
        - sum(
            (
                row[column] * multiplier
                for row, multiplier in zip(coefficient_rows, constraints, strict=True)
            ),
            Fraction(),
        )
        - lower[column]
        - upper[column]
        for column in range(len(program.variables))
    )
    wires = tuple(
        _wire_vector(values, max_digits=certificate_max_digits)
        for values in (constraints, lower, upper, stationarity)
    )
    wire_dual_objective = _wire(dual_objective, max_digits=certificate_max_digits)
    if any(value is None for value in wires) or wire_dual_objective is None:
        return _primal_feasible(
            program,
            point,
            objective,
            residuals,
            constraint_slacks,
            lower_slacks,
            upper_slacks,
        )
    constraint_wire, lower_wire, upper_wire, stationarity_wire = wires
    assert (
        constraint_wire is not None
        and lower_wire is not None
        and upper_wire is not None
        and stationarity_wire is not None
    )
    try:
        return GeneralRationalLinearProgramResult(
            program=program,
            status="OPTIMAL",
            primal_candidate=point,
            primal_objective=objective,
            primal_residuals=residuals,
            constraint_slacks=constraint_slacks,
            lower_bound_slacks=lower_slacks,
            upper_bound_slacks=upper_slacks,
            constraint_dual=constraint_wire,
            lower_bound_dual=lower_wire,
            upper_bound_dual=upper_wire,
            dual_objective=wire_dual_objective,
            stationarity_residuals=stationarity_wire,
        )
    except ValidationError:
        return _primal_feasible(
            program,
            point,
            objective,
            residuals,
            constraint_slacks,
            lower_slacks,
            upper_slacks,
        )


def _general_linear_program(
    request: GeneralRationalLinearProgramRequest,
) -> GeneralRationalLinearProgramResult:
    # Avoid a declaration-cycle: the standard operation imports this sibling
    # only to publish the immutable tuple, while its direct kernel stays private.
    from jacobian.math.optimization._operations import _linear_program

    program = request.program
    normalization = normalize_general_program(program)
    standard_result = _linear_program(
        RationalLinearProgramRequest(program=normalization.standard_program)
    )
    return _map_standard_result(
        program,
        normalization,
        standard_result,
        point_max_digits=_mapped_point_digit_bound(normalization),
        residual_max_digits=_mapped_residual_digit_bound(normalization),
        certificate_max_digits=_mapped_certificate_digit_bound(normalization),
    )


GENERAL_RATIONAL_LINEAR_OPERATIONS: MathTools = (
    MathTool(
        operation_id="optimization.linear.rational_general_optimum.compute",
        title="Solve a general-form rational linear program",
        description=(
            "Solve a bounded exact rational LP with labeled LE, EQ, and GE rows, "
            "finite bounds, and free variables. Results retain the original source "
            "coordinates and replayed optimality, Farkas, or recession evidence; "
            "private slack and sign-split columns never appear on the wire."
        ),
        request_type=GeneralRationalLinearProgramRequest,
        result_type=GeneralRationalLinearProgramResult,
        run=_general_linear_program,
        tags=(
            "optimization",
            "linear-program",
            "rational",
            "inequality",
            "bounded",
        ),
        examples=(
            example(
                "bounded_inequality_lp",
                "Minimize x subject to x>=1; all rows and bounds are closed exact rationals.",
                {
                    "program": {
                        "variables": [
                            {
                                "name": "x",
                                "lower_bound": {"num": "0", "den": "1"},
                                "upper_bound": None,
                            }
                        ],
                        "objective": {
                            "sense": "MINIMIZE",
                            "coefficients": [{"num": "1", "den": "1"}],
                        },
                        "constraints": [
                            {
                                "label": "minimum",
                                "coefficients": [{"num": "1", "den": "1"}],
                                "relation": "GE",
                                "rhs": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["GENERAL_RATIONAL_LINEAR_OPERATIONS"]
