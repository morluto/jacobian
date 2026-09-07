"""Bounded exact LP outcomes from FLINT basis linear algebra."""

from fractions import Fraction
from typing import Any, NoReturn

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.math.optimization._arithmetic import rational_dot
from jacobian.math.optimization._linear_basis import (
    LinearAdmission,
    admit_linear_program,
    independent_rows,
    linear_execution,
    search_bases,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
    _dual_diagnostics,
    _primal_diagnostics,
    _program_fractions,
)


def _execution_failure() -> NoReturn:
    raise RuntimeError("exact linear-program execution produced no mathematical result")


def _fractions(vector: Any) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(int(v.numerator), int(v.denominator)) for v in vector.entries()
    )


def _wire(values: tuple[Fraction, ...], digits: int) -> tuple[CanonicalRational, ...]:
    result = []
    for value in values:
        numerator = format_canonical_integer(value.numerator)
        denominator = format_canonical_integer(value.denominator)
        if max(len(numerator.lstrip("-")), len(denominator)) > digits:
            _execution_failure()
        result.append(CanonicalRational(num=value.numerator, den=value.denominator))
    return tuple(result)


def _certify_infeasible(
    program: StandardFormRationalLinearProgram,
    witness: tuple[Fraction, ...],
    digits: int,
) -> RationalLinearProgramResult:
    """Establish A^T y>=0 AND b^T y<0 before the trusted boundary."""
    _, a, b = _program_fractions(program)
    if (
        len(witness) != len(b)
        or rational_dot(b, witness) >= 0
        or any(
            sum((row[j] * y for row, y in zip(a, witness, strict=True)), Fraction()) < 0
            for j in range(len(program.variables))
        )
    ):
        _execution_failure()
    return RationalLinearProgramResult._from_kernel(
        program=program,
        status="INFEASIBLE",
        farkas_candidate=_wire(witness, digits),
    )


def _certify_point(
    program: StandardFormRationalLinearProgram,
    point: tuple[Fraction, ...],
    dual: tuple[Fraction, ...],
    ray: tuple[Fraction, ...] | None,
    digits: int,
) -> RationalLinearProgramResult:
    objective, residuals = _primal_diagnostics(program, point)
    if any(v < 0 for v in point) or any(residuals):
        _execution_failure()
    values: dict[str, Any] = {
        "program": program,
        "primal_candidate": _wire(point, digits),
        "primal_objective": _wire((objective,), digits)[0],
        "primal_residuals": _wire(residuals, digits),
    }
    if ray is not None:
        c, a, _ = _program_fractions(program)
        if (
            any(v < 0 for v in ray)
            or any(rational_dot(row, ray) for row in a)
            or rational_dot(c, ray) >= 0
        ):
            _execution_failure()
        return RationalLinearProgramResult._from_kernel(
            **values,
            status="UNBOUNDED",
            recession_direction=_wire(ray, digits),
        )
    dual_objective, slacks = _dual_diagnostics(program, dual)
    if any(v < 0 for v in slacks) or dual_objective != objective:
        _execution_failure()
    return RationalLinearProgramResult._from_kernel(
        **values,
        status="OPTIMAL",
        dual_candidate=_wire(dual, digits),
        dual_objective=_wire((dual_objective,), digits)[0],
        dual_slacks=_wire(slacks, digits),
    )


def _expand_vector(
    indices: tuple[int, ...], values: tuple[Fraction, ...], size: int
) -> tuple[Fraction, ...]:
    expanded = [Fraction()] * size
    for index, value in zip(indices, values, strict=True):
        expanded[index] = value
    return tuple(expanded)


def _component_programs(
    program: StandardFormRationalLinearProgram, admission: LinearAdmission
) -> RationalLinearProgramResult:
    width, height = len(program.variables), len(program.rhs)
    point, dual = [Fraction()] * width, [Fraction()] * height
    ray: tuple[Fraction, ...] | None = None
    for rows, columns in admission.components:
        request_checkpoint("linear-program component")
        component = StandardFormRationalLinearProgram.model_construct(
            variables=tuple(program.variables[j] for j in columns),
            objective=tuple(program.objective[j] for j in columns),
            coefficients=tuple(
                tuple(program.coefficients[i][j] for j in columns) for i in rows
            ),
            rhs=tuple(program.rhs[i] for i in rows),
        )
        result = _linear_program_admitted(
            component,
            LinearAdmission(tuple(range(len(columns))), admission.result_digits),
        )
        if result.status == "INFEASIBLE":
            assert result.farkas_candidate is not None
            return _certify_infeasible(
                program,
                _expand_vector(
                    rows,
                    tuple(v.as_fraction() for v in result.farkas_candidate),
                    height,
                ),
                admission.result_digits,
            )
        assert result.primal_candidate is not None
        for j, value in zip(columns, result.primal_candidate, strict=True):
            point[j] = value.as_fraction()
        if result.status == "UNBOUNDED":
            assert result.recession_direction is not None
            ray = _expand_vector(
                columns,
                tuple(v.as_fraction() for v in result.recession_direction),
                width,
            )
        else:
            assert result.dual_candidate is not None
            for i, value in zip(rows, result.dual_candidate, strict=True):
                dual[i] = value.as_fraction()
    for j, cost in enumerate(program.objective):
        if j not in admission.columns and cost.as_fraction() < 0:
            ray = _expand_vector((j,), (Fraction(1),), width)
            break
    return _certify_point(
        program, tuple(point), tuple(dual), ray, admission.result_digits
    )


def _linear_program_admitted(
    program: StandardFormRationalLinearProgram,
    admission: LinearAdmission,
) -> RationalLinearProgramResult:
    from flint import fmpq, fmpq_mat

    digits = admission.result_digits
    width, height = len(program.variables), len(program.rhs)
    zero = Fraction()
    for i, (row, rhs) in enumerate(zip(program.coefficients, program.rhs, strict=True)):
        if not any(v.num != 0 for v in row) and rhs.num != 0:
            witness = [zero] * height
            witness[i] = Fraction(1 if rhs.num < 0 else -1)
            return _certify_infeasible(program, tuple(witness), digits)
    if len(admission.components) > 1:
        return _component_programs(program, admission)
    columns = admission.columns
    active_rows = tuple(
        i for i, row in enumerate(program.coefficients) if any(v.num != 0 for v in row)
    )
    zero_ray = next(
        (j for j, c in enumerate(program.objective) if j not in columns and c.num < 0),
        None,
    )
    if not columns:
        ray = None
        if zero_ray is not None:
            ray_values = [zero] * width
            ray_values[zero_ray] = Fraction(1)
            ray = tuple(ray_values)
        return _certify_point(program, (zero,) * width, (zero,) * height, ray, digits)

    def scalar(v: CanonicalRational) -> Any:
        return fmpq(*v.as_integer_ratio())

    a = fmpq_mat(
        [[scalar(program.coefficients[i][j]) for j in columns] for i in active_rows]
    )
    b = fmpq_mat([[scalar(program.rhs[i])] for i in active_rows])
    row_indices, inconsistent = independent_rows(a, b)
    if inconsistent is not None:
        witness = [zero] * height
        for i, v in zip(active_rows, _fractions(inconsistent), strict=True):
            witness[i] = v
        return _certify_infeasible(program, tuple(witness), digits)
    reduced_a = fmpq_mat([[a[i, j] for j in range(len(columns))] for i in row_indices])
    reduced_b = fmpq_mat([[b[i, 0]] for i in row_indices])
    c = fmpq_mat([[scalar(program.objective[j]) for j in columns]])
    solved = search_bases(reduced_a, reduced_b, c)
    if solved is None:
        augmented = fmpq_mat(
            [
                [reduced_a[i, j] for j in range(len(columns))] + [reduced_b[i, 0]]
                for i in range(len(row_indices))
            ]
        )
        phase_one = search_bases(
            augmented, reduced_b, fmpq_mat([[0] * len(columns) + [1]]), artificial=True
        )
        if phase_one is None:
            _execution_failure()
        witness = [zero] * height
        for i, v in zip(row_indices, _fractions(phase_one[1]), strict=True):
            witness[active_rows[i]] = -v
        return _certify_infeasible(program, tuple(witness), digits)

    basic_point, basic_dual, basic_ray = solved
    point = _expand_vector(columns, _fractions(basic_point), width)
    dual = _expand_vector(
        tuple(active_rows[i] for i in row_indices), _fractions(basic_dual), height
    )
    ray = None
    if zero_ray is not None or basic_ray is not None:
        ray_values = [zero] * width
        if zero_ray is not None:
            ray_values[zero_ray] = Fraction(1)
        else:
            for j, v in zip(columns, _fractions(basic_ray), strict=True):
                ray_values[j] = v
        ray = tuple(ray_values)
    request_checkpoint("linear-program certificate construction")
    return _certify_point(program, tuple(point), tuple(dual), ray, digits)


def linear_program(
    program: StandardFormRationalLinearProgram,
) -> RationalLinearProgramResult:
    """Minimize c^T x over Ax=b, x>=0 with exact source certificates."""
    with linear_execution():
        return _linear_program_admitted(program, admit_linear_program(program))


__all__ = ["linear_program"]
