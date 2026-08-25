"""Bounded rational optimization operations backed by SymPy."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.optimization._general_operations import (
    GENERAL_RATIONAL_LINEAR_OPERATIONS,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
    _active_equations,
    _dual_diagnostics,
    _primal_diagnostics,
    _result_digit_bound,
)


def _rational(value: CanonicalRational) -> Any:
    import sympy

    return sympy.Rational(value.as_fraction())


def _linear(coefficients: tuple[Any, ...], symbols: tuple[Any, ...]) -> Any:
    import sympy

    return sum(
        (
            coefficient * symbol
            for coefficient, symbol in zip(coefficients, symbols, strict=True)
        ),
        sympy.S.Zero,
    )


def _wire(value: Any, *, max_digits: int) -> CanonicalRational | None:
    import sympy

    try:
        rational = sympy.Rational(value)
        numerator = int(rational.p)
        denominator = int(rational.q)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if len(str(abs(numerator))) > max_digits or len(str(denominator)) > max_digits:
        return None
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _wire_fraction(
    value: Fraction,
    *,
    max_digits: int,
) -> CanonicalRational | None:
    if (
        len(str(abs(value.numerator))) > max_digits
        or len(str(value.denominator)) > max_digits
    ):
        return None
    return CanonicalRational.from_fraction(value)


def _wire_solution(
    solution: dict[Any, Any],
    symbols: tuple[Any, ...],
    *,
    max_digits: int,
) -> tuple[CanonicalRational, ...] | None:
    import sympy

    values: list[CanonicalRational] = []
    for symbol in symbols:
        value = _wire(solution.get(symbol, sympy.S.Zero), max_digits=max_digits)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _wire_difference_solution(
    solution: dict[Any, Any],
    positive_symbols: tuple[Any, ...],
    negative_symbols: tuple[Any, ...],
    *,
    max_digits: int,
) -> tuple[CanonicalRational, ...] | None:
    import sympy

    values: list[CanonicalRational] = []
    for positive, negative in zip(positive_symbols, negative_symbols, strict=True):
        value = _wire(
            solution.get(positive, sympy.S.Zero) - solution.get(negative, sympy.S.Zero),
            max_digits=max_digits,
        )
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _program_arrays(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...], tuple[Any, ...]]:
    return (
        tuple(_rational(value) for value in program.objective),
        tuple(tuple(_rational(value) for value in row) for row in program.coefficients),
        tuple(_rational(value) for value in program.rhs),
    )


def _active_row_indices(program: StandardFormRationalLinearProgram) -> tuple[int, ...]:
    """Return source rows that impose a nontrivial equality."""

    return tuple(
        index
        for index, (row, rhs) in enumerate(
            zip(program.coefficients, program.rhs, strict=True)
        )
        if any(value.num != "0" for value in row) or rhs.num != "0"
    )


def _expand_active_row_values(
    values: tuple[CanonicalRational, ...],
    indices: tuple[int, ...],
    *,
    source_rows: int,
) -> tuple[CanonicalRational, ...]:
    """Embed an auxiliary-row vector back into the source row coordinates."""

    zero = CanonicalRational.from_integer_ratio(0, 1)
    expanded = [zero] * source_rows
    for index, value in zip(indices, values, strict=True):
        expanded[index] = value
    return tuple(expanded)


def _solve_primal(
    program: StandardFormRationalLinearProgram,
    objective: tuple[Any, ...],
    *,
    symbol_prefix: str,
) -> tuple[tuple[Any, ...], dict[Any, Any]]:
    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, lpmin

    _, coefficients, rhs = _program_arrays(program)
    symbols = tuple(sympy.symbols(f"{symbol_prefix}0:{len(program.variables)}"))
    constraints = []
    for row, expected in zip(coefficients, rhs, strict=True):
        if any(row):
            constraints.append(
                sympy.Eq(_linear(row, symbols), expected, evaluate=False)
            )
        elif expected:
            raise InfeasibleLPError("zero coefficient row has a nonzero rhs")
    constraints.extend(
        sympy.Ge(symbol, sympy.S.Zero, evaluate=False) for symbol in symbols
    )
    _, solution = lpmin(_linear(objective, symbols), constraints)
    return symbols, solution


def _solve_dual(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[int, ...], tuple[Any, ...], dict[Any, Any]]:
    import sympy
    from sympy.solvers.simplex import lpmax

    objective, all_coefficients, all_rhs = _program_arrays(program)
    indices = _active_row_indices(program)
    coefficients = tuple(all_coefficients[index] for index in indices)
    rhs = tuple(all_rhs[index] for index in indices)
    symbols = tuple(sympy.symbols(f"_lp_dual0:{len(rhs)}"))
    constraints = [
        sympy.Le(
            _linear(
                tuple(row[column] for row in coefficients),
                symbols,
            ),
            objective[column],
            evaluate=False,
        )
        for column in range(len(objective))
    ]
    _, solution = lpmax(_linear(rhs, symbols), constraints)
    return indices, symbols, solution


def _solve_farkas(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[int, ...], tuple[Any, ...], tuple[Any, ...], dict[Any, Any]]:
    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, lpmin

    _, all_coefficients, all_rhs = _program_arrays(program)
    indices = _active_row_indices(program)
    coefficients = tuple(all_coefficients[index] for index in indices)
    rhs = tuple(all_rhs[index] for index in indices)
    if not any(rhs):
        raise InfeasibleLPError("Farkas normalization requires a nonzero rhs")
    positive = tuple(sympy.symbols(f"_lp_farkas_positive0:{len(rhs)}"))
    negative = tuple(sympy.symbols(f"_lp_farkas_negative0:{len(rhs)}"))
    symbols = tuple(
        positive_value - negative_value
        for positive_value, negative_value in zip(positive, negative, strict=True)
    )
    constraints = [
        sympy.Ge(
            _linear(
                tuple(row[column] for row in coefficients),
                symbols,
            ),
            sympy.S.Zero,
            evaluate=False,
        )
        for column in range(len(program.variables))
    ]
    constraints.append(
        sympy.Eq(
            _linear(rhs, symbols),
            -sympy.S.One,
            evaluate=False,
        )
    )
    constraints.extend(
        sympy.Ge(symbol, sympy.S.Zero, evaluate=False)
        for symbol in (*positive, *negative)
    )
    _, solution = lpmin(sum((*positive, *negative), sympy.S.Zero), constraints)
    return indices, positive, negative, solution


def _solve_recession_direction(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[Any, ...], dict[Any, Any]]:
    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, lpmin

    objective, coefficients, _ = _program_arrays(program)
    if not any(objective):
        raise InfeasibleLPError("a zero objective has no decreasing direction")
    symbols = tuple(sympy.symbols(f"_lp_ray0:{len(program.variables)}"))
    constraints = [
        sympy.Eq(_linear(row, symbols), sympy.S.Zero, evaluate=False)
        for row in coefficients
        if any(row)
    ]
    constraints.append(
        sympy.Eq(
            _linear(objective, symbols),
            -sympy.S.One,
            evaluate=False,
        )
    )
    constraints.extend(
        sympy.Ge(symbol, sympy.S.Zero, evaluate=False) for symbol in symbols
    )
    one_objective = tuple(sympy.S.One for _ in symbols)
    _, solution = lpmin(_linear(one_objective, symbols), constraints)
    return symbols, solution


def _primal_data(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[CanonicalRational, ...],
    *,
    max_digits: int,
) -> tuple[CanonicalRational, tuple[CanonicalRational, ...]] | None:
    values = tuple(value.as_fraction() for value in candidate)
    objective, residuals = _primal_diagnostics(program, values)
    wire_objective = _wire_fraction(objective, max_digits=max_digits)
    wire_residuals = tuple(
        _wire_fraction(value, max_digits=max_digits) for value in residuals
    )
    if wire_objective is None or any(value is None for value in wire_residuals):
        return None
    return wire_objective, tuple(value for value in wire_residuals if value is not None)


def _dual_data(
    program: StandardFormRationalLinearProgram,
    candidate: tuple[CanonicalRational, ...],
    *,
    max_digits: int,
) -> tuple[CanonicalRational, tuple[CanonicalRational, ...]] | None:
    values = tuple(value.as_fraction() for value in candidate)
    objective, slacks = _dual_diagnostics(program, values)
    wire_objective = _wire_fraction(objective, max_digits=max_digits)
    wire_slacks = tuple(
        _wire_fraction(value, max_digits=max_digits) for value in slacks
    )
    if wire_objective is None or any(value is None for value in wire_slacks):
        return None
    return wire_objective, tuple(value for value in wire_slacks if value is not None)


def _unknown(
    program: StandardFormRationalLinearProgram,
) -> RationalLinearProgramResult:
    return RationalLinearProgramResult(program=program, status="UNKNOWN")


def _trivial_infeasibility(
    program: StandardFormRationalLinearProgram,
) -> RationalLinearProgramResult | None:
    zero = CanonicalRational.from_integer_ratio(0, 1)
    for row_index, (row, rhs) in enumerate(
        zip(program.coefficients, program.rhs, strict=True)
    ):
        if any(value.num != "0" for value in row) or rhs.num == "0":
            continue
        witness = [zero] * len(program.rhs)
        witness[row_index] = CanonicalRational.from_integer_ratio(
            1 if rhs.num.startswith("-") else -1,
            1,
        )
        return RationalLinearProgramResult(
            program=program,
            status="INFEASIBLE",
            farkas_candidate=tuple(witness),
        )
    return None


def _certify_infeasible(
    program: StandardFormRationalLinearProgram,
    *,
    result_digits: int,
) -> RationalLinearProgramResult:
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError

    try:
        active_rows, positive_symbols, negative_symbols, farkas_solution = (
            _solve_farkas(program)
        )
    except (
        InfeasibleLPError,
        UnboundedLPError,
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return _unknown(program)
    active_farkas = _wire_difference_solution(
        farkas_solution,
        positive_symbols,
        negative_symbols,
        max_digits=result_digits,
    )
    if active_farkas is None:
        return _unknown(program)
    farkas = _expand_active_row_values(
        active_farkas,
        active_rows,
        source_rows=len(program.rhs),
    )
    try:
        return RationalLinearProgramResult(
            program=program,
            status="INFEASIBLE",
            farkas_candidate=farkas,
        )
    except ValidationError:
        return _unknown(program)


def _certify_unbounded(
    program: StandardFormRationalLinearProgram,
    *,
    result_digits: int,
) -> RationalLinearProgramResult:
    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError

    feasibility_objective = tuple(sympy.S.One for _ in program.variables)
    try:
        feasible_symbols, feasible_solution = _solve_primal(
            program,
            feasibility_objective,
            symbol_prefix="_lp_feasible",
        )
        ray_symbols, ray_solution = _solve_recession_direction(program)
    except (
        InfeasibleLPError,
        UnboundedLPError,
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return _unknown(program)
    feasible = _wire_solution(
        feasible_solution,
        feasible_symbols,
        max_digits=result_digits,
    )
    ray = _wire_solution(
        ray_solution,
        ray_symbols,
        max_digits=result_digits,
    )
    if feasible is None or ray is None:
        return _unknown(program)
    primal_data = _primal_data(
        program,
        feasible,
        max_digits=result_digits,
    )
    if primal_data is None:
        return _unknown(program)
    primal_objective, primal_residuals = primal_data
    try:
        return RationalLinearProgramResult(
            program=program,
            status="UNBOUNDED",
            primal_candidate=feasible,
            primal_objective=primal_objective,
            primal_residuals=primal_residuals,
            recession_direction=ray,
        )
    except ValidationError:
        return _unknown(program)


def _positive_result(
    program: StandardFormRationalLinearProgram,
    primal_symbols: tuple[Any, ...],
    primal_solution: dict[Any, Any],
    *,
    result_digits: int,
) -> RationalLinearProgramResult:
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError

    primal = _wire_solution(
        primal_solution,
        primal_symbols,
        max_digits=result_digits,
    )
    if primal is None:
        return _certify_infeasible(program, result_digits=result_digits)
    primal_data = _primal_data(program, primal, max_digits=result_digits)
    if primal_data is None:
        return _certify_infeasible(program, result_digits=result_digits)
    primal_objective, primal_residuals = primal_data
    try:
        primal_result = RationalLinearProgramResult(
            program=program,
            status="PRIMAL_FEASIBLE",
            primal_candidate=primal,
            primal_objective=primal_objective,
            primal_residuals=primal_residuals,
        )
    except ValidationError:
        return _certify_infeasible(program, result_digits=result_digits)

    if not _active_equations(program):
        dual = tuple(CanonicalRational.from_integer_ratio(0, 1) for _ in program.rhs)
    else:
        try:
            active_rows, dual_symbols, dual_solution = _solve_dual(program)
        except (
            InfeasibleLPError,
            UnboundedLPError,
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            return primal_result
        converted_dual = _wire_solution(
            dual_solution,
            dual_symbols,
            max_digits=result_digits,
        )
        if converted_dual is None:
            return primal_result
        dual = _expand_active_row_values(
            converted_dual,
            active_rows,
            source_rows=len(program.rhs),
        )
    dual_data = _dual_data(program, dual, max_digits=result_digits)
    if dual_data is None:
        return primal_result
    dual_objective, dual_slacks = dual_data
    try:
        return RationalLinearProgramResult(
            program=program,
            status="OPTIMAL",
            primal_candidate=primal,
            primal_objective=primal_objective,
            primal_residuals=primal_residuals,
            dual_candidate=dual,
            dual_objective=dual_objective,
            dual_slacks=dual_slacks,
        )
    except ValidationError:
        return primal_result


def _linear_program(
    request: RationalLinearProgramRequest,
) -> RationalLinearProgramResult:
    import sympy
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError

    program = request.program
    trivial_infeasibility = _trivial_infeasibility(program)
    if trivial_infeasibility is not None:
        return trivial_infeasibility
    result_digits = _result_digit_bound(program)
    objective, _, _ = _program_arrays(program)
    backend_objective = (
        objective if any(objective) else tuple(sympy.S.One for _ in program.variables)
    )
    try:
        primal_symbols, primal_solution = _solve_primal(
            program,
            backend_objective,
            symbol_prefix="_lp_primal",
        )
    except InfeasibleLPError:
        return _certify_infeasible(program, result_digits=result_digits)
    except UnboundedLPError:
        return _certify_unbounded(program, result_digits=result_digits)
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return _unknown(program)
    return _positive_result(
        program,
        primal_symbols,
        primal_solution,
        result_digits=result_digits,
    )


RATIONAL_LINEAR_OPERATIONS: MathTools = (
    MathTool(
        operation_id="optimization.linear.rational_optimum.compute",
        title="Solve a rational linear program",
        description=(
            "Use exact SymPy simplex calls to return a source-bound standard-form "
            "rational LP outcome. Optimal and feasible outcomes retain replayed "
            "points; infeasible outcomes carry a Farkas witness; unbounded outcomes "
            "carry a feasible point and recession direction; UNKNOWN makes no claim."
        ),
        request_type=RationalLinearProgramRequest,
        result_type=RationalLinearProgramResult,
        run=_linear_program,
        tags=(
            "optimization",
            "linear-program",
            "rational",
            "optimum",
            "bounded",
        ),
        examples=(
            example(
                "one_variable_unit_lp",
                "Optimize x subject to x=1 and x>=0.",
                {
                    "program": {
                        "variables": ["x"],
                        "objective": [{"num": "1", "den": "1"}],
                        "coefficients": [[{"num": "1", "den": "1"}]],
                        "rhs": [{"num": "1", "den": "1"}],
                    },
                },
            ),
        ),
    ),
)

RATIONAL_LINEAR_OPERATIONS = (
    *RATIONAL_LINEAR_OPERATIONS,
    *GENERAL_RATIONAL_LINEAR_OPERATIONS,
)

__all__ = ["RATIONAL_LINEAR_OPERATIONS"]
