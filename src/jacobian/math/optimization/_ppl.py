"""Exact standard-form linear optimization through Parma Polyhedra Library.

PPL owns polyhedral optimization and generator enumeration.  This adapter returns
ordinary exact values only; the operation owner checks every defining identity
before constructing a public result.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Any


@dataclass(frozen=True)
class ExactLinearOutcome:
    status: str
    point: tuple[Fraction, ...] = ()
    dual: tuple[Fraction, ...] = ()
    witness: tuple[Fraction, ...] = ()
    ray: tuple[Fraction, ...] = ()


def _integer_row(values: tuple[Fraction, ...]) -> tuple[tuple[int, ...], int]:
    denominator = 1
    for value in values:
        denominator = lcm(denominator, value.denominator)
    return tuple(
        value.numerator * (denominator // value.denominator) for value in values
    ), denominator


def _expression(coefficients: tuple[int, ...]) -> Any:
    from ppl import Linear_Expression, Variable  # type: ignore[import-untyped]

    expression = Linear_Expression(0)
    for index, coefficient in enumerate(coefficients):
        if coefficient:
            expression += coefficient * Variable(index)
    return expression


def _point(generator: Any, dimension: int) -> tuple[Fraction, ...]:
    from ppl import Variable

    divisor = int(generator.divisor())
    return tuple(
        Fraction(int(generator.coefficient(Variable(index))), divisor)
        for index in range(dimension)
    )


def _ray(generator: Any, dimension: int) -> tuple[Fraction, ...]:
    from ppl import Variable

    return tuple(
        Fraction(int(generator.coefficient(Variable(index))))
        for index in range(dimension)
    )


def _some_point(polyhedron: Any, dimension: int) -> tuple[Fraction, ...]:
    for generator in polyhedron.minimized_generators():
        if generator.is_point():
            return _point(generator, dimension)
    raise RuntimeError("nonempty PPL polyhedron supplied no point generator")


def solve_standard_form(
    objective: tuple[Fraction, ...],
    coefficients: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
) -> ExactLinearOutcome:
    """Minimize ``objective*x`` subject to ``coefficients*x=rhs, x>=0``."""
    from ppl import C_Polyhedron, Variable

    variables, equations = len(objective), len(rhs)
    primal = C_Polyhedron(variables, "universe")
    for index in range(variables):
        primal.add_constraint(Variable(index) >= 0)
    for row, value in zip(coefficients, rhs, strict=True):
        integers, denominator = _integer_row((*row, value))
        primal.add_constraint(_expression(integers[:-1]) == integers[-1])
    if primal.is_empty():
        # Farkas alternative: A^T y >= 0 and b^T y <= -1.  Any witness with
        # b^T y < 0 certifies infeasibility; normalization makes the set closed.
        farkas = C_Polyhedron(equations, "universe")
        for column in range(variables):
            values = tuple(coefficients[row][column] for row in range(equations))
            integers, _ = _integer_row(values)
            farkas.add_constraint(_expression(integers) >= 0)
        integers, denominator = _integer_row(rhs)
        farkas.add_constraint(_expression(integers) <= -denominator)
        if farkas.is_empty():
            raise RuntimeError("PPL infeasibility had no Farkas alternative")
        return ExactLinearOutcome(
            status="INFEASIBLE", witness=_some_point(farkas, equations)
        )

    objective_integers, _ = _integer_row(objective)
    optimized = primal.minimize(_expression(objective_integers))
    if not optimized["bounded"]:
        point = _some_point(primal, variables)
        for generator in primal.minimized_generators():
            if generator.is_ray():
                ray = _ray(generator, variables)
                if (
                    sum(
                        (
                            cost * value
                            for cost, value in zip(objective, ray, strict=True)
                        ),
                        Fraction(),
                    )
                    < 0
                ):
                    return ExactLinearOutcome(status="UNBOUNDED", point=point, ray=ray)
        raise RuntimeError("PPL reported unboundedness without an improving ray")

    point = _point(optimized["generator"], variables)
    # Solve the exact dual max b^T y subject to A^T y <= c.  Strong duality
    # supplies a checkable optimality witness in source equation coordinates.
    dual = C_Polyhedron(equations, "universe")
    for column, cost in enumerate(objective):
        values = tuple(coefficients[row][column] for row in range(equations))
        integers, denominator = _integer_row((*values, cost))
        dual.add_constraint(_expression(integers[:-1]) <= integers[-1])
    rhs_integers, _ = _integer_row(rhs)
    dual_optimized = dual.maximize(_expression(rhs_integers))
    if not dual_optimized["bounded"]:
        raise RuntimeError("PPL primal optimum had no bounded dual")
    return ExactLinearOutcome(
        status="OPTIMAL",
        point=point,
        dual=_point(dual_optimized["generator"], equations),
    )
