"""Admission plan for the direct rational Laplace--Beltrami contraction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian.math.geometry.differential.metrics._dag import (
    Dag,
    Expression,
    reject,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._plan import build_connection_plan
from jacobian.math.polynomials.rational_functions._bounds import PolynomialBound
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial

MAX_LAPLACE_OUTPUT_TERMS = 262_144
MAX_LAPLACE_OUTPUT_BITS = 268_435_456
MAX_LAPLACE_OUTPUT_SLOTS = 1_048_576


@dataclass(frozen=True)
class Plan:
    dag: Dag
    determinant: Expression
    value: Expression
    fraction: tuple[int, int]


def _source_allocation(
    polynomial: SparseRationalPolynomial, dimension: int
) -> tuple[int, int, int]:
    return (
        len(polynomial.terms),
        sum(
            abs(term.coefficient.num).bit_length() + term.coefficient.den.bit_length()
            for term in polynomial.terms
        ),
        dimension * (len(polynomial.terms) + 1),
    )


def _bound_allocation(bound: PolynomialBound, dimension: int) -> tuple[int, int, int]:
    """Price a planned polynomial guard before backend materialization."""
    return (
        bound.terms,
        8 * bound.coefficient_digits * bound.terms,
        dimension * (bound.terms + 1),
    )


def build_plan(metric: RationalCoordinateMetric, scalar: RationalFunction) -> Plan:
    dimension = len(metric.tensor.coordinate_axis)
    connection_plan = build_connection_plan(metric)
    dag = connection_plan.dag
    field = dag.fraction(scalar)
    axes = tuple(range(dimension))
    value_terms: list[Expression] = []
    first = tuple(dag.derivative(field, axis) for axis in axes)
    for i in axes:
        for j in axes:
            bracket_terms = [dag.derivative(first[i], j)]
            for k in axes:
                bracket_terms.append(
                    dag.multiply(
                        Expression(Fraction(-1)),
                        connection_plan.connection[(k * dimension + i) * dimension + j],
                        first[k],
                    )
                )
            value_terms.append(
                dag.multiply(
                    connection_plan.inverse[i * dimension + j],
                    dag.add(*bracket_terms),
                )
            )
    value = dag.add(*value_terms)
    size = dag.admit_output(value)
    determinant_sizes = [
        _bound_allocation(dag.nodes[index].bound, dimension)
        for index in set(connection_plan.determinant.numerator)
    ]
    source = [
        _source_allocation(polynomial, dimension)
        for polynomial in (
            *(
                polynomial
                for component in metric.tensor.components
                for polynomial in (component.numerator, component.denominator)
            ),
            scalar.numerator,
            scalar.denominator,
        )
    ]
    inherited = [
        _source_allocation(guard, dimension)
        for guard in metric.tensor.retained_nonzero_denominators
    ]
    allocations = source + inherited + determinant_sizes + [size]
    terms, bits, slots = (
        sum(allocation[index] for allocation in allocations) for index in range(3)
    )
    slots += dimension * 8 + 8
    if (
        terms > MAX_LAPLACE_OUTPUT_TERMS
        or bits > MAX_LAPLACE_OUTPUT_BITS
        or slots > MAX_LAPLACE_OUTPUT_SLOTS
    ):
        reject(
            "output",
            "Laplace--Beltrami source and result exceed polynomial term, "
            "coefficient-bit, or coordinate allocation bounds",
        )
    fraction = (
        dag.polynomial(value.numerator, value.scalar),
        dag.polynomial(value.denominator),
    )
    return Plan(dag, connection_plan.determinant, value, fraction)


__all__ = ["Plan", "build_plan"]
