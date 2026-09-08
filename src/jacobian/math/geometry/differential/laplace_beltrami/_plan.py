"""Admission plan for the direct rational Laplace--Beltrami contraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential.metrics._dag import (
    Dag,
    Expression,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._plan import (
    _has_nonconstant_denominator,
    build_connection_plan,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    _is_unit_polynomial,
    _polynomial_key,
)
from jacobian.math.polynomials.rational_functions._bounds import PolynomialBound
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial

MAX_LAPLACE_OUTPUT_TERMS = 262_144
MAX_LAPLACE_OUTPUT_BITS = 268_435_456
MAX_LAPLACE_OUTPUT_SLOTS = 1_048_576


def _monic_polynomial_key(polynomial: SparseRationalPolynomial) -> object:
    if not polynomial.terms:
        return _polynomial_key(polynomial)
    leading = polynomial.terms[0].coefficient.as_fraction()
    return tuple(
        (
            term.exponents,
            format_canonical_integer(
                (term.coefficient.as_fraction() / leading).numerator
            ),
            format_canonical_integer(
                (term.coefficient.as_fraction() / leading).denominator
            ),
        )
        for term in polynomial.terms
    )


def _laplace_reject(reason: str, message: str) -> None:
    location = ("scalar",) if reason == "result_exponent" else ("metric",)
    raise OperationResourceAdmissionError(
        location=location,
        code=f"differential_geometry.laplace_beltrami.{reason}",
        message=message,
    )


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
    dag.ledger.limits = replace(
        dag.ledger.limits,
        reject=_laplace_reject,
        label="Laplace--Beltrami",
    )
    size = dag.admit_output(value)
    determinant_sizes: list[tuple[int, int, int]] = []
    for index in set(connection_plan.determinant.numerator):
        bound = dag.nodes[index].bound
        if (
            bound.terms > MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS
            or max(bound.degrees) > MAX_RATIONAL_TENSOR_EXPONENT
            or bound.coefficient_digits > MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS
        ):
            _laplace_reject(
                "determinant_locus",
                "determinant locus factors exceed canonical polynomial bounds",
            )
        determinant_sizes.append(_bound_allocation(bound, dimension))
    extra_keys: set[object] = set()
    if not _is_unit_polynomial(scalar.denominator, dimension):
        extra_keys.add(_monic_polynomial_key(scalar.denominator))
    if _has_nonconstant_denominator(dag, value):
        extra_keys.add(("canonical-result-denominator",))
    guard_keys = {
        _polynomial_key(guard) for guard in metric.tensor.retained_nonzero_denominators
    }
    for index in set(connection_plan.determinant.numerator):
        source = dag.nodes[index].source
        if source is not None:
            guard_keys.add(_monic_polynomial_key(source))
        else:
            guard_keys.add(("determinant", index))
    guard_keys.update(extra_keys)
    if len(guard_keys) > MAX_RATIONAL_TENSOR_LOCUS_GUARDS:
        _laplace_reject(
            "locus",
            "complete retained Laplace--Beltrami locus exceeds 768 guards",
        )
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
        _laplace_reject(
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
