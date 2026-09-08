"""Admission plan for the direct rational Laplace--Beltrami contraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics._dag import (
    Dag,
    Expression,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._plan import (
    _complete_result_denominator_keys,
    _denominator_guard_identity,
    _monic_polynomial_key,
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


_GuardKey = object


def _laplace_singular_metric() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("metric",),
        code="differential_geometry.laplace_beltrami.singular_metric",
        message="metric determinant is identically zero",
    )


def _laplace_reject_at(
    location: tuple[str, ...],
) -> Callable[[str, str], NoReturn]:
    def reject(reason: str, message: str) -> NoReturn:
        raise OperationResourceAdmissionError(
            location=location,
            code=f"differential_geometry.laplace_beltrami.{reason}",
            message=message,
        )

    return reject


_laplace_metric_reject = _laplace_reject_at(("metric",))
_laplace_scalar_reject = _laplace_reject_at(("scalar",))


def _laplace_reject(reason: str, message: str) -> NoReturn:
    _laplace_metric_reject(reason, message)


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
    connection_plan = build_connection_plan(
        metric,
        admission_reject=_laplace_metric_reject,
        label="Laplace--Beltrami",
        singular_metric=_laplace_singular_metric,
    )
    dag = connection_plan.dag
    dag.ledger.limits = replace(
        dag.ledger.limits,
        reject=_laplace_scalar_reject,
        label="Laplace--Beltrami",
    )
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
    extra_keys: set[_GuardKey] = set()
    if not _is_unit_polynomial(scalar.denominator, dimension):
        extra_keys.add(_monic_polynomial_key(scalar.denominator))
    guard_keys: set[_GuardKey] = {
        _polynomial_key(guard) for guard in metric.tensor.retained_nonzero_denominators
    }
    sourced_keys = _complete_result_denominator_keys(dag, value)
    inherited_matches = sourced_keys & guard_keys
    result_identity = _denominator_guard_identity(dag, value)
    if inherited_matches:
        extra_keys.update(inherited_matches)
        result_identity = None
    elif result_identity is not None:
        extra_keys.add(result_identity)
    returned: dict[_GuardKey, tuple[int, int, int]] = {
        _polynomial_key(guard): _source_allocation(guard, dimension)
        for guard in metric.tensor.retained_nonzero_denominators
    }
    if not _is_unit_polynomial(scalar.denominator, dimension):
        returned.setdefault(
            _monic_polynomial_key(scalar.denominator),
            _source_allocation(scalar.denominator, dimension),
        )
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
        node_source = dag.nodes[index].source
        if node_source is not None:
            key: _GuardKey = _monic_polynomial_key(node_source)
        else:
            key = ("determinant", index)
        guard_keys.add(key)
        returned.setdefault(key, _bound_allocation(bound, dimension))
    if result_identity is not None:
        denominator_bound = dag.nodes[dag.polynomial(value.denominator)].bound
        returned.setdefault(
            result_identity, _bound_allocation(denominator_bound, dimension)
        )
    guard_keys.update(extra_keys)
    if len(guard_keys) > MAX_RATIONAL_TENSOR_LOCUS_GUARDS:
        _laplace_reject(
            "locus",
            "complete retained Laplace--Beltrami locus exceeds 768 guards",
        )
    source_sizes = [
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
    allocations = source_sizes + inherited + [size] + list(returned.values())
    terms, bits, slots = (
        sum(allocation[index] for allocation in allocations) for index in range(3)
    )
    slots += dimension * 8 + 8
    if (
        terms > MAX_LAPLACE_OUTPUT_TERMS
        or bits > MAX_LAPLACE_OUTPUT_BITS
        or slots > MAX_LAPLACE_OUTPUT_SLOTS
    ):
        _laplace_scalar_reject(
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
