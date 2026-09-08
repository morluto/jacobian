"""Admission plan for the metric connection and covariant derivative DAG."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics._dag import Dag, Expression
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.metrics._plan import (
    ConnectionPlan,
    _denominator_guard_identity,
    _node_guard_key,
    build_connection_plan,
    potential_locus_guard_keys,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    MAX_RATIONAL_TENSOR_RANK,
    RationalCoordinateTensor,
    _polynomial_key,
)
from jacobian.math.polynomials.values import SparseRationalPolynomial


def _covariant_singular_metric() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("metric",),
        code="differential_geometry.covariant_derivative.singular_metric",
        message="metric determinant is identically zero",
    )


def _covariant_reject(reason: str, message: str) -> NoReturn:
    location = ("tensor",) if reason == "shape" else ("covariant_derivative",)
    raise OperationResourceAdmissionError(
        location=location,
        code=f"differential_geometry.covariant_derivative.{reason}",
        message=message,
    )


@dataclass(frozen=True)
class Plan:
    dag: Dag
    determinant: Expression
    inverse: tuple[Expression, ...]
    connection: tuple[Expression, ...]
    derivative: tuple[Expression, ...]
    fractions: dict[Expression, tuple[int, int]]


def _flatten(indices: tuple[int, ...], dimension: int) -> int:
    result = 0
    for index in indices:
        result = result * dimension + index
    return result


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


def _admit_outputs(
    dag: Dag,
    metric: RationalCoordinateMetric,
    tensor: RationalCoordinateTensor,
    determinant: Expression,
    outputs: tuple[Expression, ...],
) -> dict[Expression, tuple[int, int, int]]:
    sizes = {value: dag.admit_output(value) for value in dict.fromkeys(outputs)}
    guard_allocations: dict[object, tuple[int, int, int]] = {}
    for coordinate_tensor in (metric.tensor, tensor):
        for guard in coordinate_tensor.retained_nonzero_denominators:
            guard_allocations.setdefault(
                _polynomial_key(guard), _source_allocation(guard, dag.dimension)
            )
    for index in set(determinant.numerator):
        bound = dag.nodes[index].bound
        if (
            bound.terms > MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS
            or max(bound.degrees, default=0) > MAX_RATIONAL_TENSOR_EXPONENT
            or bound.coefficient_digits > MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS
        ):
            _covariant_reject(
                "determinant_locus",
                "determinant locus factors exceed canonical polynomial bounds",
            )
        key = _node_guard_key(dag, index)
        if key in guard_allocations:
            continue
        guard_allocations.setdefault(
            key,
            (
                bound.terms,
                8 * bound.coefficient_digits * bound.terms,
                dag.dimension * (bound.terms + 1),
            ),
        )
    potential_guards = len(
        potential_locus_guard_keys(
            dag,
            tuple(
                guard
                for coordinate_tensor in (metric.tensor, tensor)
                for guard in coordinate_tensor.retained_nonzero_denominators
            ),
            determinant,
            outputs,
        )
    )
    if potential_guards > 768:
        _covariant_reject(
            "locus", "complete covariant-derivative locus exceeds 768 guards"
        )

    dimension = dag.dimension
    source = [
        _source_allocation(polynomial, dimension)
        for coordinate_tensor in (metric.tensor, tensor)
        for component in coordinate_tensor.components
        for polynomial in (component.numerator, component.denominator)
    ]
    inherited = [
        _source_allocation(guard, dimension)
        for coordinate_tensor in (metric.tensor, tensor)
        for guard in coordinate_tensor.retained_nonzero_denominators
    ]
    for value, size in sizes.items():
        identity = _denominator_guard_identity(dag, value)
        if identity is not None:
            guard_allocations.setdefault(identity, size)
    guards = list(guard_allocations.values())
    allocations = source + inherited + [sizes[value] for value in outputs] + guards
    terms, coefficient_bits, coordinate_slots = (
        sum(allocation[index] for allocation in allocations) for index in range(3)
    )
    coordinate_slots += dimension * (len(outputs) + 3) + 4 * len(outputs)
    if (
        terms > 262_144
        or coefficient_bits > 268_435_456
        or coordinate_slots > 1_048_576
    ):
        _covariant_reject(
            "output",
            "covariant derivative exceeds polynomial term, coefficient-bit, "
            "or coordinate allocation bounds",
        )
    return sizes


def build_plan(
    metric: RationalCoordinateMetric, tensor: RationalCoordinateTensor
) -> Plan:
    dimension = len(metric.tensor.coordinate_axis)
    tensor_rank = len(tensor.variance)
    if tensor_rank + 1 > MAX_RATIONAL_TENSOR_RANK:
        _covariant_reject(
            "shape",
            "covariant derivative exceeds the rank-8 representation budget",
        )
    if dimension ** (tensor_rank + 1) > MAX_RATIONAL_TENSOR_COMPONENTS:
        _covariant_reject(
            "shape",
            "covariant derivative exceeds the dense component representation budget",
        )
    connection_plan: ConnectionPlan = build_connection_plan(
        metric,
        reject=_covariant_reject,
        label="covariant derivative",
        singular_metric=_covariant_singular_metric,
    )
    dag = connection_plan.dag
    axes = tuple(range(dimension))
    determinant = connection_plan.determinant
    inverse = connection_plan.inverse
    connection = connection_plan.connection
    source_entries = tuple(dag.fraction(value) for value in tensor.components)
    output_values = []
    source_indices = tuple(product(axes, repeat=tensor_rank))
    for axis in axes:
        for indices in source_indices:
            source_index = _flatten(indices, dimension)
            terms = [dag.derivative(source_entries[source_index], axis)]
            for position, variance in enumerate(tensor.variance):
                for replacement in axes:
                    changed = (
                        *indices[:position],
                        replacement,
                        *indices[position + 1 :],
                    )
                    changed_index = _flatten(changed, dimension)
                    if variance == "CONTRAVARIANT":
                        gamma_index = (
                            indices[position] * dimension + axis
                        ) * dimension + replacement
                        terms.append(
                            dag.multiply(
                                connection[gamma_index], source_entries[changed_index]
                            )
                        )
                    else:
                        gamma_index = (
                            replacement * dimension + axis
                        ) * dimension + indices[position]
                        terms.append(
                            dag.multiply(
                                Expression(Fraction(-1)),
                                connection[gamma_index],
                                source_entries[changed_index],
                            )
                        )
            output_values.append(dag.add(*terms))
    outputs = tuple(output_values)
    sizes = _admit_outputs(dag, metric, tensor, determinant, outputs)
    fractions = {
        value: (
            dag.polynomial(value.numerator, value.scalar),
            dag.polynomial(value.denominator),
        )
        for value in sizes
    }
    return Plan(dag, determinant, inverse, connection, outputs, fractions)


__all__ = ["Plan", "build_plan"]
