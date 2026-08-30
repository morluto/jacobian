"""Private exact SymPy polynomial kernel for rational Lie derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.math.geometry.differential._bounds import (
    FactorReference,
    LieDerivativePlan,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    RationalCoordinateTensor,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction


@dataclass(frozen=True)
class _RawFraction:
    numerator: Any
    denominator: Any

    @property
    def is_zero(self) -> bool:
        return bool(self.numerator.is_zero)


def _zero_fraction(template: Any) -> _RawFraction:
    return _RawFraction(template.zero, template.one)


def _raw_fraction(value: RationalFunction) -> _RawFraction:
    return _RawFraction(
        sparse_rational_polynomial_to_sympy(value.numerator, value.variables),
        sparse_rational_polynomial_to_sympy(value.denominator, value.variables),
    )


def _differentiate(source: _RawFraction, axis: int) -> _RawFraction:
    numerator_derivative = source.numerator.diff(axis)
    denominator_derivative = source.denominator.diff(axis)
    numerator = (
        numerator_derivative * source.denominator
        - source.numerator * denominator_derivative
    )
    if numerator.is_zero:
        return _zero_fraction(source.denominator)
    return _RawFraction(numerator, source.denominator * source.denominator)


def _multiply(left: _RawFraction, right: _RawFraction) -> _RawFraction:
    if left.is_zero:
        return _zero_fraction(left.denominator)
    if right.is_zero:
        return _zero_fraction(right.denominator)
    return _RawFraction(
        left.numerator * right.numerator,
        left.denominator * right.denominator,
    )


def _add(left: _RawFraction, right: _RawFraction) -> _RawFraction:
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    numerator = left.numerator * right.denominator + right.numerator * left.denominator
    if numerator.is_zero:
        return _zero_fraction(left.denominator)
    return _RawFraction(numerator, left.denominator * right.denominator)


def _factor(
    reference: FactorReference,
    *,
    vector_values: tuple[_RawFraction, ...],
    tensor_values: tuple[_RawFraction, ...],
    vector_derivatives: dict[tuple[int, int], _RawFraction],
) -> _RawFraction:
    if reference.owner == "VECTOR":
        if reference.derivative_axis is None:
            return vector_values[reference.component]
        return vector_derivatives[(reference.component, reference.derivative_axis)]
    if reference.derivative_axis is None:
        return tensor_values[reference.component]
    # Every tensor-component derivative occurs in exactly one directional
    # term.  Compute it at that use site instead of retaining the full
    # component-by-coordinate derivative table.
    return _differentiate(tensor_values[reference.component], reference.derivative_axis)


def compute_lie_derivative_components(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
    plan: LieDerivativePlan,
) -> tuple[RationalFunction, ...]:
    """Execute one already-admitted complete coordinate formula."""

    vector_values = tuple(_raw_fraction(value) for value in vector_field.components)
    tensor_values = tuple(_raw_fraction(value) for value in tensor.components)
    dimension = len(tensor.coordinate_axis)
    vector_derivatives = {
        (component, axis): _differentiate(vector_values[component], axis)
        for component in range(dimension)
        for axis in range(dimension)
    }
    results: list[RationalFunction] = []
    for component_plan in plan.components:
        accumulator = _zero_fraction(vector_values[0].denominator)
        for term in component_plan.terms:
            value = _multiply(
                _factor(
                    term.left,
                    vector_values=vector_values,
                    tensor_values=tensor_values,
                    vector_derivatives=vector_derivatives,
                ),
                _factor(
                    term.right,
                    vector_values=vector_values,
                    tensor_values=tensor_values,
                    vector_derivatives=vector_derivatives,
                ),
            )
            if term.sign < 0 and not value.is_zero:
                value = _RawFraction(-value.numerator, value.denominator)
            accumulator = _add(accumulator, value)
        expression = accumulator.numerator.as_expr() / accumulator.denominator.as_expr()
        results.append(
            rational_function_from_sympy(
                expression,
                tensor.coordinate_axis,
                maximum_terms=MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
            )
        )
    return tuple(results)


__all__ = ["compute_lie_derivative_components"]
