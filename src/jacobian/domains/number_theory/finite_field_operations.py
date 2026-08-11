"""Bounded exact arithmetic for finite extension-field polynomial maps."""

from __future__ import annotations

import hashlib
from collections import Counter

from jacobian.contracts.number_theory import (
    FiniteFieldCollisionWitness,
    FiniteFieldFiberCount,
    FiniteFieldPolynomialMapRequest,
    FiniteFieldPolynomialMapResult,
)


def _require_irreducible_extension(request: FiniteFieldPolynomialMapRequest) -> None:
    from sympy import Poly, symbols

    variable = symbols("t")
    polynomial = Poly(
        sum(
            coefficient * variable**index
            for index, coefficient in enumerate(request.modulus_coefficients_ascending)
        ),
        variable,
        modulus=request.characteristic,
    )
    if not polynomial.is_irreducible:
        raise ValueError("extension modulus must be irreducible over the prime field")


class _ExtensionFieldEvaluator:
    def __init__(self, request: FiniteFieldPolynomialMapRequest) -> None:
        self.p = request.characteristic
        self.modulus = request.modulus_coefficients_ascending
        self.degree = len(self.modulus) - 1
        self.terms = request.terms

    def multiply(
        self, left: tuple[int, ...], right: tuple[int, ...]
    ) -> tuple[int, ...]:
        coefficients = [0] * (2 * self.degree - 1)
        for left_index, left_coefficient in enumerate(left):
            for right_index, right_coefficient in enumerate(right):
                index = left_index + right_index
                coefficients[index] = (
                    coefficients[index] + left_coefficient * right_coefficient
                ) % self.p
        for index in range(2 * self.degree - 2, self.degree - 1, -1):
            leading = coefficients[index]
            for modulus_index in range(self.degree):
                target = index - self.degree + modulus_index
                coefficients[target] = (
                    coefficients[target] - leading * self.modulus[modulus_index]
                ) % self.p
        return tuple(coefficients[: self.degree])

    def power(self, value: tuple[int, ...], exponent: int) -> tuple[int, ...]:
        result = (1,) + (0,) * (self.degree - 1)
        base = value
        while exponent:
            if exponent & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            exponent //= 2
        return result

    def evaluate(self, value: tuple[int, ...]) -> tuple[int, ...]:
        result = [0] * self.degree
        for term in self.terms:
            contribution = self.multiply(
                term.coefficient, self.power(value, term.exponent)
            )
            result = [
                (left + right) % self.p
                for left, right in zip(result, contribution, strict=True)
            ]
        return tuple(result)


def compute_finite_field_polynomial_map_fibers(
    request: FiniteFieldPolynomialMapRequest,
) -> FiniteFieldPolynomialMapResult:
    """Exhaustively evaluate one polynomial map on a bounded extension field."""
    _require_irreducible_extension(request)
    evaluator = _ExtensionFieldEvaluator(request)
    p, degree = evaluator.p, evaluator.degree
    field_order = p**degree
    counts: Counter[tuple[int, ...]] = Counter()
    first_inputs: dict[tuple[int, ...], tuple[int, ...]] = {}
    collision: FiniteFieldCollisionWitness | None = None
    digest = hashlib.sha256()
    byte_width = max(1, (p.bit_length() + 7) // 8)
    for encoded in range(field_order):
        value = tuple((encoded // (p**index)) % p for index in range(degree))
        output = evaluator.evaluate(value)
        digest.update(
            b"".join(coordinate.to_bytes(byte_width, "big") for coordinate in output)
        )
        if collision is None and output in first_inputs:
            collision = FiniteFieldCollisionWitness(
                left_input=first_inputs[output],
                right_input=value,
                common_output=output,
            )
        first_inputs.setdefault(output, value)
        counts[output] += 1
    histogram = Counter(counts.values())
    return FiniteFieldPolynomialMapResult(
        semantics_version="finite-field-polynomial-map.v1",
        characteristic=p,
        extension_degree=degree,
        modulus_coefficients_ascending=evaluator.modulus,
        terms=request.terms,
        enumeration_order="BASE_P_LEAST_SIGNIFICANT_COEFFICIENT_FIRST",
        total_inputs=field_order,
        distinct_outputs=len(counts),
        collision_excess=field_order - len(counts),
        fiber_histogram=tuple(
            FiniteFieldFiberCount(fiber_size=size, output_count=histogram[size])
            for size in sorted(histogram)
        ),
        is_permutation=len(counts) == field_order,
        first_collision=collision,
        output_sequence_sha256=digest.hexdigest(),
    )


__all__ = ["compute_finite_field_polynomial_map_fibers"]
