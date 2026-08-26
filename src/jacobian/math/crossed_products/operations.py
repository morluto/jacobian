"""Exact multiplication in finite-coset crossed products."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.crossed_products._budget import require_multiplication_budget
from jacobian.math.crossed_products.values import (
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductPresentation,
    FiniteCosetCrossedProductTerm,
    _integer_matrix_vector_product,
)


def _integer_actions(
    presentation: FiniteCosetCrossedProductPresentation,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(tuple(parse_canonical_integer(entry) for entry in row) for row in matrix)
        for matrix in presentation.action_matrices
    )


def _integer_cocycle(
    presentation: FiniteCosetCrossedProductPresentation,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(parse_canonical_integer(entry) for entry in vector) for vector in row
        )
        for row in presentation.cocycle_table
    )


def _multiply_admitted(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> FiniteCosetCrossedProductElement:
    presentation = left.presentation
    characteristic = presentation.characteristic
    coset_index = {
        label: position for position, label in enumerate(presentation.cosets)
    }
    actions = _integer_actions(presentation)
    cocycle = _integer_cocycle(presentation)
    coefficients: dict[tuple[int, tuple[int, ...]], int] = {}

    for left_term in left.terms:
        left_coset = coset_index[left_term.coset]
        left_exponents = tuple(
            parse_canonical_integer(entry) for entry in left_term.exponents
        )
        for right_term in right.terms:
            right_coset = coset_index[right_term.coset]
            right_exponents = tuple(
                parse_canonical_integer(entry) for entry in right_term.exponents
            )
            product_coset_label = presentation.quotient_multiplication[left_coset][
                right_coset
            ]
            product_coset = coset_index[product_coset_label]
            transformed = _integer_matrix_vector_product(
                actions[left_coset], right_exponents
            )
            product_exponents = tuple(
                left_coordinate + transformed_coordinate + cocycle_coordinate
                for left_coordinate, transformed_coordinate, cocycle_coordinate in zip(
                    left_exponents,
                    transformed,
                    cocycle[left_coset][right_coset],
                    strict=True,
                )
            )
            key = (product_coset, product_exponents)
            coefficient = (
                coefficients.get(key, 0)
                + left_term.coefficient * right_term.coefficient
            ) % characteristic
            if coefficient:
                coefficients[key] = coefficient
            else:
                coefficients.pop(key, None)

    return FiniteCosetCrossedProductElement(
        presentation=presentation,
        terms=tuple(
            FiniteCosetCrossedProductTerm(
                coefficient=coefficient,
                coset=presentation.cosets[coset],
                exponents=tuple(format_canonical_integer(entry) for entry in exponents),
            )
            for (coset, exponents), coefficient in sorted(coefficients.items())
        ),
    )


def multiply(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
) -> FiniteCosetCrossedProductElement:
    """Multiply two finite-support elements in one explicit crossed product."""

    require_multiplication_budget(left, right)
    return _multiply_admitted(left, right)


__all__ = ["multiply"]
