"""Exact multiplication in finite-coset crossed products."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.crossed_products._budget import require_multiplication_budget
from jacobian.math.crossed_products._models import CrossedProductMultiplyResult
from jacobian.math.crossed_products.values import (
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductTerm,
    _integer_matrix_vector_product,
)


def _multiply_admitted(
    left: FiniteCosetCrossedProductElement,
    right: FiniteCosetCrossedProductElement,
    actions: tuple[tuple[tuple[int, ...], ...], ...],
    cocycle: tuple[tuple[tuple[int, ...], ...], ...],
) -> FiniteCosetCrossedProductElement:
    presentation = left.presentation
    characteristic = presentation.characteristic
    coset_index = {
        label: position for position, label in enumerate(presentation.cosets)
    }
    coefficients: dict[tuple[int, tuple[int, ...]], int] = {}

    for left_term in left.terms:
        left_coset = coset_index[left_term.coset]
        left_exponents = tuple(entry for entry in left_term.exponents)
        for right_term in right.terms:
            right_coset = coset_index[right_term.coset]
            right_exponents = tuple(entry for entry in right_term.exponents)
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
                exponents=exponents,
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
    from sympy import isprime

    presentation = left.presentation
    actions = presentation.action_matrices
    cocycle = presentation.cocycle_table
    index = {label: position for position, label in enumerate(presentation.cosets)}
    try:
        if not isprime(presentation.characteristic):
            raise ValueError("characteristic must be prime")
        presentation._require_quotient_group(index)
        presentation._require_action_laws(index, actions)
        presentation._require_cocycle_laws(index, actions, cocycle)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="crossed_product.presentation_laws",
            message=str(exc),
        ) from exc
    return _multiply_admitted(left, right, actions, cocycle)


def verify_multiply(claim: CrossedProductMultiplyResult) -> bool:
    """Verify a serialized sparse product against both retained operands."""
    try:
        if (
            claim.left.presentation != claim.right.presentation
            or claim.product.presentation != claim.left.presentation
        ):
            return False
        return multiply(claim.left, claim.right) == claim.product
    except (AttributeError, TypeError, ValueError, OperationDomainValidationError):
        return False


__all__ = ["multiply", "verify_multiply"]
