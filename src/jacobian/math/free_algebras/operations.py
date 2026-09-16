"""Native exact free associative algebra polynomial multiplication.

The single V1 operation establishes one atomic mathematical postcondition:
the exact distributive noncommutative product of two sparse ``QQ``-linear
polynomials over one shared ordered generator alphabet.  All semantic
admission is performed once, here, before the multiplication kernel runs; the
result value is constructed through a trusted factory rather than replaying
the product.
"""

from __future__ import annotations

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._kernel import multiply_sparse
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS,
    MAX_FREE_ALGEBRA_OPERAND_TERMS,
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    MAX_FREE_ALGEBRA_TERM_PAIRS,
    MAX_FREE_ALGEBRA_WORD_LENGTH,
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialProductResult,
)


def _reject_resource(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"free_algebra.{code}",
        message=message,
    )


def _admit_product(left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial) -> None:
    """Admit one product request before any word-pair expansion.

    The structural value contracts already established alphabet distinctness,
    canonical support, and per-coefficient digit bounds.  This shared
    admission helper adds the product operation's envelope: alphabet identity,
    operand term and word-length budgets, the term-pair product count, the
    result term count, and coefficient growth.
    """

    if left.alphabet != right.alphabet:
        raise OperationDomainValidationError(
            location=("right", "alphabet"),
            code="free_algebra.alphabet_mismatch",
            message=(
                "both operands must be bound to the same ordered generator alphabet"
            ),
        )

    max_operand_digits = 0
    for side, polynomial in (("left", left), ("right", right)):
        if len(polynomial.terms) > MAX_FREE_ALGEBRA_OPERAND_TERMS:
            _reject_resource(
                (side, "terms"),
                "operand_term_budget",
                f"{side} operand exceeds the "
                f"{MAX_FREE_ALGEBRA_OPERAND_TERMS}-term multiplication budget",
            )
        for index, term in enumerate(polynomial.terms):
            if len(term.word) > MAX_FREE_ALGEBRA_WORD_LENGTH:
                _reject_resource(
                    (side, "terms", index, "word"),
                    "operand_word_length_budget",
                    f"{side} operand word exceeds the "
                    f"{MAX_FREE_ALGEBRA_WORD_LENGTH}-letter multiplication "
                    "budget",
                )
            max_operand_digits = max(
                max_operand_digits,
                canonical_rational_component_digits(term.coefficient),
            )

    term_pair_count = len(left.terms) * len(right.terms)
    if term_pair_count > MAX_FREE_ALGEBRA_TERM_PAIRS:
        _reject_resource(
            ("left", "terms"),
            "term_pair_budget",
            "product term pairs exceed the "
            f"{MAX_FREE_ALGEBRA_TERM_PAIRS}-pair multiplication budget",
        )
    # Distinct product words are a subset of the term pairs, so this bounds
    # the result term count before product expansion.
    if term_pair_count > MAX_FREE_ALGEBRA_RESULT_TERMS:
        _reject_resource(
            ("left", "terms"),
            "result_term_budget",
            "product can exceed the "
            f"{MAX_FREE_ALGEBRA_RESULT_TERMS}-term result budget",
        )

    addition_digits = len(str(term_pair_count - 1)) if term_pair_count >= 2 else 0
    predicted_coefficient_digits = 2 * max_operand_digits + addition_digits
    if predicted_coefficient_digits > MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS:
        _reject_resource(
            ("left", "terms"),
            "coefficient_growth_budget",
            "predicted product coefficient growth exceeds the "
            f"{MAX_FREE_ALGEBRA_COEFFICIENT_DIGITS}-digit multiplication budget",
        )


def multiply(
    left: FreeAlgebraPolynomial, right: FreeAlgebraPolynomial
) -> FreeAlgebraPolynomialProductResult:
    """Multiply two sparse noncommutative polynomials exactly."""

    _admit_product(left, right)
    product, ledger = multiply_sparse(left, right)
    return FreeAlgebraPolynomialProductResult.model_construct(
        left=left,
        right=right,
        product=product,
        ledger=ledger,
    )


__all__ = ["multiply"]
