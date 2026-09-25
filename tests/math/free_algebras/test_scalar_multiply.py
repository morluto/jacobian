from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_RESULT_TERMS,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
)
from jacobian.math.free_algebras.scalar_multiply._models import (
    FreeAlgebraPolynomialScalarMultiplyRequest,
)
from jacobian.math.free_algebras.scalar_multiply.operations import scalar_multiply


def polynomial(
    alphabet: tuple[str, ...], values: tuple[tuple[tuple[str, ...], Fraction], ...]
) -> FreeAlgebraPolynomial:
    return FreeAlgebraPolynomial(
        alphabet=alphabet,
        terms=tuple(
            FreeAlgebraTerm(
                coefficient=CanonicalRational.from_fraction(coefficient), word=word
            )
            for word, coefficient in values
        ),
    )


def test_rational_scaling_matches_independent_sparse_coefficient_oracle() -> None:
    source = polynomial(
        ("x", "y"),
        (
            (("y", "x"), Fraction(-7, 6)),
            (("x", "y"), Fraction(3, 5)),
            (("x",), Fraction(2, 3)),
        ),
    )
    scalar = CanonicalRational.from_fraction(Fraction(-9, 10))

    result = scalar_multiply(source, scalar)
    expected = {
        term.word: term.coefficient.as_fraction() * Fraction(-9, 10)
        for term in source.terms
    }
    actual = {term.word: term.coefficient.as_fraction() for term in result.terms}

    assert result.alphabet == source.alphabet
    assert actual == expected
    assert tuple(term.word for term in result.terms) == tuple(
        term.word for term in source.terms
    )

    matrices = {
        "x": ((Fraction(1), Fraction(1)), (Fraction(0), Fraction(1))),
        "y": ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0))),
    }

    def matrix_product(left, right):
        return tuple(
            tuple(
                sum(left[row][index] * right[index][column] for index in range(2))
                for column in range(2)
            )
            for row in range(2)
        )

    def evaluate(value):
        total = ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0)))
        identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
        for term in value.terms:
            word_value = identity
            for letter in term.word:
                word_value = matrix_product(word_value, matrices[letter])
            total = tuple(
                tuple(
                    total[row][column]
                    + term.coefficient.as_fraction() * word_value[row][column]
                    for column in range(2)
                )
                for row in range(2)
            )
        return total

    assert evaluate(result) == tuple(
        tuple(Fraction(-9, 10) * entry for entry in row) for row in evaluate(source)
    )


def test_zero_degenerate_values_and_full_word_boundary() -> None:
    alphabet = ("x",)
    zero = polynomial(alphabet, ())
    huge_scalar = CanonicalRational.from_fraction(Fraction(10**1000))
    assert scalar_multiply(zero, huge_scalar) == zero

    source = polynomial(alphabet, ((("x",) * 64, Fraction(1)),))
    zero_result = scalar_multiply(source, CanonicalRational.from_fraction(Fraction(0)))
    assert zero_result == polynomial(alphabet, ())
    assert (
        scalar_multiply(source, CanonicalRational.from_fraction(Fraction(1))) == source
    )


def test_large_scalar_with_exact_cross_cancellation_is_admitted() -> None:
    cancelland = 10**64 - 1
    source = polynomial(("x",), ((("x",), Fraction(1, cancelland)),))
    scalar = CanonicalRational.from_fraction(Fraction(cancelland**2))

    result = scalar_multiply(source, scalar)

    assert [term.coefficient.as_fraction() for term in result.terms] == [
        Fraction(cancelland)
    ]


def test_full_polynomial_support_boundary_is_preserved() -> None:
    alphabet = ("a", "b", "c", "d")
    words = tuple(reversed(tuple(product(alphabet, repeat=6))))
    assert len(words) == MAX_FREE_ALGEBRA_RESULT_TERMS
    source = polynomial(alphabet, tuple((word, Fraction(1)) for word in words))

    result = scalar_multiply(source, CanonicalRational.from_fraction(Fraction(-2)))

    assert len(result.terms) == MAX_FREE_ALGEBRA_RESULT_TERMS
    assert tuple(term.word for term in result.terms) == words
    assert all(term.coefficient.as_fraction() == -2 for term in result.terms)


def test_coefficient_growth_is_admitted_then_exactly_rejected() -> None:
    source = polynomial(("x",), ((("x",), Fraction(1)),))
    at_bound = scalar_multiply(
        source, CanonicalRational.from_fraction(Fraction(10**63))
    )
    assert at_bound.terms[0].coefficient.as_fraction() == 10**63

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        scalar_multiply(source, CanonicalRational.from_fraction(Fraction(10**64)))
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.scalar_multiply_coefficient_growth"
    )


def test_scalar_above_cross_cancellation_bound_is_rejected() -> None:
    source = polynomial(("x",), ((("x",), Fraction(1)),))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        scalar_multiply(source, CanonicalRational.from_fraction(Fraction(10**128)))
    assert exc_info.value.errors()[0]["type"] == (
        "free_algebra.scalar_multiply_coefficient_growth"
    )


def test_request_json_round_trip_preserves_the_operation_arguments() -> None:
    request = FreeAlgebraPolynomialScalarMultiplyRequest(
        polynomial=polynomial(("x", "y"), ((("x", "y"), Fraction(3, 2)),)),
        scalar=CanonicalRational.from_fraction(Fraction(-4, 3)),
    )
    decoded = FreeAlgebraPolynomialScalarMultiplyRequest.model_validate_json(
        request.model_dump_json()
    )

    assert (
        scalar_multiply(decoded.polynomial, decoded.scalar)
        .terms[0]
        .coefficient.as_fraction()
        == -2
    )


def test_invalid_scalar_native_argument_uses_domain_error() -> None:
    source = polynomial(("x",), ((("x",), Fraction(1)),))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        scalar_multiply(source, object())  # type: ignore[arg-type]
    assert exc_info.value.errors()[0]["type"] == "free_algebra.scalar_shape"
