"""Exact quadratic composita checked by independent quotient arithmetic."""

from fractions import Fraction
from math import isqrt

from jacobian.math.number_theory.galois._compositum import (
    GaloisCompositumResult,
    galois_compositum,
)
from jacobian.math.number_theory.galois.operations import splitting_field
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": [power],
                    }
                    for power, coefficient in reversed(tuple(enumerate(coefficients)))
                    if coefficient
                ]
            },
        }
    )


def _field(radicand: int):
    return splitting_field(_polynomial((-radicand, 0, 1))).field


def _multiply(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    defining_descending: tuple[int, ...],
) -> tuple[Fraction, ...]:
    degree = len(defining_descending) - 1
    product = [Fraction(0)] * (2 * degree - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            product[i + j] += left_value * right_value
    for power in range(len(product) - 1, degree - 1, -1):
        coefficient = product[power]
        for basis_power in range(degree):
            product[power - degree + basis_power] -= coefficient * Fraction(
                defining_descending[degree - basis_power], defining_descending[0]
            )
    return tuple(product[:degree])


def _assert_source_relation(result: GaloisCompositumResult, *, side: str) -> None:
    source = getattr(result, side).extension
    embedding = getattr(result, f"{side}_embedding")
    image = tuple(
        value.as_fraction()
        for value in embedding.generator_image.coefficients_ascending
    )
    if source.degree == 1:
        a, b = (Fraction(value) for value in source.coefficients_descending)
        residual = tuple(
            a * image[i] + (b if i == 0 else 0) for i in range(result.compositum.degree)
        )
        assert residual == (Fraction(0),) * result.compositum.degree
        return
    a, b, c = (Fraction(value) for value in source.coefficients_descending)
    square = _multiply(image, image, result.compositum.coefficients_descending)
    residual = tuple(
        a * square[i] + b * image[i] + (c if i == 0 else 0)
        for i in range(result.compositum.degree)
    )
    assert residual == (Fraction(0),) * result.compositum.degree


def test_independent_quadratic_fields_have_exact_degree_four_compositum() -> None:
    result = galois_compositum(_field(2), _field(3))

    # The discriminants are 8 and 12; their ratio 3/2 is not a rational square,
    # so these are distinct quadratic classes and their compositum has degree 4.
    assert isqrt(3) ** 2 != 3
    # For theta=sqrt(8)+sqrt(12), the exact minimal polynomial is
    # theta^4 - 40 theta^2 + 16. Distinct quadratic classes give degree four.
    assert result.compositum.coefficients_descending == (1, 0, -40, 0, 16)
    assert result.left_embedding.source == result.left.extension
    assert result.right_embedding.source == result.right.extension
    assert result.left_embedding.target == result.compositum
    assert result.right_embedding.target == result.compositum
    _assert_source_relation(result, side="left")
    _assert_source_relation(result, side="right")

    decoded = GaloisCompositumResult.model_validate_json(result.model_dump_json())
    assert decoded == result


def test_same_quadratic_field_class_reuses_field_with_exact_inclusion() -> None:
    result = galois_compositum(_field(2), _field(8))

    assert result.compositum == result.left.extension
    assert result.compositum.degree == 2
    assert tuple(
        value.as_fraction()
        for value in result.right_embedding.generator_image.coefficients_ascending
    ) == (Fraction(0), Fraction(2))
    _assert_source_relation(result, side="left")
    _assert_source_relation(result, side="right")


def test_rational_splitting_field_is_the_identity_input_to_compositum() -> None:
    rational = splitting_field(_polynomial((-1, 0, 1))).field
    quadratic = _field(2)
    result = galois_compositum(rational, quadratic)

    assert result.compositum == quadratic.extension
    assert tuple(
        value.as_fraction()
        for value in result.left_embedding.generator_image.coefficients_ascending
    ) == (Fraction(0), Fraction(0))
    _assert_source_relation(result, side="left")
    _assert_source_relation(result, side="right")
