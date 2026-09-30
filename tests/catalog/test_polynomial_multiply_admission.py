"""Public rational multiplication checks for collected numerator growth."""

from fractions import Fraction

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import decimal_digit_width, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.values import RationalPolynomial


@pytest.mark.parametrize("oversized", (False, True))
def test_public_multiply_bounds_common_denominator_numerator_growth(
    oversized: bool,
) -> None:
    exponent = 20_000 if oversized else MAX_CANONICAL_RATIONAL_DIGITS - 14_000 - 10
    content = 10**exponent
    a = 10**7000

    def operand(numerator: int, offsets: tuple[int, int]) -> dict[str, object]:
        return {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "exponents": [1 - index],
                        "coefficient": CanonicalRational.from_fraction(
                            Fraction(numerator, a + offset)
                        ).model_dump(mode="json"),
                    }
                    for index, offset in enumerate(offsets)
                ]
            },
        }

    payload = {"left": operand(content, (1, 3)), "right": operand(1, (7, 9))}
    catalog = Catalog.open()
    if oversized:
        with pytest.raises(OperationDomainValidationError) as error:
            invoke_operation("polynomial.rational.multiply.compute", payload, catalog)
        assert error.value.errors()[0]["type"] == "polynomial.invariant"
    else:
        result = invoke_operation(
            "polynomial.rational.multiply.compute", payload, catalog
        )
        decoded = RationalPolynomial.model_validate_json(
            encode_strict_json(result.output)
        )
        expected = Fraction(content, (a + 1) * (a + 9)) + Fraction(
            content, (a + 3) * (a + 7)
        )
        assert decoded.variables == ("x",)
        assert decoded.polynomial.terms[1].exponents == (1,)
        assert decoded.polynomial.terms[1].coefficient.as_fraction() == expected
        assert (
            decimal_digit_width(expected.numerator) == MAX_CANONICAL_RATIONAL_DIGITS - 9
        )


@pytest.mark.parametrize(
    "digits",
    (
        MAX_CANONICAL_RATIONAL_DIGITS - 1,
        MAX_CANONICAL_RATIONAL_DIGITS,
        MAX_CANONICAL_RATIONAL_DIGITS + 1,
    ),
)
@pytest.mark.parametrize("reciprocal", (False, True))
def test_public_single_nonunit_product_keeps_the_exact_boundary(
    digits: int, reciprocal: bool
) -> None:
    def constant(numerator: int, denominator: int) -> dict[str, object]:
        return {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": CanonicalRational(
                            num=numerator, den=denominator
                        ).model_dump(mode="json"),
                        "exponents": [0],
                    }
                ]
            },
        }

    large = 5 * 10 ** (digits - 2)
    payload = {
        "left": constant(1, large) if reciprocal else constant(large, 1),
        "right": constant(1, 2) if reciprocal else constant(2, 1),
    }
    catalog = Catalog.open()
    if digits > MAX_CANONICAL_RATIONAL_DIGITS:
        with pytest.raises(OperationDomainValidationError):
            invoke_operation("polynomial.rational.multiply.compute", payload, catalog)
    else:
        result = invoke_operation(
            "polynomial.rational.multiply.compute", payload, catalog
        )
        decoded = RationalPolynomial.model_validate_json(
            encode_strict_json(result.output)
        )
        expected = (
            Fraction(1, 10 ** (digits - 1))
            if reciprocal
            else Fraction(10 ** (digits - 1))
        )
        assert decoded.polynomial.terms[0].coefficient.as_fraction() == expected
