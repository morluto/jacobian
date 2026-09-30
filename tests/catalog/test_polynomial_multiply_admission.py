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
