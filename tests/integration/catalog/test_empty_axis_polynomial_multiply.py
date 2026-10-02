"""Constant normalization composes with multiplication in the zero-axis QQ ring."""

from fractions import Fraction

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.operations import multiply
from jacobian.math.polynomials.values import RationalPolynomial


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (Fraction(3), Fraction(3)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(3)),
        (Fraction(3), Fraction(0)),
        (Fraction(3), Fraction(1)),
        (Fraction(1), Fraction(3)),
        (Fraction(-2, 3), Fraction(9, 4)),
    ],
)
def test_normalized_constants_multiply_with_empty_axes(
    left: Fraction, right: Fraction
) -> None:
    catalog = Catalog.open()

    def normalize(value: Fraction) -> dict[str, object]:
        result = invoke_operation(
            "polynomial.expression.normalize",
            {
                "coefficient_domain": "QQ",
                "variables": [],
                "expression": {
                    "kind": "LITERAL",
                    "value": {
                        "num": str(value.numerator),
                        "den": str(value.denominator),
                    },
                },
            },
            catalog,
        )
        return dict(result.output["polynomial"])

    left_wire, right_wire = normalize(left), normalize(right)
    result = invoke_operation(
        "polynomial.rational.multiply.compute",
        {"left": left_wire, "right": right_wire},
        catalog,
    )
    decoded = RationalPolynomial.model_validate_json(encode_strict_json(result.output))
    native = multiply(
        RationalPolynomial.model_validate_json(encode_strict_json(left_wire)),
        RationalPolynomial.model_validate_json(encode_strict_json(right_wire)),
    )
    assert decoded == native
    assert decoded.variables == ()
    expected = left * right
    if expected:
        assert len(decoded.polynomial.terms) == 1
        term = decoded.polynomial.terms[0]
        assert term.exponents == ()
        assert term.coefficient.as_fraction() == expected
    else:
        assert decoded.polynomial.terms == ()
