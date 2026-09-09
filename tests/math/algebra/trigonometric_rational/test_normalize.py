from jacobian.math.algebra.trigonometric_rational.operations import (
    TrigonometricRationalNormalizeRequest,
    normalize_trigonometric_rational,
)


def test_pythagorean_identity_normalizes_to_one() -> None:
    request = TrigonometricRationalNormalizeRequest.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "ADD",
                "children": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "SINE", "angle": {"coefficients": [1]}},
                        "exponent": 2,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "COSINE", "angle": {"coefficients": [1]}},
                        "exponent": 2,
                    },
                ],
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 1
    assert result.numerator.terms[0].exponents == (0,)
    assert (
        result.numerator.terms[0].coefficient == result.denominator.terms[0].coefficient
    )
    assert result.denominator == result.denominator_nonzero


def test_tangent_retains_cosine_nonzero_locus() -> None:
    request = TrigonometricRationalNormalizeRequest.model_validate(
        {
            "variables": ["x"],
            "expression": {
                "kind": "DIVIDE",
                "numerator": {"kind": "SINE", "angle": {"coefficients": [1]}},
                "denominator": {"kind": "COSINE", "angle": {"coefficients": [1]}},
            },
        }
    )
    result = normalize_trigonometric_rational(request)
    assert len(result.numerator.terms) == 2
    assert len(result.denominator.terms) == 2
    assert result.denominator == result.denominator_nonzero
