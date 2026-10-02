"""A public GCD result admits equivalent native Bézout witnesses after decoding."""

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._models import PolynomialGcdResult
from jacobian.math.polynomials.operations import verify_polynomial_gcd


def test_public_gcd_with_alternative_witness_round_trips_to_native_consumer() -> None:
    def polynomial(coefficients: tuple[int, ...]) -> dict[str, object]:
        return {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "exponents": [degree],
                        "coefficient": {"num": str(value), "den": "1"},
                    }
                    for degree, value in reversed(tuple(enumerate(coefficients)))
                    if value
                ]
            },
        }

    invocation = invoke_operation(
        "polynomial.compute.gcd",
        {"left": polynomial((-1, 0, 1)), "right": polynomial((-1, 1))},
        Catalog.open(),
    )
    body = invocation.output
    original = PolynomialGcdResult.model_validate_json(encode_strict_json(body))
    assert verify_polynomial_gcd(original)
    body["bezout"] = {
        "left_multiplier": polynomial((1,)),
        "right_multiplier": polynomial((0, -1)),
    }
    alternative = PolynomialGcdResult.model_validate_json(encode_strict_json(body))
    assert alternative != original
    assert verify_polynomial_gcd(alternative)
