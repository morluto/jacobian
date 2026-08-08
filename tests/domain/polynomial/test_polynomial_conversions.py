from __future__ import annotations

from jacobian.contracts.polynomial_operations import PolynomialGcdRequest
from jacobian.contracts.polynomials import RationalPolynomial
from jacobian.domains.polynomial.conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.domains.polynomial.operations import polynomial_gcd


def _polynomial() -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["y", "x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "3", "den": "2"},
                        "exponents": [2, 0],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "3"},
                        "exponents": [0, 1],
                    },
                ]
            },
        }
    )


def _univariate_polynomial() -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [2],
                    },
                    {
                        "coefficient": {"num": "-1", "den": "1"},
                        "exponents": [0],
                    },
                ]
            },
        }
    )


def test_contract_sympy_contract_round_trip_preserves_ring_and_order() -> None:
    source = _polynomial()

    assert (
        rational_polynomial_from_sympy(
            rational_polynomial_to_sympy(source), source.variables
        )
        == source
    )


def test_gcd_result_composes_without_reshaping_or_json_round_trip() -> None:
    source = _univariate_polynomial()
    request = PolynomialGcdRequest(left=source, right=source)
    result = polynomial_gcd(request)

    direct_consumer = PolynomialGcdRequest(left=result.gcd, right=source)
    assert direct_consumer.left is result.gcd

    serialized_consumer = PolynomialGcdRequest.model_validate(
        {
            "left": result.gcd.model_dump(mode="json"),
            "right": source.model_dump(mode="json"),
        }
    )
    assert polynomial_gcd(serialized_consumer).gcd == result.gcd
