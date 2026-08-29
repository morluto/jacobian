from __future__ import annotations

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import PolynomialGcdRequest
from jacobian.math.polynomials.operations import (
    polynomial_discriminant,
    polynomial_gcd,
    polynomial_resultant,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    SparseRationalPolynomial,
)


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
    result = polynomial_gcd(request.left, request.right)

    direct_consumer = PolynomialGcdRequest(left=result.gcd, right=source)
    assert direct_consumer.left is result.gcd

    serialized_consumer = PolynomialGcdRequest.model_validate(
        {
            "left": result.gcd.model_dump(mode="json"),
            "right": source.model_dump(mode="json"),
        }
    )
    assert (
        polynomial_gcd(serialized_consumer.left, serialized_consumer.right).gcd
        == result.gcd
    )


def test_invariant_operations_accept_canonical_polynomial_values() -> None:
    source = _univariate_polynomial()

    discriminant = polynomial_discriminant(source, "x")
    resultant = polynomial_resultant(source, source, "x")

    assert discriminant.variable == "x"
    assert discriminant.discriminant.kind == "SCALAR"
    assert discriminant.discriminant.value.num == "4"
    assert resultant.elimination_variable == "x"
    assert resultant.resultant.kind == "SCALAR"
    assert resultant.resultant.value.num == "0"


def test_sparse_polynomial_schema_explains_canonical_term_order() -> None:
    terms = SparseRationalPolynomial.model_json_schema()["properties"]["terms"]

    assert terms["description"] == (
        "Nonzero monomials in descending lexicographic order of their exponent "
        "tuples (highest first). For one variable, list [2] before [0]."
    )
    assert terms["examples"] == [
        [
            {
                "coefficient": {"num": "1", "den": "1"},
                "exponents": [2],
            },
            {
                "coefficient": {"num": "-1", "den": "1"},
                "exponents": [0],
            },
        ]
    ]
