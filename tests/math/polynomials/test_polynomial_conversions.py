from __future__ import annotations

from copy import deepcopy

import pytest
import sympy

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials._models import PolynomialGcdRequest
from jacobian.math.polynomials.operations import (
    polynomial_discriminant,
    polynomial_factorization,
    polynomial_gcd,
    polynomial_resultant,
    polynomial_square_free_decomposition,
    verify_polynomial_discriminant,
    verify_polynomial_factorization,
    verify_polynomial_gcd,
    verify_polynomial_resultant,
    verify_polynomial_square_free_decomposition,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    SparseRationalPolynomial,
    require_canonical_rational_function,
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


def _binomial(degree: int, constant: int) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [degree],
                    },
                    {
                        "coefficient": {"num": str(constant), "den": "1"},
                        "exponents": [0],
                    },
                ]
            },
        }
    )


def _nonreduced_rational_function_payload() -> dict[str, object]:
    return {
        "variables": ["x"],
        "numerator": {
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
        "denominator": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [1],
                },
                {
                    "coefficient": {"num": "-1", "den": "1"},
                    "exponents": [0],
                },
            ]
        },
    }


def test_contract_sympy_contract_round_trip_preserves_ring_and_order() -> None:
    source = _polynomial()

    assert (
        rational_polynomial_from_sympy(
            rational_polynomial_to_sympy(source), source.variables
        )
        == source
    )


def test_rational_function_wire_parse_is_structural_before_owner_recognition() -> None:
    parsed = RationalFunction.model_validate(_nonreduced_rational_function_payload())

    assert parsed.variables == ("x",)
    with pytest.raises(ValueError, match="must be coprime"):
        require_canonical_rational_function(parsed)


def test_normalized_rational_function_round_trip_is_recognized_by_a_consumer() -> None:
    x = sympy.Symbol("x")
    produced = rational_function_from_sympy((x**2 - 1) / (x - 1), ("x",))
    consumed = RationalFunction.model_validate(produced.model_dump(mode="json"))

    assert require_canonical_rational_function(consumed) == produced
    assert rational_function_to_sympy(consumed) == x + 1


@pytest.mark.parametrize(
    ("expression", "variables"),
    (
        (sympy.Integer(0), ()),
        (sympy.Rational(2, 3), ()),
        (
            sympy.Symbol("x") / (sympy.Symbol("y") + 1),
            ("x", "y"),
        ),
    ),
)
def test_recognizer_accepts_canonical_zero_constant_and_multivariate_values(
    expression: object,
    variables: tuple[str, ...],
) -> None:
    value = rational_function_from_sympy(expression, variables)

    assert require_canonical_rational_function(value) is value


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


@pytest.mark.parametrize(
    ("operation", "verifier", "mutation"),
    (
        (
            lambda source: polynomial_gcd(source, source),
            verify_polynomial_gcd,
            lambda payload: payload["gcd"]["polynomial"]["terms"][0][
                "coefficient"
            ].update(num="2"),
        ),
        (
            lambda source: polynomial_resultant(source, source, "x"),
            verify_polynomial_resultant,
            lambda payload: payload["resultant"]["value"].update(num="1"),
        ),
        (
            lambda source: polynomial_discriminant(source, "x"),
            verify_polynomial_discriminant,
            lambda payload: payload["discriminant"]["value"].update(num="5"),
        ),
        (
            polynomial_square_free_decomposition,
            verify_polynomial_square_free_decomposition,
            lambda payload: payload["reconstructed"]["polynomial"]["terms"][0][
                "coefficient"
            ].update(num="2"),
        ),
        (
            polynomial_factorization,
            verify_polynomial_factorization,
            lambda payload: payload["reconstructed"]["polynomial"]["terms"][0][
                "coefficient"
            ].update(num="2"),
        ),
    ),
)
def test_invariant_claim_verifiers_reject_forged_serialized_results(
    operation: object,
    verifier: object,
    mutation: object,
) -> None:
    source = _univariate_polynomial()
    result = operation(source)  # type: ignore[operator]
    payload = deepcopy(result.model_dump(mode="json"))
    mutation(payload)  # type: ignore[operator]

    forged = type(result).model_validate(payload)

    assert not verifier(forged)  # type: ignore[operator]


def test_zero_polynomial_has_zero_discriminant() -> None:
    zero = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(),
    )

    result = polynomial_discriminant(zero, "x")

    assert result.discriminant.kind == "SCALAR"
    assert result.discriminant.value.as_fraction() == 0


def test_flint_discriminant_accepts_degree_above_previous_ceiling() -> None:
    degree = 512
    result = polynomial_discriminant(_binomial(degree, -2), "x")

    assert result.discriminant.kind == "SCALAR"
    expected = (
        (-1) ** (degree * (degree - 1) // 2) * degree**degree * (-2) ** (degree - 1)
    )
    assert result.discriminant.value.as_fraction() == expected


def test_flint_resultant_accepts_degree_sum_above_previous_ceiling() -> None:
    degree = 96
    result = polynomial_resultant(_binomial(degree, -2), _binomial(degree, -3), "x")

    assert result.resultant.kind == "SCALAR"
    assert result.resultant.value.as_fraction() == 1


def test_univariate_discriminant_rejects_degree_above_work_bound() -> None:
    with pytest.raises(OperationDomainValidationError, match="operation budget"):
        polynomial_discriminant(_binomial(1_025, -2), "x")


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
