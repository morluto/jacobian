import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._models import (
    PolynomialDiscriminantRequest,
)
from jacobian.math.number_theory.galois.operations import polynomial_discriminant
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": value, "den": 1},
                        "exponents": [power],
                    }
                    for power, value in reversed(tuple(enumerate(coefficients)))
                    if value
                ]
            },
        }
    )


@pytest.mark.parametrize(
    ("coefficients", "discriminant", "is_square"),
    [
        ((-1, 0, 1), 4, True),  # x^2 - 1, reducible
        ((-2, 0, 1), 8, False),  # x^2 - 2
        ((1, 0, 1), -4, False),  # x^2 + 1
        ((-2, 0, 0, 1), -108, False),  # x^3 - 2
        ((1, 0, 0, 0, 1), 256, True),  # x^4 + 1
    ],
)
def test_exact_discriminant_profile_matches_closed_forms(
    coefficients: tuple[int, ...], discriminant: int, is_square: bool
) -> None:
    result = polynomial_discriminant(_polynomial(coefficients))
    assert result.discriminant == discriminant
    assert result.is_rational_square is is_square
    assert PolynomialDiscriminantRequest(polynomial=result.polynomial).coefficients == (
        coefficients
    )


def test_discriminant_scales_by_scalar_to_power_two_n_minus_two() -> None:
    base = polynomial_discriminant(_polynomial((-1, 0, 1)))
    scaled = polynomial_discriminant(_polynomial((-2, 0, 2)))
    assert base.discriminant == 4
    assert scaled.discriminant == 16


def test_discriminant_rejects_out_of_domain_before_backend(monkeypatch) -> None:
    import sympy

    def forbidden(*args, **kwargs):
        raise AssertionError("backend was reached before domain admission")

    monkeypatch.setattr(sympy, "Poly", forbidden)
    with pytest.raises(OperationDomainValidationError) as error:
        polynomial_discriminant(_polynomial((0,) * 7 + (1,)))
    assert error.value.errors()[0]["type"] == "galois_theory.degree_bound"
