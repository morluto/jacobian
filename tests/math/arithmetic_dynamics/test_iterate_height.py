import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.arithmetic_dynamics._models import (
    DynatomicPolynomialRequest,
    MapIterateRequest,
)


def _integer(value: str) -> CanonicalRational:
    return CanonicalRational(num=value, den="1")


def test_monomial_counterexample_is_rejected_by_request_validation() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError) as exc_info:
        MapIterateRequest(
            coefficients=(_integer("0"), _integer("0"), _integer(coefficient)), n=10
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )


def test_near_boundary_monomial_remains_admitted() -> None:
    coefficient = "1" + "0" * 30
    request = MapIterateRequest(
        coefficients=(_integer("0"), _integer("0"), _integer(coefficient)), n=10
    )

    assert request.n == 10


def test_dense_polynomial_additive_growth_is_propagated() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError) as exc_info:
        MapIterateRequest(coefficients=(_integer(coefficient),) * 3, n=9)
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )


def test_dynatomic_request_checks_each_required_iterate() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError) as exc_info:
        DynatomicPolynomialRequest(
            coefficients=(_integer("0"), _integer("0"), _integer(coefficient)), n=9
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )
