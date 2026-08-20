import pytest
from pydantic import ValidationError

from jacobian.math.arithmetic_dynamics._models import (
    DynatomicPolynomialRequest,
    MapIterateRequest,
)


def test_monomial_counterexample_is_rejected_by_request_validation() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError, match="iterate coefficient growth"):
        MapIterateRequest(coefficients=("0", "0", coefficient), n=10)


def test_near_boundary_monomial_remains_admitted() -> None:
    coefficient = "1" + "0" * 30
    request = MapIterateRequest(coefficients=("0", "0", coefficient), n=10)

    assert request.n == 10


def test_dense_polynomial_additive_growth_is_propagated() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError, match="iterate coefficient growth"):
        MapIterateRequest(coefficients=(coefficient, coefficient, coefficient), n=9)


def test_dynatomic_request_checks_each_required_iterate() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(ValidationError, match="iterate coefficient growth"):
        DynatomicPolynomialRequest(coefficients=("0", "0", coefficient), n=9)
