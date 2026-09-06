import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.dynamics.arithmetic._models import (
    DynatomicPolynomialRequest,
    MapIterateRequest,
)
from jacobian.math.dynamics.arithmetic._tools import (
    compute_dynatomic_polynomial,
    compute_map_iterate,
)


def _integer(value: str) -> CanonicalRational:
    return CanonicalRational(num=int(value), den=1)


def _p(*values: CanonicalRational):
    from jacobian.math.dynamics.arithmetic import polynomial_from_coefficients

    return polynomial_from_coefficients(tuple(value.as_fraction() for value in values))


def test_monomial_counterexample_is_rejected_by_native_admission() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_map_iterate(
            MapIterateRequest(
                polynomial=_p(_integer("0"), _integer("0"), _integer(coefficient)), n=10
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )


def test_near_boundary_monomial_remains_admitted() -> None:
    coefficient = "1" + "0" * 30
    request = MapIterateRequest(
        polynomial=_p(_integer("0"), _integer("0"), _integer(coefficient)), n=10
    )

    assert request.n == 10


def test_dense_polynomial_additive_growth_is_propagated() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_map_iterate(
            MapIterateRequest(
                polynomial=_p(
                    _integer(coefficient), _integer(coefficient), _integer(coefficient)
                ),
                n=9,
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )


def test_dynatomic_request_checks_each_required_iterate() -> None:
    coefficient = "1" + "0" * 127
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_dynatomic_polynomial(
            DynatomicPolynomialRequest(
                polynomial=_p(_integer("0"), _integer("0"), _integer(coefficient)), n=9
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "arithmetic_dynamics.iterate_coefficient_growth_exceeds_bound"
    )
