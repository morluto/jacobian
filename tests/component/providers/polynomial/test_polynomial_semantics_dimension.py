"""Polynomial-map semantics must reflect the public value contract."""

from tests.component.providers.polynomial.polynomial_operations_support import (
    PolynomialTestServices,
)

from jacobian.contracts.polynomials import MAX_POLYNOMIAL_VARIABLES


def test_map_semantics_uses_the_public_dimension_owner(
    polynomial_services: PolynomialTestServices,
) -> None:
    descriptor = polynomial_services.core.store.get_descriptor(
        polynomial_services.polynomial.semantics_uri,
        expected_kind="semantics",
    )

    assert descriptor["definition"]["maximum_dimension"] == MAX_POLYNOMIAL_VARIABLES
