"""Catalog invocation for integer content and primitive-part profiles."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials import integer_polynomial_primitive_part
from jacobian.math.polynomials._models import IntegerPolynomial


def test_catalog_integer_primitive_profile_matches_native() -> None:
    polynomial = IntegerPolynomial(coefficients=(6, 0, -6))
    native = integer_polynomial_primitive_part(polynomial)
    dispatched = invoke_operation(
        "polynomial.integer.content_primitive_profile.compute",
        {"polynomial": {"coefficients": ["6", "0", "-6"]}},
        Catalog.open(),
    )

    assert dispatched.output == native.model_dump(mode="json")
    assert dispatched.output["convention"] == "NONNEGATIVE_CONTENT_POSITIVE_LEADING"
