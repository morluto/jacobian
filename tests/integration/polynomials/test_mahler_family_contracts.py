"""Native/catalog parity for the exact Mahler measure family (#1787)."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials import (
    integer_polynomial_primitive_part,
    mahler_measure,
    quadratic_root_profile,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomial,
)


def test_catalog_mahler_family_matches_domain_valued_natives() -> None:
    polynomial = IntegerPolynomial(coefficients=(1, -1, -1))
    catalog = Catalog.open()
    native_measure = mahler_measure(polynomial)
    dispatched_measure = invoke_operation(
        "polynomial.mahler_measure.compute",
        {"polynomial": {"coefficients": ["1", "-1", "-1"]}},
        catalog,
    )
    assert dispatched_measure.output == native_measure.model_dump(mode="json")
    native_roots = quadratic_root_profile(polynomial)
    dispatched_roots = invoke_operation(
        "polynomial.quadratic.real_root_profile.compute",
        {"polynomial": {"coefficients": ["1", "-1", "-1"]}},
        catalog,
    )
    assert dispatched_roots.output == native_roots.model_dump(mode="json")
    native_content = integer_polynomial_primitive_part(
        IntegerPolynomial(coefficients=(6, 0, -6))
    )
    dispatched_content = invoke_operation(
        "polynomial.integer.content_primitive_profile.compute",
        {"polynomial": {"coefficients": ["6", "0", "-6"]}},
        catalog,
    )
    assert dispatched_content.output == native_content.model_dump(mode="json")
    assert dispatched_content.output["convention"] == (
        "NONNEGATIVE_CONTENT_POSITIVE_LEADING"
    )
