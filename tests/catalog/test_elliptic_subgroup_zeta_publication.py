"""Publication checks for the elliptic-curve subgroup and zeta operations.

These open the catalog, so they live in the catalog lane rather than under
``tests/math``.
"""

import json
from fractions import Fraction

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldShortWeierstrassCurve,
    finite_field_zeta_function,
)


def test_generated_subgroup_membership_is_publicly_discoverable() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(
        "elliptic_curve.finite_field.point.membership_in_generated_subgroup.decide"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["belongs"] is True
    assert len(result.output["generators"]) == 1


def test_full_zeta_function_is_a_native_projection_of_the_published_numerator() -> None:
    """The zeta rational function is a projection, not a second operation.

    `elliptic_curve.finite_field.zeta.compute` was published alongside
    `elliptic_curve.finite_field.zeta_polynomial.compute`, which already
    performs the same point count and returns the same curve, cardinality, and
    trace. The new operation only attached the universal denominator
    `(1-T)(1-q*T)` and repackaged the result, so the two shared one
    mathematical postcondition under two discovery IDs.

    The published operation keeps the postcondition; the rational-function form
    stays reachable natively.
    """
    catalog = Catalog.open()
    assert catalog.operation("elliptic_curve.finite_field.zeta.compute") is None

    published = catalog.operation("elliptic_curve.finite_field.zeta_polynomial.compute")
    assert published is not None
    result = invoke_operation(
        published.operation_id, published.examples[0].input, catalog
    )
    assert result.output["cardinality"] == 9
    assert result.output["trace"] == -3

    published_curve = FiniteFieldShortWeierstrassCurve.model_validate_json(
        json.dumps(result.output["curve"])
    )
    native = finite_field_zeta_function(published_curve)
    assert native.cardinality == result.output["cardinality"]
    assert native.trace == result.output["trace"]
    assert native.zeta_function.variables == ("T",)
    # (1 - a*T + q*T^2)/((1-T)(1-q*T)) with a = -3 and q = 5, normalized
    # over QQ(T) so the denominator's leading coefficient is 1
    numerator = {
        term.exponents: term.coefficient.as_fraction()
        for term in native.zeta_function.numerator.terms
    }
    denominator = {
        term.exponents: term.coefficient.as_fraction()
        for term in native.zeta_function.denominator.terms
    }
    assert numerator == {
        (2,): Fraction(1),
        (1,): Fraction(3, 5),
        (0,): Fraction(1, 5),
    }
    assert denominator == {
        (2,): Fraction(1),
        (1,): Fraction(-6, 5),
        (0,): Fraction(1, 5),
    }
