"""Exact standard inclusions between rational cyclotomic fields."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Poly, cyclotomic_poly, symbols

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.cyclic_linear import (
    CyclotomicElementMapRequest,
    CyclotomicFieldInclusion,
    CyclotomicFieldInclusionCompositionRequest,
    CyclotomicFieldInclusionRequest,
    RationalCyclotomicElement,
    RationalCyclotomicField,
    apply_cyclotomic_field_inclusion,
    compose_cyclotomic_field_inclusions,
    cyclotomic_field_inclusion,
)


def _element(order: int, *coordinates: tuple[int, int]) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=RationalCyclotomicField(order=order),
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(numerator, denominator))
            for numerator, denominator in coordinates
        ),
    )


def test_standard_inclusion_has_reduced_generator_image_and_maps_exactly() -> None:
    inclusion = cyclotomic_field_inclusion(
        CyclotomicFieldInclusionRequest(
            source=RationalCyclotomicField(order=3),
            target=RationalCyclotomicField(order=6),
        )
    )
    assert tuple(value.as_fraction() for value in inclusion.generator_image) == (
        Fraction(-1),
        Fraction(1),
    )

    mapped = apply_cyclotomic_field_inclusion(
        CyclotomicElementMapRequest(
            inclusion=inclusion,
            element=_element(3, (1, 1), (2, 1)),
        )
    )
    assert tuple(value.as_fraction() for value in mapped.coefficients_ascending) == (
        Fraction(-1),
        Fraction(2),
    )


@pytest.mark.parametrize(
    "source_order,target_order", [(1, 1), (1, 8), (2, 4), (3, 12), (5, 10), (8, 16)]
)
def test_element_map_matches_independent_polynomial_substitution(
    source_order: int, target_order: int
) -> None:
    inclusion = cyclotomic_field_inclusion(
        CyclotomicFieldInclusionRequest(
            source=RationalCyclotomicField(order=source_order),
            target=RationalCyclotomicField(order=target_order),
        )
    )
    degree = inclusion.source.degree
    element = _element(
        source_order, *((index - 2, index + 1) for index in range(degree))
    )
    actual = apply_cyclotomic_field_inclusion(
        CyclotomicElementMapRequest(inclusion=inclusion, element=element)
    )

    x = symbols("x")
    modulus = Poly(cyclotomic_poly(target_order, x), x, domain="QQ")
    substituted = Poly(0, x, domain="QQ")
    for power, coefficient in enumerate(element.coefficients_ascending):
        term = Poly(x ** (power * (target_order // source_order)), x, domain="QQ")
        substituted += term * coefficient.as_fraction()
    expected = substituted.rem(modulus)
    assert tuple(
        value.as_fraction() for value in actual.coefficients_ascending
    ) == tuple(
        Fraction(int(expected.nth(i).p), int(expected.nth(i).q))
        for i in range(inclusion.target.degree)
    )


def test_standard_inclusions_compose_and_apply_after_json_round_trip() -> None:
    catalog = Catalog.open()
    direct = invoke_operation(
        "matrix.cyclic.cyclotomic_inclusion.compute",
        {"source": {"order": 3}, "target": {"order": 12}},
        catalog,
    ).output
    first = cyclotomic_field_inclusion(
        CyclotomicFieldInclusionRequest(
            source=RationalCyclotomicField(order=3),
            target=RationalCyclotomicField(order=6),
        )
    )
    second = cyclotomic_field_inclusion(
        CyclotomicFieldInclusionRequest(
            source=RationalCyclotomicField(order=6),
            target=RationalCyclotomicField(order=12),
        )
    )
    composed = compose_cyclotomic_field_inclusions(
        CyclotomicFieldInclusionCompositionRequest(first=first, second=second)
    )
    assert composed.model_dump(mode="json") == direct

    request = CyclotomicElementMapRequest(
        inclusion=composed,
        element=_element(3, (1, 1), (2, 1)),
    )
    decoded = CyclotomicElementMapRequest.model_validate_json(
        encode_strict_json(request.model_dump(mode="json")), strict=True
    )
    mapped = apply_cyclotomic_field_inclusion(decoded)
    assert tuple(value.as_fraction() for value in mapped.coefficients_ascending) == (
        Fraction(-1),
        Fraction(0),
        Fraction(2),
        Fraction(0),
    )


def test_catalog_examples_run_and_results_pass_strict_json_validation() -> None:
    catalog = Catalog.open()
    for operation_id in (
        "matrix.cyclic.cyclotomic_inclusion.compute",
        "matrix.cyclic.cyclotomic_inclusion.compose",
        "matrix.cyclic.cyclotomic_element.map",
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None and operation.examples
        for example in operation.examples:
            result = invoke_operation(operation_id, example.input, catalog)
            validated = operation.result_type.model_validate_json(
                encode_strict_json(result.output)
            )
            assert validated.model_dump(mode="json") == result.output


def test_inclusion_rejects_nondividing_parent() -> None:
    with pytest.raises(OperationDomainValidationError):
        invoke_operation(
            "matrix.cyclic.cyclotomic_inclusion.compute",
            {"source": {"order": 4}, "target": {"order": 6}},
            Catalog.open(),
        )


def test_native_integer_generator_image_coordinates_obey_height_bound() -> None:
    source = RationalCyclotomicField(order=3)
    target = RationalCyclotomicField(order=6)
    valid_boundary = CyclotomicFieldInclusion(
        source=source,
        target=target,
        generator_image=(
            CanonicalRational(num=10**255, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    assert len(str(valid_boundary.generator_image[0].num)) == 256

    with pytest.raises(ValidationError, match="256-digit bound"):
        CyclotomicFieldInclusion(
            source=source,
            target=target,
            generator_image=(
                CanonicalRational(num=10**256, den=1),
                CanonicalRational(num=0, den=1),
            ),
        )
