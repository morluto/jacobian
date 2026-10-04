"""Exact standard inclusions between rational cyclotomic fields."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Poly, cyclotomic_poly, symbols

import jacobian.math.matrices.cyclic_linear.operations as cyclic_operations
from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear import (
    CyclotomicFieldInclusion,
    RationalCyclotomicElement,
    RationalCyclotomicField,
    apply_cyclotomic_field_inclusion,
    compose_cyclotomic_field_inclusions,
    cyclotomic_field_inclusion,
)
from jacobian.math.matrices.cyclic_linear._models import (
    CyclotomicElementMapRequest,
    CyclotomicFieldInclusionRequest,
)
from jacobian.math.matrices.cyclic_linear._tools import TOOLS


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
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    assert tuple(value.as_fraction() for value in inclusion.generator_image) == (
        Fraction(-1),
        Fraction(1),
    )

    mapped = apply_cyclotomic_field_inclusion(inclusion, _element(3, (1, 1), (2, 1)))
    assert tuple(value.as_fraction() for value in mapped.coefficients_ascending) == (
        Fraction(-1),
        Fraction(2),
    )


@pytest.mark.parametrize("operand", ("first", "second"))
def test_composition_reports_forged_generator_image_operand(operand: str) -> None:
    first = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    second = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=6), RationalCyclotomicField(order=12)
    )
    forged = first if operand == "first" else second
    forged = forged.model_copy(
        update={
            "generator_image": tuple(
                CanonicalRational(num=0, den=1) for _ in forged.generator_image
            )
        }
    )
    if operand == "first":
        first = forged
    else:
        second = forged

    with pytest.raises(OperationDomainValidationError) as error:
        compose_cyclotomic_field_inclusions(first, second)

    assert error.value.errors()[0]["loc"] == (operand, "generator_image")


@pytest.mark.parametrize(
    "source_order,target_order", [(1, 1), (1, 8), (2, 4), (3, 12), (5, 10), (8, 16)]
)
def test_element_map_matches_independent_polynomial_substitution(
    source_order: int, target_order: int
) -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=source_order),
        RationalCyclotomicField(order=target_order),
    )
    degree = inclusion.source.degree
    element = _element(
        source_order, *((index - 2, index + 1) for index in range(degree))
    )
    actual = apply_cyclotomic_field_inclusion(inclusion, element)

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
    tool = next(
        t
        for t in TOOLS
        if t.operation_id == "matrix.cyclic.cyclotomic_inclusion.compute"
    )
    direct = tool.run(
        tool.request_type.model_validate(
            {"source": {"order": 3}, "target": {"order": 12}}
        )
    ).model_dump(mode="json")
    first = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    second = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=6), RationalCyclotomicField(order=12)
    )
    composed = compose_cyclotomic_field_inclusions(first, second)
    assert composed.model_dump(mode="json") == direct

    request = CyclotomicElementMapRequest(
        inclusion=composed,
        element=_element(3, (1, 1), (2, 1)),
    )
    decoded = CyclotomicElementMapRequest.model_validate_json(
        encode_strict_json(request.model_dump(mode="json")), strict=True
    )
    mapped = apply_cyclotomic_field_inclusion(decoded.inclusion, decoded.element)
    assert tuple(value.as_fraction() for value in mapped.coefficients_ascending) == (
        Fraction(-1),
        Fraction(0),
        Fraction(2),
        Fraction(0),
    )


def test_composition_is_native_only_and_not_a_catalog_operation() -> None:
    """Composition is a call-order helper, not a distinct postcondition.

    ``cyclotomic_field_inclusion(first.source, second.target)`` already
    publishes the same canonical inclusion, so a catalog entry for
    composition would add a redundant discovery target that describes a
    call order rather than a mathematical relation.
    """

    assert all(
        tool.operation_id != "matrix.cyclic.cyclotomic_inclusion.compose"
        for tool in TOOLS
    )
    first = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    second = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=6), RationalCyclotomicField(order=12)
    )
    assert compose_cyclotomic_field_inclusions(first, second) == (
        cyclotomic_field_inclusion(
            RationalCyclotomicField(order=3), RationalCyclotomicField(order=12)
        )
    )


def test_identity_inclusion_accepts_large_bounded_element() -> None:
    field = RationalCyclotomicField(order=3)
    inclusion = cyclotomic_field_inclusion(field, field)
    element = _element(3, (1, 10**129), (0, 1))
    mapped = apply_cyclotomic_field_inclusion(inclusion, element)
    assert mapped == element


def test_sparse_constant_maps_at_height_boundary_under_nonidentity_inclusion() -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=5), RationalCyclotomicField(order=10)
    )
    element = _element(5, (10**255, 1), (0, 1), (0, 1), (0, 1))
    mapped = apply_cyclotomic_field_inclusion(inclusion, element)
    assert mapped.field == inclusion.target
    assert mapped.coefficients_ascending[0].as_fraction() == 10**255
    assert all(value.as_fraction() == 0 for value in mapped.coefficients_ascending[1:])


def test_target_coordinate_cancellation_is_kept_within_height_boundary() -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    value = 9 * 10**255
    element = _element(3, (value, 1), (value, 1))

    mapped = apply_cyclotomic_field_inclusion(inclusion, element)

    assert tuple(c.as_fraction() for c in mapped.coefficients_ascending) == (
        Fraction(0),
        Fraction(value),
    )


def test_mapped_coordinate_denominators_are_bounded_after_reduction() -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=4), RationalCyclotomicField(order=8)
    )
    p = 10**255 + 1
    q = 10**255 + 3
    element = _element(4, (1, p), (1, q))

    mapped = apply_cyclotomic_field_inclusion(inclusion, element)

    assert tuple(c.as_fraction() for c in mapped.coefficients_ascending) == (
        Fraction(1, p),
        Fraction(0),
        Fraction(1, q),
        Fraction(0),
    )


def test_mapped_scalar_growth_is_admitted_before_coordinate_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cyclic_operations, "MAX_CYCLIC_MAP_INTERMEDIATE_DIGITS", 500)
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    p = 10**255 + 1
    q = 10**255 + 3
    element = _element(3, (1, p), (1, q))

    with pytest.raises(OperationResourceAdmissionError) as error:
        apply_cyclotomic_field_inclusion(inclusion, element)

    assert error.value.errors()[0]["type"] == (
        "matrix.cyclic.element_intermediate_digits_bound"
    )


def test_catalog_element_map_preserves_resource_admission_classification() -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    value = 9 * 10**255
    element = _element(3, (value, 1), (-value, 1))
    tool = next(
        t for t in TOOLS if t.operation_id == "matrix.cyclic.cyclotomic_element.map"
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        tool.run(CyclotomicElementMapRequest(inclusion=inclusion, element=element))

    assert error.value.errors()[0]["type"] == "matrix.cyclic.element_height_bound"


def test_inclusion_rejects_nondividing_parent() -> None:
    tool = next(
        t
        for t in TOOLS
        if t.operation_id == "matrix.cyclic.cyclotomic_inclusion.compute"
    )
    with pytest.raises(OperationDomainValidationError) as error:
        tool.run(
            CyclotomicFieldInclusionRequest.model_validate(
                {"source": {"order": 4}, "target": {"order": 6}}
            )
        )
    assert error.value.errors()[0]["loc"] == ("target",)


def test_composition_and_element_parent_errors_point_to_real_fields() -> None:
    first = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    second = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=4), RationalCyclotomicField(order=8)
    )
    with pytest.raises(OperationDomainValidationError) as composition_error:
        compose_cyclotomic_field_inclusions(first, second)
    assert composition_error.value.errors()[0]["loc"] == ("second", "source")

    wrong_parent_element = _element(4, (1, 1), (0, 1))
    with pytest.raises(OperationDomainValidationError) as element_error:
        apply_cyclotomic_field_inclusion(first, wrong_parent_element)
    assert element_error.value.errors()[0]["loc"] == ("element", "field")


def test_invalid_map_operands_report_their_own_paths() -> None:
    inclusion = cyclotomic_field_inclusion(
        RationalCyclotomicField(order=3), RationalCyclotomicField(order=6)
    )
    element = _element(3, (1, 1), (0, 1))
    forged_inclusion = inclusion.model_copy(update={"generator_image": ()})
    with pytest.raises(OperationDomainValidationError) as inclusion_error:
        apply_cyclotomic_field_inclusion(forged_inclusion, element)
    assert inclusion_error.value.errors()[0]["loc"] == ("inclusion",)

    forged_element = element.model_copy(update={"coefficients_ascending": ()})
    with pytest.raises(OperationDomainValidationError) as element_error:
        apply_cyclotomic_field_inclusion(inclusion, forged_element)
    assert element_error.value.errors()[0]["loc"] == ("element",)


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

    with pytest.raises(ValidationError) as exc_info:
        CyclotomicFieldInclusion(
            source=source,
            target=target,
            generator_image=(
                CanonicalRational(num=10**256, den=1),
                CanonicalRational(num=0, den=1),
            ),
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "matrix.cyclic.cyclotomic_coordinate_digits"
    )
