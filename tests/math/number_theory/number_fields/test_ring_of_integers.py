"""Ring-of-integers bases for presented simple number fields (#1667)."""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldRingOfIntegersRequest,
)
from jacobian.math.number_theory.number_fields._ring_of_integers import (
    NumberFieldRingOfIntegersResult,
    ring_of_integers,
)
from jacobian.math.number_theory.number_fields._tools import (
    compute_ring_of_integers,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)


def _coordinates(result: NumberFieldRingOfIntegersResult) -> list[list[Fraction]]:
    return [
        [coefficient.as_fraction() for coefficient in element.coefficients_ascending]
        for element in result.basis
    ]


def _cyclotomic_coefficients(order: int) -> tuple[int, ...]:
    symbol = sympy.Symbol("alpha")
    polynomial = sympy.cyclotomic_poly(order, symbol)
    return tuple(
        int(coefficient)
        for coefficient in sympy.Poly(polynomial, symbol, domain=sympy.ZZ).all_coeffs()
    )


def test_golden_field_is_not_the_naive_power_basis() -> None:
    """Q(sqrt(5)) has integral basis 1 and (1+sqrt(5))/2, so halve coordinates."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    assert result.field_discriminant == 5
    assert _coordinates(result) == [
        [Fraction(1), Fraction(0)],
        [Fraction(1, 2), Fraction(1, 2)],
    ]
    assert all(
        isinstance(element, SimpleNumberFieldElement) for element in result.basis
    )
    assert all(element.presentation is result.field for element in result.basis)


def test_squarefree_discriminant_field_keeps_the_power_basis() -> None:
    """Q(sqrt(2)) is already integrally closed, so its basis is 1, sqrt(2)."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -2))
    )
    assert result.field_discriminant == 8
    assert _coordinates(result) == [
        [Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(1)],
    ]


@pytest.mark.parametrize(
    ("order", "discriminant"),
    ((3, -3), (4, -4), (5, 125), (7, -16807)),
)
def test_cyclotomic_fields_have_full_power_basis(order: int, discriminant: int) -> None:
    """A cyclotomic field of prime order is monogenic with 1, zeta, ..., zeta^n."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(
            coefficients_descending=_cyclotomic_coefficients(order)
        )
    )
    degree = result.field.degree
    assert result.field_discriminant == discriminant
    assert _coordinates(result) == [
        [Fraction(1 if row == column else 0) for column in range(degree)]
        for row in range(degree)
    ]


def test_basis_vectors_are_bounded_rationals_of_full_degree() -> None:
    """Every returned vector spans the complete power basis with exact rationals."""
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, 0, 0, -2))
    )
    assert len(result.basis) == result.field.degree
    assert all(
        len(element.coefficients_ascending) == result.field.degree
        for element in result.basis
    )


def test_reducible_polynomial_is_rejected_without_a_basis() -> None:
    """A reducible defining polynomial does not present a number field."""
    # (x^2 - 2)(x^2 - 3) is reducible over QQ while remaining primitive.
    with pytest.raises(OperationDomainValidationError) as error:
        ring_of_integers(
            SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5, 0, 6))
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.defining_polynomial_must_be_irreducible"
    )


def test_ring_of_integers_round_trips_through_strict_json() -> None:
    """The declared basis survives strict JSON serialization unchanged."""
    result = compute_ring_of_integers(
        NumberFieldRingOfIntegersRequest(
            field=SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
        )
    )
    restored = NumberFieldRingOfIntegersResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


def test_ring_of_integers_request_schema_and_parser_share_degree_boundary() -> None:
    operation = Catalog.open().inspect("number_field.ring_of_integers.compute")
    assert operation is not None
    field_schema = operation.input_schema["properties"]["field"]
    assert field_schema["properties"]["coefficients_descending"]["maxItems"] == 32

    schema_validator = Draft202012Validator(operation.input_schema)
    schema_validator.validate(
        {"field": {"coefficients_descending": ["1", *(["0"] * 30), "-2"]}}
    )
    with pytest.raises(JSONSchemaValidationError):
        schema_validator.validate(
            {"field": {"coefficients_descending": ["1", *(["0"] * 31), "-2"]}}
        )

    accepted = NumberFieldRingOfIntegersRequest(
        field=SimpleNumberFieldPresentation(coefficients_descending=(1, *(0,) * 30, -2))
    )
    assert accepted.field.degree == 31

    with pytest.raises(ValidationError, match="degree at most 31"):
        NumberFieldRingOfIntegersRequest(
            field={"coefficients_descending": (1, *(0,) * 31, -2)}
        )


def test_ring_of_integers_result_rejects_malformed_basis_on_deserialization() -> None:
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    payload = result.model_dump(mode="json")
    payload["basis"] = payload["basis"][:1]
    with pytest.raises(ValidationError) as error:
        NumberFieldRingOfIntegersResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_basis_length"
    )


def test_ring_of_integers_result_rejects_a_basis_from_another_field() -> None:
    result = ring_of_integers(
        SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )
    payload = result.model_dump(mode="json")
    payload["basis"][1]["presentation"]["coefficients_descending"] = [
        "1",
        "0",
        "-2",
    ]

    with pytest.raises(ValidationError) as error:
        NumberFieldRingOfIntegersResult.model_validate_json(
            encode_strict_json(payload), strict=True
        )
    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_basis_field"
    )
