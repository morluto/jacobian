from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbeddingRequest,
    apply_simple_number_field_embedding,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)


def _field(*coefficients: int) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(coefficients_descending=coefficients)


def _element(
    field: SimpleNumberFieldPresentation, *coefficients: int
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field,
        coefficients_ascending=[{"num": value, "den": 1} for value in coefficients],
    )


def test_exact_quadratic_embedding_into_quartic_transports_element() -> None:
    source = _field(1, 0, -2)
    target = _field(1, 0, 0, 0, -2)
    request = SimpleNumberFieldEmbeddingRequest(
        source=source,
        target=target,
        generator_image=_element(target, 0, 0, 1, 0),
        element=_element(source, 1, 1),
    )

    result = apply_simple_number_field_embedding(request)

    assert result.embedding.source == source
    assert result.embedding.target == target
    assert result.image == _element(target, 1, 0, 1, 0)


def test_rejects_image_that_does_not_satisfy_source_relation() -> None:
    source = _field(1, 0, -2)
    target = _field(1, 0, 0, 0, -2)
    request = SimpleNumberFieldEmbeddingRequest(
        source=source,
        target=target,
        generator_image=_element(target, 0, 1, 0, 0),
        element=_element(source, 0, 1),
    )

    with pytest.raises(OperationDomainValidationError, match="exact root"):
        apply_simple_number_field_embedding(request)


def test_rejects_reducible_source_before_map_evaluation() -> None:
    reducible = _field(1, 0, -1)
    target = _field(1, 0, 0, 0, -2)
    request = SimpleNumberFieldEmbeddingRequest(
        source=reducible,
        target=target,
        generator_image=_element(target, 1, 0, 0, 0),
        element=_element(reducible, 0, 1),
    )

    with pytest.raises(OperationDomainValidationError, match="must define a field"):
        apply_simple_number_field_embedding(request)


def test_models_bind_source_target_and_element_parents() -> None:
    source = _field(1, 0, -2)
    target = _field(1, 0, 0, 0, -2)
    with pytest.raises(ValidationError):
        SimpleNumberFieldEmbeddingRequest(
            source=source,
            target=target,
            generator_image=_element(source, 0, 1),
            element=_element(source, 0, 1),
        )


def test_transport_preserves_rational_coordinates() -> None:
    source = _field(1, 0, -2)
    target = _field(1, 0, 0, 0, -2)
    half_alpha = SimpleNumberFieldElement(
        presentation=source,
        coefficients_ascending=[{"num": 0, "den": 1}, {"num": 1, "den": 2}],
    )
    result = apply_simple_number_field_embedding(
        SimpleNumberFieldEmbeddingRequest(
            source=source,
            target=target,
            generator_image=_element(target, 0, 0, 1, 0),
            element=half_alpha,
        )
    )
    assert result.image.coefficients_ascending[2].as_fraction() == Fraction(1, 2)


def test_transport_preserves_products_against_independent_polynomial_remainder() -> (
    None
):
    from sympy import Poly, Symbol

    source = _field(1, 0, -2)
    target = _field(1, 0, 0, 0, -2)
    generator_image = _element(target, 0, 0, 1, 0)
    mapped_alpha = apply_simple_number_field_embedding(
        SimpleNumberFieldEmbeddingRequest(
            source=source,
            target=target,
            generator_image=generator_image,
            element=_element(source, 0, 1),
        )
    ).image
    mapped_one_plus_alpha = apply_simple_number_field_embedding(
        SimpleNumberFieldEmbeddingRequest(
            source=source,
            target=target,
            generator_image=generator_image,
            element=_element(source, 1, 1),
        )
    ).image
    mapped_product = apply_simple_number_field_embedding(
        SimpleNumberFieldEmbeddingRequest(
            source=source,
            target=target,
            generator_image=generator_image,
            element=_element(source, 2, 1),
        )
    ).image

    x = Symbol("x")
    target_poly = Poly.from_list(list(target.coefficients_descending), x, domain="QQ")
    left_poly = Poly.from_list(
        [
            value.as_fraction()
            for value in reversed(mapped_alpha.coefficients_ascending)
        ],
        x,
        domain="QQ",
    )
    right_poly = Poly.from_list(
        [
            value.as_fraction()
            for value in reversed(mapped_one_plus_alpha.coefficients_ascending)
        ],
        x,
        domain="QQ",
    )
    oracle_remainder = (left_poly * right_poly).rem(target_poly)
    expected = tuple(reversed(oracle_remainder.all_coeffs()))
    expected += (Fraction(0),) * (target.degree - len(expected))
    assert (
        tuple(value.as_fraction() for value in mapped_product.coefficients_ascending)
        == expected
    )
