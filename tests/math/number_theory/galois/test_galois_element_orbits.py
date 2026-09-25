from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._models import (
    ElementEmbeddingOrbitRequest,
    SplittingFieldRequest,
)
from jacobian.math.number_theory.galois._tools import TOOLS
from jacobian.math.number_theory.galois.operations import (
    element_embedding_orbit,
    splitting_field,
)
from jacobian.math.number_theory.number_fields.values import SimpleNumberFieldElement
from jacobian.math.polynomials.values import MonicPolynomial, RationalPolynomial


def _poly(coefficients: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": [degree],
                    }
                    for degree, coefficient in reversed(tuple(enumerate(coefficients)))
                    if coefficient
                ]
            },
        }
    )


def _element(field, *coordinates: int) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field.extension,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(value)) for value in coordinates
        ),
    )


def _polynomial_coefficients(polynomial: RationalPolynomial) -> tuple[Fraction, ...]:
    degree = max(term.exponents[0] for term in polynomial.polynomial.terms)
    values = [Fraction(0)] * (degree + 1)
    for term in polynomial.polynomial.terms:
        values[term.exponents[0]] = term.coefficient.as_fraction()
    return tuple(values)


def test_element_orbit_retains_exact_maps_stabilizer_and_minimal_polynomial() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    alpha = _element(field, 0, 1)

    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=alpha)
    )

    assert result.source_element == alpha
    assert len(result.action) == 2
    assert tuple(item.automorphism.root_permutation for item in result.action) == (
        (0, 1),
        (1, 0),
    )
    assert tuple(
        tuple(value.as_fraction() for value in image.coefficients_ascending)
        for image in result.orbit
    ) == ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1)))
    assert result.orbit_size == 2
    assert len(result.stabilizer.elements) == 1
    assert _polynomial_coefficients(result.minimal_polynomial) == (
        Fraction(-2), Fraction(0), Fraction(1)
    )
    assert type(result).model_validate(result.model_dump()) == result


def test_rational_element_has_singleton_orbit_and_full_stabilizer() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    three = _element(field, 3, 0)

    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=three)
    )

    assert result.orbit == (three,)
    assert result.orbit_size == 1
    assert len(result.stabilizer.elements) == 2
    assert _polynomial_coefficients(result.minimal_polynomial) == (
        Fraction(-3), Fraction(1)
    )


def test_degree_one_qq_field_has_identity_orbit_and_stabilizer() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-1, 1)))).field
    five = _element(field, 5)

    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=five)
    )

    assert result.orbit == (five,)
    assert result.orbit_size == 1
    assert len(result.action) == 1
    assert result.action[0].automorphism.root_permutation == (0,)
    assert result.action[0].automorphism.basis_images == (_element(field, 1),)
    assert result.action[0].image == five
    assert result.stabilizer.elements == (result.action[0].automorphism,)
    assert _polynomial_coefficients(result.minimal_polynomial) == (
        Fraction(-5), Fraction(1)
    )
    assert type(result).model_validate(result.model_dump()) == result

def test_orbit_accepts_wide_coordinates_when_zero_products_preserve_them() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    denominator = 10**127 + 1
    element = SimpleNumberFieldElement(
        presentation=field.extension,
        coefficients_ascending=(
            CanonicalRational.from_fraction(Fraction(1, denominator)),
            CanonicalRational.from_fraction(Fraction(1, denominator)),
        ),
    )

    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=element)
    )

    assert len(result.orbit) == 2
    assert {tuple(c.as_fraction() for c in value.coefficients_ascending) for value in result.orbit} == {
        (Fraction(1, denominator), Fraction(1, denominator)),
        (Fraction(1, denominator), Fraction(-1, denominator)),
    }


def test_orbit_accepts_rational_carrier_boundary_without_a_zero_addend() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    denominator = 10**255 + 1
    element = SimpleNumberFieldElement(
        presentation=field.extension,
        coefficients_ascending=(
            CanonicalRational.from_fraction(Fraction(1, denominator)),
            CanonicalRational.from_fraction(Fraction(0)),
        ),
    )

    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=element)
    )

    assert result.orbit == (element,)
    assert result.orbit_size == 1
    assert len(result.stabilizer.elements) == 2
    assert _polynomial_coefficients(result.minimal_polynomial) == (
        Fraction(-1, denominator),
        Fraction(1),
    )
    assert type(result).model_validate(result.model_dump()) == result


def test_orbit_minimal_polynomial_uses_monic_carrier() -> None:
    field = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    result = element_embedding_orbit(
        ElementEmbeddingOrbitRequest(field=field, element=_element(field, 0, 1))
    )

    assert isinstance(result.minimal_polynomial, MonicPolynomial)

    tampered = result.model_dump()
    tampered["minimal_polynomial"]["polynomial"]["terms"][0]["coefficient"] = {
        "num": 2,
        "den": 1,
    }
    with pytest.raises(ValidationError):
        type(result).model_validate(tampered)


def test_orbit_rejects_element_from_isomorphic_but_distinct_parent() -> None:
    first = splitting_field(SplittingFieldRequest(polynomial=_poly((-2, 0, 1)))).field
    second = splitting_field(SplittingFieldRequest(polynomial=_poly((-8, 0, 1)))).field

    with pytest.raises(ValueError, match="element must belong"):
        ElementEmbeddingOrbitRequest(field=first, element=_element(second, 0, 1))


def test_operation_is_published_with_a_valid_example() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "number_field.element.embedding_orbit.compute"
    )
    assert len(tool.examples) == 1
    assert tool.result_type is not None


def test_native_call_rejects_non_field_request_shape() -> None:
    with pytest.raises(OperationDomainValidationError):
        element_embedding_orbit(None)  # type: ignore[arg-type]
