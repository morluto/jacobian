from fractions import Fraction

import pytest

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
from jacobian.math.polynomials.values import RationalPolynomial


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
