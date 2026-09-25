"""Exact maps between bounded simple number fields over QQ.

A map is specified by the image of the source primitive element.  The kernel
checks that both presentations are fields and that the proposed image
annihilates the source polynomial before transporting any element.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import ValidationError, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)

# The coordinate-growth estimate below admits the identity map only up to
# source degree six (1 + 6*6*6 = 217 digits); degrees seven and eight would be
# advertised but unusable, so the admitted contract stays aligned with the
# estimate.
MAX_FIELD_MAP_DEGREE = 6
MAX_FIELD_MAP_INPUT_DIGITS = 32


def _error(
    code: str, message: str, *, location: tuple[str | int, ...]
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location, code=f"number_field.embedding.{code}", message=message
    )


class SimpleNumberFieldEmbedding(StrictModel):
    """A QQ-field embedding determined by the image of its source generator."""

    source: SimpleNumberFieldPresentation
    target: SimpleNumberFieldPresentation
    generator_image: SimpleNumberFieldElement

    @model_validator(mode="after")
    def bind_image(self) -> Self:
        if self.generator_image.presentation != self.target:
            raise PydanticCustomError(
                "number_field.embedding.image_parent",
                "generator image must belong to the target field",
            )
        return self


class SimpleNumberFieldEmbeddingRequest(StrictModel):
    source: SimpleNumberFieldPresentation
    target: SimpleNumberFieldPresentation
    generator_image: SimpleNumberFieldElement
    element: SimpleNumberFieldElement

    @model_validator(mode="after")
    def bind_elements(self) -> Self:
        if self.generator_image.presentation != self.target:
            raise PydanticCustomError(
                "number_field.embedding.image_parent",
                "generator image must belong to the target field",
            )
        if self.element.presentation != self.source:
            raise PydanticCustomError(
                "number_field.embedding.element_parent",
                "element must belong to the source field",
            )
        return self


class SimpleNumberFieldEmbeddingResult(StrictModel):
    embedding: SimpleNumberFieldEmbedding
    source_element: SimpleNumberFieldElement
    image: SimpleNumberFieldElement

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        if (
            self.source_element.presentation != self.embedding.source
            or self.image.presentation != self.embedding.target
        ):
            raise PydanticCustomError(
                "number_field.embedding.result_parent",
                "source element and image must match the embedding parents",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        embedding: SimpleNumberFieldEmbedding,
        source_element: SimpleNumberFieldElement,
        image: SimpleNumberFieldElement,
    ) -> Self:
        return cls.model_construct(
            embedding=embedding, source_element=source_element, image=image
        )


def _fractions(element: SimpleNumberFieldElement) -> list[Fraction]:
    return [value.as_fraction() for value in element.coefficients_ascending]


def _element(
    presentation: SimpleNumberFieldPresentation, values: list[Fraction]
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in values
        ),
    )


def _add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    return [a + b for a, b in zip(left, right, strict=True)]


def _multiply(
    presentation: SimpleNumberFieldPresentation,
    left: list[Fraction],
    right: list[Fraction],
) -> list[Fraction]:
    degree = presentation.degree
    product = [Fraction(0)] * (2 * degree - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    coefficients = [Fraction(value) for value in presentation.coefficients_descending]
    leading = coefficients[0]
    for power in range(2 * degree - 2, degree - 1, -1):
        high = product[power]
        for offset in range(degree):
            product[power - 1 - offset] -= high * coefficients[offset + 1] / leading
    return product[:degree]


def _evaluate_integer_polynomial(
    presentation: SimpleNumberFieldPresentation,
    coefficients_descending: tuple[int, ...],
    value: list[Fraction],
) -> list[Fraction]:
    result = [Fraction(0)] * presentation.degree
    for coefficient in coefficients_descending:
        result = _multiply(presentation, result, value)
        result[0] += coefficient
    return result


def _admit(request: SimpleNumberFieldEmbeddingRequest) -> None:
    if (
        request.source.degree > MAX_FIELD_MAP_DEGREE
        or request.target.degree > MAX_FIELD_MAP_DEGREE
    ):
        raise _error(
            "degree_bound",
            f"field maps admit source and target degrees at most {MAX_FIELD_MAP_DEGREE}",
            location=("source",),
        )
    values = (
        *request.source.coefficients_descending,
        *request.target.coefficients_descending,
    )
    if any(len(str(abs(value))) > MAX_FIELD_MAP_INPUT_DIGITS for value in values):
        raise OperationResourceAdmissionError(
            location=("source",),
            code="number_field.embedding.coefficient_bound",
            message=f"field-map polynomial coefficients are limited to {MAX_FIELD_MAP_INPUT_DIGITS} digits",
        )
    rationals = (
        *request.generator_image.coefficients_ascending,
        *request.element.coefficients_ascending,
    )
    if any(
        canonical_rational_component_digits(value) > MAX_FIELD_MAP_INPUT_DIGITS
        for value in rationals
    ):
        raise OperationResourceAdmissionError(
            location=("element",),
            code="number_field.embedding.coordinate_bound",
            message=f"field-map coordinates are limited to {MAX_FIELD_MAP_INPUT_DIGITS} digits",
        )
    # Horner evaluation uses at most source degree multiplications in the
    # target quotient.  This estimate bounds exact rational coordinate growth.
    input_digits = max(
        1,
        *(len(str(abs(value))) for value in values),
        *(canonical_rational_component_digits(value) for value in rationals),
    )
    growth = input_digits + request.source.degree * request.target.degree * (
        3 * input_digits + 3
    )
    if growth > 256:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="number_field.embedding.output_bound",
            message="predicted field-map coordinate growth exceeds the 256-digit exact-value limit",
        )
    from sympy import Poly, Symbol

    for label, field in (("source", request.source), ("target", request.target)):
        poly = Poly.from_list(
            list(field.coefficients_descending), Symbol("x"), domain="QQ"
        )
        if not poly.is_irreducible:
            raise _error(
                "reducible_presentation",
                f"{label} presentation must define a field",
                location=(label,),
            )


def _canonical_request(
    request: SimpleNumberFieldEmbeddingRequest,
) -> SimpleNumberFieldEmbeddingRequest:
    """Revalidate the full request so model-constructed bindings cannot bypass them."""
    try:
        return SimpleNumberFieldEmbeddingRequest.model_validate(request.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _error(
            "invalid_request",
            "embedding request must be a canonical validated field-map specification",
            location=("request",),
        ) from exc


def apply_simple_number_field_embedding(
    request: SimpleNumberFieldEmbeddingRequest,
) -> SimpleNumberFieldEmbeddingResult:
    """Validate a proposed exact field map and transport one source element."""
    request = _canonical_request(request)
    _admit(request)
    target = request.target
    image_coordinates = _fractions(request.generator_image)
    relation_value = _evaluate_integer_polynomial(
        target, request.source.coefficients_descending, image_coordinates
    )
    if any(relation_value):
        raise _error(
            "image_not_root",
            "generator_image must be an exact root of the source defining polynomial",
            location=("generator_image",),
        )

    element_coordinates = _fractions(request.element)
    mapped = [Fraction(0)] * target.degree
    power = [Fraction(0)] * target.degree
    power[0] = Fraction(1)
    for coefficient in element_coordinates:
        for index in range(target.degree):
            mapped[index] += coefficient * power[index]
        power = _multiply(target, power, image_coordinates)

    embedding = SimpleNumberFieldEmbedding.model_construct(
        source=request.source, target=target, generator_image=request.generator_image
    )
    return SimpleNumberFieldEmbeddingResult._from_kernel(
        embedding=embedding,
        source_element=request.element,
        image=_element(target, mapped),
    )


__all__ = [
    "SimpleNumberFieldEmbedding",
    "SimpleNumberFieldEmbeddingRequest",
    "SimpleNumberFieldEmbeddingResult",
    "apply_simple_number_field_embedding",
]
