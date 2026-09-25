"""Exact composita of the currently supported quadratic QQ splitting fields."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.galois._models import (
    GaloisCompositumResult,
    QQSplittingField,
)
from jacobian.math.number_theory.galois.operations import _canonical_splitting_field
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbedding,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)

MAX_COMPOSITUM_INPUT_COEFFICIENT = 10**12
MAX_COMPOSITUM_OUTPUT_BYTES = 32_768


def _element(
    field: SimpleNumberFieldPresentation, coefficients: tuple[Fraction, ...]
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in coefficients
        ),
    )


def _quadratic_data(field: QQSplittingField) -> tuple[int, int, int, int]:
    a, b, c = field.extension.coefficients_descending
    if field.degree != 2 or a == 0:
        raise ValueError("expected an irreducible quadratic splitting field")
    return a, b, c, b * b - 4 * a * c


def _square_root_fraction(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        return None
    return Fraction(numerator, denominator)


def _rational_field() -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(coefficients_descending=(1, 0))


def _identity_embedding(
    field: SimpleNumberFieldPresentation,
) -> SimpleNumberFieldEmbedding:
    if field.degree == 1:
        generator_image = _element(field, (Fraction(0),))
    else:
        generator_image = _element(
            field, (Fraction(0), Fraction(1)) + (Fraction(0),) * (field.degree - 2)
        )
    return SimpleNumberFieldEmbedding(
        source=field, target=field, generator_image=generator_image
    )


def _inclusion_from_linear_image(
    source: SimpleNumberFieldPresentation,
    target: SimpleNumberFieldPresentation,
    coefficients: tuple[Fraction, ...],
) -> SimpleNumberFieldEmbedding:
    return SimpleNumberFieldEmbedding(
        source=source,
        target=target,
        generator_image=_element(target, coefficients),
    )


def _admit_source(field: QQSplittingField, location: str) -> QQSplittingField:
    if not isinstance(field, QQSplittingField):
        raise OperationDomainValidationError(
            location=(location,),
            code="galois_theory.splitting_field_type",
            message="compositum inputs must be exact QQ splitting-field values",
        )
    try:
        canonical = QQSplittingField.model_validate(field.model_dump())
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=(location,),
            code="galois_theory.compositum_source",
            message=f"{location} must be a canonical bounded QQ splitting-field value",
        ) from exc
    coefficients = tuple(
        term.coefficient.num for term in canonical.source.polynomial.terms
    )
    if any(abs(value) > MAX_COMPOSITUM_INPUT_COEFFICIENT for value in coefficients):
        raise OperationResourceAdmissionError(
            location=(location, "source"),
            code="galois_theory.compositum_coefficient_bound",
            message="compositum input coefficients exceed the 10^12 envelope",
        )
    return canonical


def galois_compositum(
    left: QQSplittingField, right: QQSplittingField
) -> GaloisCompositumResult:
    """Return the exact compositum and both source-field inclusions.

    Each input is QQ or a quadratic splitting field. The result degree is at
    most four. Distinct quadratic classes use theta=sqrt(D1)+sqrt(D2), whose
    quartic polynomial and both inclusion maps are computed exactly.
    """

    if not isinstance(left, QQSplittingField) or not isinstance(
        right, QQSplittingField
    ):
        raise OperationDomainValidationError(
            location=(),
            code="galois_theory.splitting_field_type",
            message="compositum inputs must be exact QQ splitting-field values",
        )
    left = _admit_source(left, "left")
    right = _admit_source(right, "right")
    # The source carriers cap degree at two and integral coefficients at 10^12.
    # A conservative 4x input size plus fixed map/model overhead admits every
    # possible degree-four output before replaying either exact splitting field.
    input_bytes = len(left.model_dump_json()) + len(right.model_dump_json())
    if 4 * input_bytes + 8192 > MAX_COMPOSITUM_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=(),
            code="galois_theory.compositum_output_bound",
            message="predicted compositum output exceeds the 32768-byte envelope",
        )
    left = _canonical_splitting_field(left, location=("left",))
    right = _canonical_splitting_field(right, location=("right",))
    if left.degree == 1 and right.degree == 1:
        field = _rational_field()
        left_embedding = _identity_embedding(field)
        right_embedding = _identity_embedding(field)
    elif left.degree == 1:
        field = right.extension
        left_embedding = _inclusion_from_linear_image(
            left.extension, field, (Fraction(0),) * field.degree
        )
        right_embedding = _identity_embedding(field)
    elif right.degree == 1:
        field = left.extension
        left_embedding = _identity_embedding(field)
        right_embedding = _inclusion_from_linear_image(
            right.extension, field, (Fraction(0),) * field.degree
        )
    else:
        a1, b1, _c1, d1 = _quadratic_data(left)
        a2, b2, _c2, d2 = _quadratic_data(right)
        ratio_root = _square_root_fraction(Fraction(d2, d1))
        if ratio_root is not None:
            field = left.extension
            left_embedding = _identity_embedding(field)
            alpha_right = (
                Fraction(-b2, 2 * a2) + ratio_root * Fraction(b1, 2 * a2),
                ratio_root * Fraction(a1, a2),
            )
            right_embedding = _inclusion_from_linear_image(
                right.extension, field, alpha_right
            )
        else:
            total = d1 + d2
            difference = d1 - d2
            constant = difference * difference
            # Primitive element theta=sqrt(D1)+sqrt(D2) of the compositum.
            field = SimpleNumberFieldPresentation(
                coefficients_descending=(1, 0, -2 * total, 0, constant)
            )
            common_denominator = 2 * constant
            radical_left = (
                Fraction(0),
                Fraction(constant + 2 * total * difference, common_denominator),
                Fraction(0),
                Fraction(-difference, common_denominator),
            )
            radical_right = (
                Fraction(0),
                Fraction(constant - 2 * total * difference, common_denominator),
                Fraction(0),
                Fraction(difference, common_denominator),
            )
            image_left = tuple(value / (2 * a1) for value in radical_left)
            image_right = tuple(value / (2 * a2) for value in radical_right)
            image_left = (image_left[0] - Fraction(b1, 2 * a1), *image_left[1:])
            image_right = (image_right[0] - Fraction(b2, 2 * a2), *image_right[1:])
            left_embedding = _inclusion_from_linear_image(
                left.extension, field, image_left
            )
            right_embedding = _inclusion_from_linear_image(
                right.extension, field, image_right
            )
    _verify_embedding_relation(left.extension, left_embedding)
    _verify_embedding_relation(right.extension, right_embedding)
    result = GaloisCompositumResult(
        left=left,
        right=right,
        compositum=field,
        left_embedding=left_embedding,
        right_embedding=right_embedding,
    )
    return result


def _multiply_in_field(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
    field: SimpleNumberFieldPresentation,
) -> tuple[Fraction, ...]:
    degree = field.degree
    product = [Fraction(0)] * (2 * degree - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            product[i + j] += left_value * right_value
    defining = tuple(Fraction(value) for value in field.coefficients_descending)
    leading = defining[0]
    for power in range(len(product) - 1, degree - 1, -1):
        coefficient = product[power]
        if not coefficient:
            continue
        shift = power - degree
        for basis_power in range(degree):
            product[shift + basis_power] -= (
                coefficient * defining[degree - basis_power] / leading
            )
    return tuple(product[:degree])


def _verify_embedding_relation(
    source: SimpleNumberFieldPresentation,
    embedding: SimpleNumberFieldEmbedding,
) -> None:
    image = tuple(
        coefficient.as_fraction()
        for coefficient in embedding.generator_image.coefficients_ascending
    )
    target = embedding.target
    degree = source.degree
    if degree == 1:
        if any(image):
            raise ArithmeticError("the rational-field generator image must be zero")
        return
    if degree != 2:
        raise ArithmeticError("compositum source degree must be at most two")
    a, b, c = (Fraction(value) for value in source.coefficients_descending)
    square = _multiply_in_field(image, image, target)
    relation = tuple(
        a * square[i] + b * image[i] + (c if i == 0 else 0)
        for i in range(target.degree)
    )
    if any(relation):
        raise ArithmeticError(
            "compositum inclusion does not preserve the source relation"
        )
