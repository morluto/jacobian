"""Exact order in one real quadratic field."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.arithmetic._integer_predicates import is_square_free

_MAX_RADICAND = 1_000_000
_MAX_DIGITS = 256
# For a,b with numerator and denominator at most 256 decimal digits,
# a^2 - d*b^2 has a denominator of at most 1,024 digits and a numerator
# of at most 1,032 digits after bringing the two terms to that denominator
# (d <= 10^6).  The trace is smaller.  This covers production and reconstruction.
_MAX_EMBEDDING_PROFILE_RESULT_DIGITS = 1_032


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by real-quadratic contracts."""

    return PydanticCustomError(f"real_quadratic.{reason}", message)


def _require_bounded_rational(
    value: CanonicalRational, *, max_digits: int, label: str
) -> None:
    if (
        len(value.num.lstrip("-")) > max_digits
        or len(value.den.lstrip("-")) > max_digits
    ):
        raise _validation_error(
            "rational_bound_exceeded", f"{label} exceeds the {max_digits}-digit bound"
        )


RealQuadraticSignBasis = Literal[
    "RATIONAL_ONLY",
    "RADICAL_ONLY",
    "SAME_SIGN",
    "OPPOSING_SIGNS_SQUARED_MAGNITUDES",
]
RealQuadraticEmbedding = Literal["POSITIVE_ROOT", "NEGATIVE_ROOT"]
RealQuadraticEmbeddingConvention = Literal["REAL_QUADRATIC_ROOTS_V1"]


def _order(left: Fraction, right: Fraction) -> Literal["LT", "EQ", "GT"]:
    return "LT" if left < right else "GT" if left > right else "EQ"


def _sign(a: Fraction, b: Fraction, d: int) -> int:
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    rational_square = a * a
    radical_square = b * b * d
    if rational_square == radical_square:
        raise ValueError("square-free quadratic magnitudes cannot tie")
    dominant = b if radical_square > rational_square else a
    return (dominant > 0) - (dominant < 0)


class RealQuadraticValue(StrictModel):
    rational_part: CanonicalRational
    radical_coefficient: CanonicalRational
    radicand: StrictInt = Field(ge=2, le=_MAX_RADICAND)

    @model_validator(mode="after")
    def require_canonical_field_value(self) -> Self:
        _require_bounded_rational(
            self.rational_part, max_digits=_MAX_DIGITS, label="rational part"
        )
        _require_bounded_rational(
            self.radical_coefficient,
            max_digits=_MAX_DIGITS,
            label="radical coefficient",
        )
        if not is_square_free(self.radicand):
            raise _validation_error(
                "radicand_not_square_free",
                "real-quadratic radicand must be square-free",
            )
        return self


def _require_order_admission(
    left: RealQuadraticValue, right: RealQuadraticValue
) -> None:
    """Validate the owner-local envelope of an exact order comparison."""

    if left.radicand != right.radicand:
        raise OperationDomainValidationError(
            location=("right", "radicand"),
            code="real_quadratic.radicand_mismatch",
            message="comparison requires one shared radicand",
        )
    difference_components = (
        left.rational_part.as_fraction() - right.rational_part.as_fraction(),
        left.radical_coefficient.as_fraction()
        - right.radical_coefficient.as_fraction(),
    )
    if any(
        len(format_canonical_integer(component.numerator).lstrip("-")) > _MAX_DIGITS
        or len(format_canonical_integer(component.denominator)) > _MAX_DIGITS
        for component in difference_components
    ):
        raise OperationResourceAdmissionError(
            location=(),
            code="real_quadratic.difference_bound_exceeded",
            message="exact quadratic difference exceeds the 256-digit result bound",
        )


def _embedding_scalars(
    element: RealQuadraticValue,
) -> tuple[Fraction, Fraction]:
    """Return the exact trace and norm of one real-quadratic element."""

    rational_part = element.rational_part.as_fraction()
    radical_coefficient = element.radical_coefficient.as_fraction()
    return (
        2 * rational_part,
        rational_part * rational_part
        - element.radicand * radical_coefficient * radical_coefficient,
    )


class RealQuadraticEmbeddingImage(StrictModel):
    """One named embedding image in the canonical positive-root coordinate."""

    embedding: RealQuadraticEmbedding
    value: RealQuadraticValue


class RealQuadraticEmbeddingProfile(StrictModel):
    """The complete exact Archimedean embedding profile of one element.

    ``value`` is always represented with the canonical positive square root
    in ``RealQuadraticValue``.  ``embedding`` records whether it is the
    identity map sqrt(d) -> +sqrt(d), or the conjugate map sqrt(d) ->
    -sqrt(d), so the profile never conflates an abstract field element with
    an unlabeled numerical approximation.
    """

    source: RealQuadraticValue
    real_embedding_count: Literal[2] = 2
    complex_conjugate_pair_count: Literal[0] = 0
    images: tuple[RealQuadraticEmbeddingImage, RealQuadraticEmbeddingImage]
    trace: CanonicalRational
    norm: CanonicalRational
    convention: RealQuadraticEmbeddingConvention = "REAL_QUADRATIC_ROOTS_V1"

    @model_validator(mode="after")
    def bind_profile_to_source(self) -> Self:
        source_conjugate = RealQuadraticValue(
            rational_part=self.source.rational_part,
            radical_coefficient=CanonicalRational.from_fraction(
                -self.source.radical_coefficient.as_fraction()
            ),
            radicand=self.source.radicand,
        )
        expected_images = (
            RealQuadraticEmbeddingImage(embedding="POSITIVE_ROOT", value=self.source),
            RealQuadraticEmbeddingImage(
                embedding="NEGATIVE_ROOT", value=source_conjugate
            ),
        )
        if self.images != expected_images:
            raise _validation_error(
                "embedding_images_mismatch",
                "images must use the ordered embeddings of the source field",
            )
        expected_trace, expected_norm = _embedding_scalars(self.source)
        if self.trace.as_fraction() != expected_trace:
            raise _validation_error(
                "trace_mismatch", "trace must equal the sum of the source embeddings"
            )
        if self.norm.as_fraction() != expected_norm:
            raise _validation_error(
                "norm_mismatch", "norm must equal the product of the source embeddings"
            )
        for label, value in (("trace", self.trace), ("norm", self.norm)):
            _require_bounded_rational(
                value,
                max_digits=_MAX_EMBEDDING_PROFILE_RESULT_DIGITS,
                label=f"real-quadratic embedding {label}",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: RealQuadraticValue,
        images: tuple[RealQuadraticEmbeddingImage, RealQuadraticEmbeddingImage],
        trace: CanonicalRational,
        norm: CanonicalRational,
    ) -> Self:
        return cls.model_construct(
            source=source,
            real_embedding_count=2,
            complex_conjugate_pair_count=0,
            images=images,
            trace=trace,
            norm=norm,
            convention="REAL_QUADRATIC_ROOTS_V1",
        )


class RealQuadraticSignCertificate(StrictModel):
    rational_part_squared: CanonicalRational
    radical_part_squared: CanonicalRational
    magnitude_order: Literal["LT", "EQ", "GT"]


class RealQuadraticOrderValue(StrictModel):
    left: RealQuadraticValue
    right: RealQuadraticValue
    difference: RealQuadraticValue
    order: Literal["LT", "EQ", "GT"]
    sign_basis: RealQuadraticSignBasis
    sign_certificate: RealQuadraticSignCertificate

    @model_validator(mode="after")
    def bind_exact_order(self) -> Self:
        if (
            len({self.left.radicand, self.right.radicand, self.difference.radicand})
            != 1
        ):
            raise _validation_error(
                "difference_field_mismatch",
                "left, right, and difference must use one quadratic field",
            )
        expected_rational = (
            self.left.rational_part.as_fraction()
            - self.right.rational_part.as_fraction()
        )
        expected_radical = (
            self.left.radical_coefficient.as_fraction()
            - self.right.radical_coefficient.as_fraction()
        )
        if (
            self.difference.rational_part.as_fraction() != expected_rational
            or self.difference.radical_coefficient.as_fraction() != expected_radical
        ):
            raise _validation_error(
                "difference_mismatch", "difference must equal left minus right"
            )

        rational_square = expected_rational * expected_rational
        radical_square = expected_radical * expected_radical * self.left.radicand
        expected_magnitude_order = _order(rational_square, radical_square)
        if (
            self.sign_certificate.rational_part_squared.as_fraction() != rational_square
            or self.sign_certificate.radical_part_squared.as_fraction()
            != radical_square
            or self.sign_certificate.magnitude_order != expected_magnitude_order
        ):
            raise _validation_error(
                "sign_certificate_mismatch",
                "sign certificate must match the exact quadratic difference",
            )

        expected_basis: RealQuadraticSignBasis = (
            "RATIONAL_ONLY"
            if expected_radical == 0
            else "RADICAL_ONLY"
            if expected_rational == 0
            else "SAME_SIGN"
            if (expected_rational > 0) == (expected_radical > 0)
            else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
        )
        expected_sign = _sign(expected_rational, expected_radical, self.left.radicand)
        expected_order = (
            "LT" if expected_sign < 0 else "GT" if expected_sign > 0 else "EQ"
        )
        if self.sign_basis != expected_basis or self.order != expected_order:
            raise _validation_error(
                "order_mismatch",
                "order and sign basis must match the exact quadratic difference",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: RealQuadraticValue,
        right: RealQuadraticValue,
        difference: RealQuadraticValue,
        order: Literal["LT", "EQ", "GT"],
        sign_basis: RealQuadraticSignBasis,
        sign_certificate: RealQuadraticSignCertificate,
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            difference=difference,
            order=order,
            sign_basis=sign_basis,
            sign_certificate=sign_certificate,
        )


def real_quadratic_order(
    left: RealQuadraticValue,
    right: RealQuadraticValue,
) -> RealQuadraticOrderValue:
    """Compare two canonical values from the same real quadratic field."""

    _require_order_admission(left, right)
    a = left.rational_part.as_fraction() - right.rational_part.as_fraction()
    b = left.radical_coefficient.as_fraction() - right.radical_coefficient.as_fraction()
    d = left.radicand
    sign = _sign(a, b, d)
    basis: RealQuadraticSignBasis = (
        "RATIONAL_ONLY"
        if b == 0
        else "RADICAL_ONLY"
        if a == 0
        else "SAME_SIGN"
        if (a > 0) == (b > 0)
        else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
    )
    rational_square = a * a
    radical_square = b * b * d
    return RealQuadraticOrderValue._from_kernel(
        left=left,
        right=right,
        difference=RealQuadraticValue(
            rational_part=CanonicalRational.from_fraction(a),
            radical_coefficient=CanonicalRational.from_fraction(b),
            radicand=d,
        ),
        order="LT" if sign < 0 else "GT" if sign > 0 else "EQ",
        sign_basis=basis,
        sign_certificate=RealQuadraticSignCertificate(
            rational_part_squared=CanonicalRational.from_fraction(rational_square),
            radical_part_squared=CanonicalRational.from_fraction(radical_square),
            magnitude_order=_order(rational_square, radical_square),
        ),
    )


def real_quadratic_embeddings(
    element: RealQuadraticValue,
) -> RealQuadraticEmbeddingProfile:
    """Return both exact real embeddings, trace, and norm of one element."""

    source = element
    trace, norm = _embedding_scalars(source)
    for label, value in (("trace", trace), ("norm", norm)):
        _require_bounded_rational(
            CanonicalRational.from_fraction(value),
            max_digits=_MAX_EMBEDDING_PROFILE_RESULT_DIGITS,
            label=f"real-quadratic embedding {label}",
        )
    radical_coefficient = source.radical_coefficient.as_fraction()
    return RealQuadraticEmbeddingProfile._from_kernel(
        source=source,
        images=(
            RealQuadraticEmbeddingImage(embedding="POSITIVE_ROOT", value=source),
            RealQuadraticEmbeddingImage(
                embedding="NEGATIVE_ROOT",
                value=RealQuadraticValue(
                    rational_part=source.rational_part,
                    radical_coefficient=CanonicalRational.from_fraction(
                        -radical_coefficient
                    ),
                    radicand=source.radicand,
                ),
            ),
        ),
        trace=CanonicalRational.from_fraction(trace),
        norm=CanonicalRational.from_fraction(norm),
    )


__all__ = [
    "RealQuadraticEmbedding",
    "RealQuadraticEmbeddingConvention",
    "RealQuadraticEmbeddingImage",
    "RealQuadraticEmbeddingProfile",
    "RealQuadraticOrderValue",
    "RealQuadraticSignCertificate",
    "RealQuadraticValue",
    "real_quadratic_embeddings",
    "real_quadratic_order",
]
