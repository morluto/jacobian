"""Exact order in one real quadratic field."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

_MAX_RADICAND = 1_000_000
_MAX_DIGITS = 256
# For a,b with numerator and denominator at most 256 decimal digits,
# a^2 - d*b^2 has a denominator of at most 1,024 digits and a numerator
# of at most 1,032 digits after bringing the two terms to that denominator
# (d <= 10^6).  The trace is smaller.  This covers producer and replay.
_MAX_EMBEDDING_PROFILE_RESULT_DIGITS = 1_032
RealQuadraticSignBasis = Literal[
    "RATIONAL_ONLY",
    "RADICAL_ONLY",
    "SAME_SIGN",
    "OPPOSING_SIGNS_SQUARED_MAGNITUDES",
]
RealQuadraticEmbedding = Literal["POSITIVE_ROOT", "NEGATIVE_ROOT"]
RealQuadraticEmbeddingConvention = Literal["REAL_QUADRATIC_ROOTS_V1"]


def _is_square_free(value: int) -> bool:
    return all(value % (divisor * divisor) for divisor in range(2, isqrt(value) + 1))


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
        require_bounded_rational(
            self.rational_part, max_digits=_MAX_DIGITS, label="rational part"
        )
        require_bounded_rational(
            self.radical_coefficient,
            max_digits=_MAX_DIGITS,
            label="radical coefficient",
        )
        if not _is_square_free(self.radicand):
            raise ValueError("real-quadratic radicand must be square-free")
        return self


class RealQuadraticOrderRequest(StrictModel):
    left: RealQuadraticValue
    right: RealQuadraticValue

    @model_validator(mode="after")
    def require_shared_field(self) -> Self:
        if self.left.radicand != self.right.radicand:
            raise ValueError("comparison requires one shared radicand")
        return self


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


class RealQuadraticEmbeddingsRequest(StrictModel):
    """One bounded element whose two real embeddings are requested."""

    element: RealQuadraticValue = Field(
        description=(
            "The exact element a + b*sqrt(d) in a real quadratic field. "
            "Its square-free radicand selects the field and its rational "
            "components are bounded to 256 decimal digits."
        ),
    )

    @model_validator(mode="after")
    def require_profile_within_result_bound(self) -> Self:
        trace, norm = _embedding_scalars(self.element)
        for label, value in (("trace", trace), ("norm", norm)):
            require_bounded_rational(
                CanonicalRational.from_fraction(value),
                max_digits=_MAX_EMBEDDING_PROFILE_RESULT_DIGITS,
                label=f"real-quadratic embedding {label}",
            )
        return self


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
        radical_coefficient = self.source.radical_coefficient.as_fraction()
        conjugate = RealQuadraticValue(
            rational_part=self.source.rational_part,
            radical_coefficient=CanonicalRational.from_fraction(-radical_coefficient),
            radicand=self.source.radicand,
        )
        expected_images = (
            RealQuadraticEmbeddingImage(embedding="POSITIVE_ROOT", value=self.source),
            RealQuadraticEmbeddingImage(embedding="NEGATIVE_ROOT", value=conjugate),
        )
        if self.images != expected_images:
            raise ValueError(
                "images must be the ordered positive-root and negative-root "
                "embeddings of the retained source"
            )
        expected_trace, expected_norm = _embedding_scalars(self.source)
        expected_trace_value = CanonicalRational.from_fraction(expected_trace)
        expected_norm_value = CanonicalRational.from_fraction(expected_norm)
        for label, value in (
            ("trace", expected_trace_value),
            ("norm", expected_norm_value),
        ):
            require_bounded_rational(
                value,
                max_digits=_MAX_EMBEDDING_PROFILE_RESULT_DIGITS,
                label=f"real-quadratic embedding {label}",
            )
        if self.trace != expected_trace_value or self.norm != expected_norm_value:
            raise ValueError(
                "trace and norm must be the exact trace and norm of the retained source"
            )
        return self


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
        a = (
            self.left.rational_part.as_fraction()
            - self.right.rational_part.as_fraction()
        )
        b = (
            self.left.radical_coefficient.as_fraction()
            - self.right.radical_coefficient.as_fraction()
        )
        if (
            self.difference.radicand != self.left.radicand
            or self.difference.rational_part.as_fraction() != a
            or self.difference.radical_coefficient.as_fraction() != b
        ):
            raise ValueError("difference must equal left minus right")
        expected_order = (
            "LT"
            if _sign(a, b, self.left.radicand) < 0
            else "GT"
            if _sign(a, b, self.left.radicand) > 0
            else "EQ"
        )
        if self.order != expected_order:
            raise ValueError("order must match exact quadratic sign")
        expected_basis: RealQuadraticSignBasis = (
            "RATIONAL_ONLY"
            if b == 0
            else "RADICAL_ONLY"
            if a == 0
            else "SAME_SIGN"
            if (a > 0) == (b > 0)
            else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
        )
        if self.sign_basis != expected_basis:
            raise ValueError("sign basis does not match difference structure")
        rational_square = a * a
        radical_square = b * b * self.left.radicand
        if (
            self.sign_certificate.rational_part_squared.as_fraction() != rational_square
            or self.sign_certificate.radical_part_squared.as_fraction()
            != radical_square
            or self.sign_certificate.magnitude_order
            != _order(rational_square, radical_square)
        ):
            raise ValueError("sign certificate does not match squared magnitudes")
        return self


def real_quadratic_order(
    request: RealQuadraticOrderRequest,
) -> RealQuadraticOrderValue:
    a = (
        request.left.rational_part.as_fraction()
        - request.right.rational_part.as_fraction()
    )
    b = (
        request.left.radical_coefficient.as_fraction()
        - request.right.radical_coefficient.as_fraction()
    )
    d = request.left.radicand
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
    return RealQuadraticOrderValue(
        left=request.left,
        right=request.right,
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
    radical_coefficient = source.radical_coefficient.as_fraction()
    trace, norm = _embedding_scalars(source)
    return RealQuadraticEmbeddingProfile(
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
    "RealQuadraticEmbeddingsRequest",
    "RealQuadraticOrderRequest",
    "RealQuadraticOrderValue",
    "RealQuadraticSignCertificate",
    "RealQuadraticValue",
    "real_quadratic_embeddings",
    "real_quadratic_order",
]
