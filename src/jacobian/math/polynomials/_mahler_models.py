"""Exact integer-polynomial structural profiles and the Mahler measure (#1787).

The Mahler-measure family here keeps the leading-coefficient boundary explicit:
the content/primitive profile exposes sign, content, and primitive part, the
reciprocal profile exposes the reversal structure, and the real-quadratic root
profile classifies each root's location relative to the closed unit disk.  The
measure operation then consumes a complete root-location ledger and multiplies
the outside-root contributions by the leading coefficient, which is exactly the
step the audited partial formula dropped.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

MAX_MAHLER_DEGREE = 64
MAX_MAHLER_COEFFICIENT_DIGITS = 256

RootLocation = Literal[
    "INSIDE_UNIT_DISK", "ON_UNIT_CIRCLE", "OUTSIDE_UNIT_DISK", "UNRESOLVED"
]


class QuadraticSurd(StrictModel):
    """An exact element ``a + b*sqrt(d)`` with ``d`` squarefree or zero.

    ``d = 0`` denotes a rational value and requires ``b = 0``; otherwise the
    radical coefficient is nonzero.  The representation is canonical, so two
    equal surd values have byte-identical serializations.
    """

    rational_part: CanonicalRational
    radical_coefficient: CanonicalRational
    radicand: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical_surd(self) -> Self:
        if self.radicand == 0:
            if self.radical_coefficient.as_fraction() != 0:
                raise _validation_error(
                    "polynomial.mahler_surd_radical_with_zero_radicand",
                    "a rational surd must have zero radical coefficient",
                )
            return self
        root = isqrt(self.radicand)
        if root * root == self.radicand:
            raise _validation_error(
                "polynomial.mahler_surd_square_radicand",
                "a quadratic-surd radicand must not be a perfect square",
            )
        if self.radical_coefficient.as_fraction() == 0:
            raise _validation_error(
                "polynomial.mahler_surd_zero_radical",
                "a nonrational surd must have a nonzero radical coefficient",
            )
        return self

    def as_fractions(self) -> tuple[Fraction, Fraction]:
        return (
            self.rational_part.as_fraction(),
            self.radical_coefficient.as_fraction(),
        )

    @classmethod
    def rational(cls, value: Fraction) -> QuadraticSurd:
        return cls(
            rational_part=CanonicalRational(num=value.numerator, den=value.denominator),
            radical_coefficient=CanonicalRational(num=0, den=1),
            radicand=0,
        )

    @classmethod
    def from_fractions(
        cls, rational: Fraction, radical: Fraction, radicand: int
    ) -> QuadraticSurd:
        """Canonicalize ``rational + radical*sqrt(radicand)``.

        Square factors are pulled out of the radicand so the published value is
        unique: ``sqrt(20)`` becomes ``2*sqrt(5)``.
        """

        if radicand == 0 or radical == 0:
            return cls.rational(rational)
        square = 1
        remaining = radicand
        factor = 2
        while factor * factor <= remaining:
            while remaining % (factor * factor) == 0:
                remaining //= factor * factor
                square *= factor
            factor += 1
        if remaining == 1:
            return cls.rational(rational)
        return cls(
            rational_part=CanonicalRational(
                num=rational.numerator, den=rational.denominator
            ),
            radical_coefficient=CanonicalRational(
                num=(radical * square).numerator, den=(radical * square).denominator
            ),
            radicand=remaining,
        )

    def multiply(self, other: QuadraticSurd) -> QuadraticSurd:
        """Exact product; the radicands either agree, or one side is rational."""

        left_a, left_b = self.as_fractions()
        right_a, right_b = other.as_fractions()
        if self.radicand == 0 or other.radicand == 0:
            radicand = self.radicand or other.radicand
            return QuadraticSurd.from_fractions(
                left_a * right_a, left_a * right_b + left_b * right_a, radicand
            )
        if self.radicand != other.radicand:
            raise _validation_error(
                "polynomial.mahler_surd_mixed_radicands",
                "combining surds with different radicands is outside this envelope",
            )
        radicand = self.radicand
        return QuadraticSurd.from_fractions(
            left_a * right_a + left_b * right_b * radicand,
            left_a * right_b + right_a * left_b,
            radicand,
        )

    def __pow__(self, exponent: int) -> QuadraticSurd:
        result = QuadraticSurd.rational(Fraction(1))
        for _ in range(exponent):
            result = result.multiply(self)
        return result

    def is_zero(self) -> bool:
        """Exact test of ``a + b*sqrt(d) == 0``."""

        a, b = self.as_fractions()
        return a == 0 and (b == 0 or self.radicand == 0)

    def is_one(self) -> bool:
        """Exact test of ``a + b*sqrt(d) == 1``."""

        a, b = self.as_fractions()
        return a == 1 and (b == 0 or self.radicand == 0)

    def is_nonnegative(self) -> bool:
        """Exact test of ``a + b*sqrt(d) >= 0``."""

        a, b = self.as_fractions()
        if b == 0:
            return a >= 0
        if self.radicand == 0:
            return a >= 0
        # Compare a and -b*sqrt(d) exactly by squaring with sign analysis.
        if a >= 0 and b >= 0:
            return True
        if a < 0 and b < 0:
            return False
        if a >= 0:
            return a * a >= b * b * self.radicand
        return b * b * self.radicand >= a * a


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class IntegerPolynomialValue(StrictModel):
    """One canonical nonzero integer polynomial, retaining its leading sign."""

    coefficients_descending: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_MAHLER_DEGREE + 1
    )

    @model_validator(mode="after")
    def require_nonzero_leading_and_bounded_digits(self) -> Self:
        if self.coefficients_descending[0] == 0:
            raise _validation_error(
                "polynomial.mahler_zero_leading",
                "an integer polynomial cannot have a zero leading coefficient",
            )
        if any(
            len(format_canonical_integer(abs(coefficient)))
            > MAX_MAHLER_COEFFICIENT_DIGITS
            for coefficient in self.coefficients_descending
        ):
            raise _validation_error(
                "polynomial.mahler_coefficient_bound",
                "a profile polynomial coefficient exceeds the admitted digit bound",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients_descending) - 1


class IntegerPolynomialProfileValue(IntegerPolynomialValue):
    """One canonical integer polynomial with a positive leading coefficient.

    This is the carrier for primitive factors and keeps their normalization
    explicit. Requests use :class:`IntegerPolynomialValue` so content/sign
    operations can represent arbitrary source polynomials.
    """

    coefficients_descending: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_MAHLER_DEGREE + 1
    )

    @model_validator(mode="after")
    def require_positive_leading_and_bounded_digits(self) -> Self:
        if self.coefficients_descending[0] <= 0:
            raise _validation_error(
                "polynomial.mahler_positive_leading_required",
                "a profile polynomial must have a positive leading coefficient",
            )
        if any(
            len(format_canonical_integer(abs(coefficient)))
            > MAX_MAHLER_COEFFICIENT_DIGITS
            for coefficient in self.coefficients_descending
        ):
            raise _validation_error(
                "polynomial.mahler_coefficient_bound",
                "a profile polynomial coefficient exceeds the admitted digit bound",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients_descending) - 1


class ContentPrimitiveProfileRequest(StrictModel):
    polynomial: IntegerPolynomialValue


class ContentPrimitiveProfileResult(StrictModel):
    sign: Literal[-1, 1]
    content: ExactInteger
    primitive_part: IntegerPolynomialProfileValue
    degree: StrictInt = Field(ge=0, le=MAX_MAHLER_DEGREE)
    reconstruction: IntegerPolynomialValue

    @model_validator(mode="after")
    def require_exact_reconstruction(self) -> Self:
        scaled = tuple(
            self.sign * self.content * coefficient
            for coefficient in self.primitive_part.coefficients_descending
        )
        if scaled != self.reconstruction.coefficients_descending:
            raise _validation_error(
                "polynomial.mahler_content_reconstruction",
                "sign*content*primitive_part must reconstruct the input polynomial",
            )
        if self.degree != self.primitive_part.degree:
            raise _validation_error(
                "polynomial.mahler_content_degree",
                "the reported degree is the primitive part's degree",
            )
        return self


class ReciprocalProfileRequest(StrictModel):
    polynomial: IntegerPolynomialValue


class ReciprocalProfileResult(StrictModel):
    degree: StrictInt = Field(ge=0, le=MAX_MAHLER_DEGREE)
    reversed_coefficients: tuple[ExactInteger, ...]
    state: Literal["RECIPROCAL", "ANTIRECIPROCAL", "NEITHER"]
    leading_coefficient: ExactInteger
    constant_coefficient: ExactInteger
    coefficient_pair_ledger: tuple[tuple[ExactInteger, ExactInteger], ...]

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        reconstructed = tuple(reversed(self.reversed_coefficients))
        if len(reconstructed) != self.degree + 1:
            raise _validation_error(
                "polynomial.mahler_reciprocal_length",
                "the reversed coefficient tuple must have degree+1 entries",
            )
        if self.leading_coefficient != self.reversed_coefficients[-1]:
            raise _validation_error(
                "polynomial.mahler_reciprocal_leading",
                "the leading coefficient is the reversed tuple's last entry",
            )
        if self.constant_coefficient != self.reversed_coefficients[0]:
            raise _validation_error(
                "polynomial.mahler_reciprocal_constant",
                "the constant coefficient is the reversed tuple's first entry",
            )
        if len(self.coefficient_pair_ledger) != (self.degree + 2) // 2:
            raise _validation_error(
                "polynomial.mahler_reciprocal_pair_count",
                "the pair ledger covers each symmetric coefficient pair once",
            )
        expected_pairs = tuple(
            (reconstructed[index], reconstructed[self.degree - index])
            for index in range((self.degree + 2) // 2)
        )
        if self.coefficient_pair_ledger != expected_pairs:
            raise _validation_error(
                "polynomial.mahler_reciprocal_pairs",
                "the coefficient-pair ledger must match the source coefficients",
            )
        reciprocal = all(
            coefficient == reconstructed[self.degree - index]
            for index, coefficient in enumerate(reconstructed)
        )
        antireciprocal = all(
            coefficient == -reconstructed[self.degree - index]
            for index, coefficient in enumerate(reconstructed)
        )
        expected_state = (
            "RECIPROCAL"
            if reciprocal
            else "ANTIRECIPROCAL"
            if antireciprocal
            else "NEITHER"
        )
        if self.state != expected_state:
            raise _validation_error(
                "polynomial.mahler_reciprocal_state",
                "the reciprocal state must match every coefficient pair",
            )
        return self


class RealQuadraticRootProfileRequest(StrictModel):
    """One real quadratic ``a x^2 + b x + c`` with nonzero leading coefficient."""

    coefficients_descending: tuple[ExactInteger, ExactInteger, ExactInteger]

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if self.coefficients_descending[0] == 0:
            raise _validation_error(
                "polynomial.mahler_quadratic_leading",
                "a real quadratic profile needs a nonzero leading coefficient",
            )
        return self


class RealQuadraticRootProfileResult(StrictModel):
    coefficients_descending: tuple[ExactInteger, ExactInteger, ExactInteger]
    discriminant: ExactInteger
    root_kind: Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"]
    sum_of_roots: tuple[ExactInteger, ExactInteger]
    product_of_roots: tuple[ExactInteger, ExactInteger]
    roots: tuple[QuadraticSurd, ...]
    root_locations: tuple[RootLocation, ...]

    @model_validator(mode="after")
    def require_ledger_length(self) -> Self:
        expected = 2 if self.root_kind == "DISTINCT_REAL" else 1
        if self.root_kind == "COMPLEX_CONJUGATE":
            expected = 1
        if len(self.root_locations) != expected:
            raise _validation_error(
                "polynomial.mahler_quadratic_location_count",
                "the location ledger covers each distinct root of the quadratic",
            )
        if len(self.roots) != len(self.root_locations):
            raise _validation_error(
                "polynomial.mahler_quadratic_root_count",
                "the root and location ledgers must have equal length",
            )
        return self


class MahlerMeasureRequest(StrictModel):
    """One bounded integer polynomial of degree at most two."""

    coefficients_descending: tuple[ExactInteger, ...] = Field(
        min_length=2, max_length=3
    )

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if self.coefficients_descending[0] == 0:
            raise _validation_error(
                "polynomial.mahler_leading_nonzero",
                "the Mahler measure needs a nonzero leading coefficient",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients_descending) - 1


class MahlerMeasureResult(StrictModel):
    """The exact Mahler measure ``|a_d| * prod_i max(1, |alpha_i|)``."""

    coefficients_descending: tuple[ExactInteger, ...]
    degree: StrictInt = Field(ge=1, le=2)
    leading_coefficient: ExactInteger
    root_locations: tuple[RootLocation, ...]
    outside_root_product: QuadraticSurd
    mahler_measure: QuadraticSurd
    convention: Literal["ABSOLUTE_LEADING_TIMES_OUTSIDE_ROOT_PRODUCT"] = (
        "ABSOLUTE_LEADING_TIMES_OUTSIDE_ROOT_PRODUCT"
    )

    @model_validator(mode="after")
    def require_bounded_root_ledger(self) -> Self:
        expected = 1 if self.degree == 1 else 2
        if (
            self.degree == 2
            and self.coefficients_descending[1] == 0
            and self.coefficients_descending[2] != 0
        ):
            # Pure quadratic a x^2 + c has two real or two conjugate roots.
            expected = 2
        if len(self.root_locations) != expected:
            raise _validation_error(
                "polynomial.mahler_root_ledger_length",
                "the root-location ledger must cover every root of the polynomial",
            )
        return self


__all__ = [
    "MAX_MAHLER_COEFFICIENT_DIGITS",
    "MAX_MAHLER_DEGREE",
    "ContentPrimitiveProfileRequest",
    "ContentPrimitiveProfileResult",
    "IntegerPolynomialProfileValue",
    "IntegerPolynomialValue",
    "MahlerMeasureRequest",
    "MahlerMeasureResult",
    "QuadraticSurd",
    "RealQuadraticRootProfileRequest",
    "RealQuadraticRootProfileResult",
    "ReciprocalProfileRequest",
    "ReciprocalProfileResult",
    "RootLocation",
]
