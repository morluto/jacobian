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

from math import gcd
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue

MAX_MAHLER_DEGREE = 64
MAX_MAHLER_COEFFICIENT_DIGITS = 256
# Square-free extraction is delegated to the certified factorization kernel,
# whose request carrier admits at most 30 decimal digits. Keep both the
# decimal contract and a coarse bit envelope on the value itself so forged
# surd JSON cannot reach ``isqrt`` outside the kernel's admitted domain.
MAX_MAHLER_RADICAND_DIGITS = 30
MAX_MAHLER_RADICAND_BITS = 100

RootLocation = Literal[
    "INSIDE_UNIT_DISK", "ON_UNIT_CIRCLE", "OUTSIDE_UNIT_DISK", "UNRESOLVED"
]


MahlerAlgebraicValue = CanonicalRational | RealAlgebraicValue


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
        if self.content < 1:
            raise _validation_error(
                "polynomial.mahler_content_positive",
                "content must be the positive coefficient gcd",
            )
        primitive_content = 0
        for coefficient in self.primitive_part.coefficients_descending:
            primitive_content = gcd(primitive_content, abs(coefficient))
        if primitive_content != 1:
            raise _validation_error(
                "polynomial.mahler_primitive_content",
                "the primitive part must have coefficient gcd one",
            )
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
        if (self.reconstruction.coefficients_descending[0] > 0) != (self.sign == 1):
            raise _validation_error(
                "polynomial.mahler_content_sign",
                "the sign must match the reconstructed leading coefficient",
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
        if any(
            abs(coefficient) >= 10**MAX_MAHLER_COEFFICIENT_DIGITS
            for coefficient in self.coefficients_descending
        ):
            raise _validation_error(
                "polynomial.mahler_coefficient_bound",
                "coefficients exceed the 256-digit bound",
            )
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
    sum_of_roots: CanonicalRational
    product_of_roots: CanonicalRational
    roots: tuple[MahlerAlgebraicValue, ...]
    complex_pair_squared_modulus: CanonicalRational | None = None
    root_locations: tuple[RootLocation, ...]

    @model_validator(mode="after")
    def require_ledger_length(self) -> Self:
        if self.coefficients_descending[0] == 0:
            raise _validation_error(
                "polynomial.mahler_quadratic_leading",
                "a root profile needs a nonzero leading coefficient",
            )
        if any(location == "UNRESOLVED" for location in self.root_locations):
            raise _validation_error(
                "polynomial.mahler_quadratic_unresolved_location",
                "a root profile result must resolve every root location",
            )
        expected = 2 if self.root_kind == "DISTINCT_REAL" else 1
        if self.root_kind == "COMPLEX_CONJUGATE":
            expected = 1
        if len(self.root_locations) != expected:
            raise _validation_error(
                "polynomial.mahler_quadratic_location_count",
                "the location ledger covers each distinct root of the quadratic",
            )
        expected_roots = 0 if self.root_kind == "COMPLEX_CONJUGATE" else expected
        if self.root_kind == "COMPLEX_CONJUGATE":
            if self.complex_pair_squared_modulus is None:
                raise _validation_error(
                    "polynomial.mahler_quadratic_modulus",
                    "complex pair must retain its squared-modulus carrier",
                )
        elif self.complex_pair_squared_modulus is not None:
            raise _validation_error(
                "polynomial.mahler_quadratic_modulus",
                "real roots must not carry a complex-pair modulus",
            )
        if len(self.roots) != expected_roots:
            raise _validation_error(
                "polynomial.mahler_quadratic_root_count",
                "the root ledger must contain exactly the distinct real roots",
            )
        return self


class MahlerMeasureRequest(StrictModel):
    """One bounded integer polynomial of degree zero, one, or two."""

    coefficients_descending: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=3
    )

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if any(
            abs(coefficient) >= 10**MAX_MAHLER_COEFFICIENT_DIGITS
            for coefficient in self.coefficients_descending
        ):
            raise _validation_error(
                "polynomial.mahler_coefficient_bound",
                "coefficients exceed the 256-digit bound",
            )
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
    degree: StrictInt = Field(ge=0, le=2)
    leading_coefficient: ExactInteger
    root_locations: tuple[RootLocation, ...]
    outside_root_product: MahlerAlgebraicValue
    mahler_measure: MahlerAlgebraicValue
    convention: Literal["ABSOLUTE_LEADING_TIMES_OUTSIDE_ROOT_PRODUCT"] = (
        "ABSOLUTE_LEADING_TIMES_OUTSIDE_ROOT_PRODUCT"
    )

    @model_validator(mode="after")
    def require_bounded_root_ledger(self) -> Self:
        if len(self.coefficients_descending) != self.degree + 1:
            raise _validation_error(
                "polynomial.mahler_result_degree",
                "the coefficient tuple length must equal degree+1",
            )
        if self.coefficients_descending[0] == 0:
            raise _validation_error(
                "polynomial.mahler_result_leading",
                "the result polynomial must have a nonzero leading coefficient",
            )
        if self.leading_coefficient != self.coefficients_descending[0]:
            raise _validation_error(
                "polynomial.mahler_result_leading_binding",
                "the retained leading coefficient must match the source polynomial",
            )
        if any(location == "UNRESOLVED" for location in self.root_locations):
            raise _validation_error(
                "polynomial.mahler_result_unresolved_location",
                "a Mahler-measure result must resolve every root location",
            )
        expected = 0 if self.degree == 0 else 1 if self.degree == 1 else 2
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
    "MAX_MAHLER_RADICAND_BITS",
    "MAX_MAHLER_RADICAND_DIGITS",
    "ContentPrimitiveProfileRequest",
    "ContentPrimitiveProfileResult",
    "IntegerPolynomialProfileValue",
    "IntegerPolynomialValue",
    "MahlerAlgebraicValue",
    "MahlerMeasureRequest",
    "MahlerMeasureResult",
    "RealQuadraticRootProfileRequest",
    "RealQuadraticRootProfileResult",
    "ReciprocalProfileRequest",
    "ReciprocalProfileResult",
    "RootLocation",
]
