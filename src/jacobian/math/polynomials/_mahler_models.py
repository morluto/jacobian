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

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

MAX_MAHLER_DEGREE = 64
MAX_MAHLER_COEFFICIENT_DIGITS = 256

RootLocation = Literal[
    "INSIDE_UNIT_DISK", "ON_UNIT_CIRCLE", "OUTSIDE_UNIT_DISK", "UNRESOLVED"
]


MahlerAlgebraicValue = CanonicalRational | RealAlgebraicValue


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _require_mahler_polynomial_envelope(polynomial: IntegerPolynomial) -> None:
    """Enforce the structural Mahler/quadratic degree envelope."""

    if len(polynomial.coefficients) > MAX_MAHLER_DEGREE + 1:
        raise _validation_error(
            "polynomial.mahler_degree_bound",
            f"a profile polynomial has degree at most {MAX_MAHLER_DEGREE}",
        )


class ReciprocalProfileRequest(StrictModel):
    polynomial: IntegerPolynomial


class ReciprocalProfileResult(StrictModel):
    degree: StrictInt = Field(ge=0, le=MAX_POLYNOMIAL_TERMS - 1)
    reversed_coefficients: tuple[ExactInteger, ...]
    state: Literal["RECIPROCAL", "ANTIRECIPROCAL", "NEITHER"]
    leading_coefficient: ExactInteger
    constant_coefficient: ExactInteger
    coefficient_pair_ledger: tuple[tuple[ExactInteger, ExactInteger], ...]

    @model_validator(mode="after")
    def require_structural_ledger(self) -> Self:
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        degree: int,
        reversed_coefficients: tuple[ExactInteger, ...],
        state: Literal["RECIPROCAL", "ANTIRECIPROCAL", "NEITHER"],
        leading_coefficient: ExactInteger,
        constant_coefficient: ExactInteger,
        coefficient_pair_ledger: tuple[tuple[ExactInteger, ExactInteger], ...],
    ) -> Self:
        return cls.model_construct(
            degree=degree,
            reversed_coefficients=reversed_coefficients,
            state=state,
            leading_coefficient=leading_coefficient,
            constant_coefficient=constant_coefficient,
            coefficient_pair_ledger=coefficient_pair_ledger,
        )


class RealQuadraticRootProfileRequest(StrictModel):
    """One real quadratic ``a x^2 + b x + c`` with nonzero leading coefficient."""

    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_quadratic_envelope(self) -> Self:
        _require_mahler_polynomial_envelope(self.polynomial)
        if len(self.polynomial.coefficients) != 3:
            raise _validation_error(
                "polynomial.mahler_quadratic_degree",
                "a real quadratic profile needs exactly three coefficients",
            )
        return self


class RealQuadraticRootProfileResult(StrictModel):
    polynomial: IntegerPolynomial
    discriminant: ExactInteger
    root_kind: Literal["DISTINCT_REAL", "DOUBLE_REAL", "COMPLEX_CONJUGATE"]
    sum_of_roots: CanonicalRational
    product_of_roots: CanonicalRational
    roots: tuple[MahlerAlgebraicValue, ...]
    complex_pair_squared_modulus: CanonicalRational | None = None
    root_locations: tuple[RootLocation, ...]

    @model_validator(mode="after")
    def require_ledger_length(self) -> Self:
        _require_mahler_polynomial_envelope(self.polynomial)
        if len(self.polynomial.coefficients) != 3:
            raise _validation_error(
                "polynomial.mahler_quadratic_degree",
                "a root profile needs exactly three coefficients",
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

    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_low_degree_envelope(self) -> Self:
        _require_mahler_polynomial_envelope(self.polynomial)
        if len(self.polynomial.coefficients) > 3:
            raise _validation_error(
                "polynomial.mahler_degree_bound",
                "the Mahler measure needs degree at most two",
            )
        if self.polynomial.coefficients[0] == 0:
            raise _validation_error(
                "polynomial.mahler_leading_nonzero",
                "the Mahler measure needs a nonzero leading coefficient",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.polynomial.coefficients) - 1


class MahlerMeasureResult(StrictModel):
    """The exact Mahler measure ``|a_d| * prod_i max(1, |alpha_i|)``."""

    polynomial: IntegerPolynomial
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
        _require_mahler_polynomial_envelope(self.polynomial)
        if len(self.polynomial.coefficients) != self.degree + 1:
            raise _validation_error(
                "polynomial.mahler_result_degree",
                "the coefficient tuple length must equal degree+1",
            )
        if self.polynomial.coefficients[0] == 0:
            raise _validation_error(
                "polynomial.mahler_result_leading",
                "the result polynomial must have a nonzero leading coefficient",
            )
        if self.leading_coefficient != self.polynomial.coefficients[0]:
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
        coefficients = self.polynomial.coefficients
        if self.degree == 2 and coefficients[1] == 0 and coefficients[2] != 0:
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
    "MahlerAlgebraicValue",
    "MahlerMeasureRequest",
    "MahlerMeasureResult",
    "RealQuadraticRootProfileRequest",
    "RealQuadraticRootProfileResult",
    "ReciprocalProfileRequest",
    "ReciprocalProfileResult",
    "RootLocation",
]
