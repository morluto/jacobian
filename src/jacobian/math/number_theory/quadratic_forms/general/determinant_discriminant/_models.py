"""Typed contracts for exact rational quadratic-form determinants."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)

MAX_POLAR_DETERMINANT_AXIS = 64
MAX_POLAR_DETERMINANT_MATRIX_ENTRIES = 4096
MAX_POLAR_DETERMINANT_SUPPORT_TERMS = 4096
MAX_POLAR_DETERMINANT_WORK = 100_000_000
MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS = 65_536
MAX_POLAR_DETERMINANT_OUTPUT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS = 1_200_000

DeterminantDiscriminantConvention = Literal["SIGNED_FULL_POLAR_GRAM_V1"]


class DeterminantDiscriminantRequest(StrictModel):
    """Compute the determinant and signed discriminant of one rational form."""

    form: RationalQuadraticForm = Field(
        description=(
            "Rational quadratic form in polynomial-coefficient convention; "
            "dimension, support, Bareiss work, exact intermediate growth, and "
            "retained result size are admitted by the operation."
        )
    )


class DeterminantDiscriminantResult(StrictModel):
    """Exact determinant data for the full polar Gram matrix of one form.

    For ``Q(x)=sum a_i*x_i^2 + sum c_ij*x_i*x_j``, the full polar pairing
    ``B_Q(x,y)=Q(x+y)-Q(x)-Q(y)`` has Gram entries ``G_ii=2*a_i`` and
    ``G_ij=c_ij``. ``polar_gram_determinant`` is ``det(G)`` and
    ``signed_discriminant`` is ``(-1)^(n(n-1)/2)*det(G)``. The signed
    scalar is a rational representative; only its square class is invariant
    under invertible rational basis change, and that class is defined only
    when the form is nondegenerate.
    """

    form: RationalQuadraticForm
    polar_gram_determinant: CanonicalRational
    signed_discriminant: CanonicalRational
    convention: DeterminantDiscriminantConvention = "SIGNED_FULL_POLAR_GRAM_V1"

    @model_validator(mode="after")
    def require_bounded_scalars(self) -> Self:
        n = len(self.form.axis)
        if n > MAX_POLAR_DETERMINANT_AXIS:
            raise ValueError("quadratic-form determinant source exceeds its axis bound")
        if n + len(self.form.cross_terms) > MAX_POLAR_DETERMINANT_SUPPORT_TERMS:
            raise ValueError(
                "quadratic-form determinant source exceeds its support bound"
            )
        source_digits = 64 * n
        source_digits += sum(
            len(str(abs(value.num))) + len(str(value.den))
            for value in self.form.diagonal_coefficients
        )
        source_digits += sum(
            len(str(abs(term.coefficient.num))) + len(str(term.coefficient.den))
            for term in self.form.cross_terms
        )
        if source_digits > MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS:
            raise ValueError(
                "quadratic-form determinant source exceeds its result envelope"
            )
        for value in (self.polar_gram_determinant, self.signed_discriminant):
            require_bounded_rational(
                value,
                max_digits=MAX_POLAR_DETERMINANT_OUTPUT_DIGITS,
                label="quadratic-form determinant result",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: DeterminantDiscriminantRequest,
        *,
        determinant: int,
        denominator: int,
    ) -> Self:
        from fractions import Fraction

        value = Fraction(determinant, denominator)
        n = len(request.form.axis)
        signed = value if (n * (n - 1) // 2) % 2 == 0 else -value
        return cls.model_construct(
            form=request.form,
            polar_gram_determinant=CanonicalRational.from_fraction(value),
            signed_discriminant=CanonicalRational.from_fraction(signed),
            convention="SIGNED_FULL_POLAR_GRAM_V1",
        )


__all__ = [
    "MAX_POLAR_DETERMINANT_AXIS",
    "MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS",
    "MAX_POLAR_DETERMINANT_MATRIX_ENTRIES",
    "MAX_POLAR_DETERMINANT_OUTPUT_DIGITS",
    "MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS",
    "MAX_POLAR_DETERMINANT_SUPPORT_TERMS",
    "MAX_POLAR_DETERMINANT_WORK",
    "DeterminantDiscriminantConvention",
    "DeterminantDiscriminantRequest",
    "DeterminantDiscriminantResult",
]
