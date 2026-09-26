"""Canonical structural contracts for proper hypergeometric summands.

The carrier models the classical bivariate form

    P(n, k) * a**n * b**k * product((A*n + B*k + C)!)**power.

Positive powers are numerator factorials and define where the term is
well-defined. Negative powers are reciprocal factorials, with reciprocal
factorial set to zero at negative integer arguments; these factors determine
the support polyhedron.  The exponent convention and the two regions are
deliberately retained because creative telescoping depends on both.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial

MAX_PROPER_HYPERGEOMETRIC_FACTORS = 32
MAX_PROPER_HYPERGEOMETRIC_AFFINE_COEFFICIENT = 128
MAX_PROPER_HYPERGEOMETRIC_FACTOR_POWER = 32
_VARIABLES = ("n", "k")


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"ore_algebra.{code}", message)


class IntegerAffineFactorial(StrictModel):
    """A signed multiplicity of one integer-affine factorial factor."""

    n_coefficient: StrictInt = Field(
        ge=-MAX_PROPER_HYPERGEOMETRIC_AFFINE_COEFFICIENT,
        le=MAX_PROPER_HYPERGEOMETRIC_AFFINE_COEFFICIENT,
    )
    k_coefficient: StrictInt = Field(
        ge=-MAX_PROPER_HYPERGEOMETRIC_AFFINE_COEFFICIENT,
        le=MAX_PROPER_HYPERGEOMETRIC_AFFINE_COEFFICIENT,
    )
    # Offsets change the affine argument's location, not carrier/work size.
    # Keep their wire representation intrinsically bounded without a numeric cap:
    # the shared exact-integer codec carries an arbitrary-size offset losslessly
    # as a canonical decimal string when it exceeds the interoperable JSON range.
    offset: ExactInteger = Field(
        description=(
            "Exact arbitrary-size integer offset of the affine factorial argument; "
            "a native Python int and a canonical decimal string in JSON."
        )
    )
    power: StrictInt = Field(
        ge=-MAX_PROPER_HYPERGEOMETRIC_FACTOR_POWER,
        le=MAX_PROPER_HYPERGEOMETRIC_FACTOR_POWER,
        description=(
            "Positive powers place this factorial in the numerator; negative "
            "powers place its reciprocal in the denominator."
        ),
    )

    @model_validator(mode="after")
    def require_nonconstant_affine_argument_and_nonzero_power(self) -> Self:
        if self.n_coefficient == 0 and self.k_coefficient == 0:
            raise _error(
                "proper_hypergeometric_constant_factor",
                "constant factorial factors belong in the polynomial prefactor",
            )
        if self.power == 0:
            raise _error(
                "proper_hypergeometric_zero_power",
                "factorial factor powers must be nonzero",
            )
        return self

    @property
    def affine_key(self) -> tuple[int, int, int]:
        """Return the exact integral affine form used for canonical ordering."""
        return (self.n_coefficient, self.k_coefficient, self.offset)


class ProperHypergeometricTerm(StrictModel):
    """One exact bivariate proper hypergeometric term over ``QQ``.

    Its pointwise expression is ``P(n,k) * a**n * b**k`` times the declared
    signed factorial powers.  The polynomial has fixed ordered axes ``(n,k)``.
    Factorial factors are sorted by their integer-affine argument and each
    argument occurs once with its net signed multiplicity.

    A positive factorial power requires its affine argument to be nonnegative;
    this is the term's well-defined region. A negative power means a reciprocal
    factorial, extended by zero at negative integer arguments; these
    inequalities define the term's support. The two regions need not coincide.
    Summation bounds remain explicit inputs to a later summation operation.
    """

    polynomial: RationalPolynomial = Field(
        description="Polynomial prefactor in QQ[n,k], with variable order (n,k)."
    )
    factorial_factors: tuple[IntegerAffineFactorial, ...] = Field(
        default=(),
        max_length=MAX_PROPER_HYPERGEOMETRIC_FACTORS,
        description="Distinct integer-affine factorial arguments in canonical order.",
    )
    n_base: CanonicalRational = Field(
        default_factory=lambda: CanonicalRational(num=1, den=1),
        description="Nonzero exact rational base raised to the integer parameter n.",
    )
    k_base: CanonicalRational = Field(
        default_factory=lambda: CanonicalRational(num=1, den=1),
        description="Nonzero exact rational base raised to the integer index k.",
    )

    @model_validator(mode="after")
    def require_exact_axes_and_canonical_factors(self) -> Self:
        if self.polynomial.variables != _VARIABLES:
            raise _error(
                "proper_hypergeometric_polynomial_axes",
                "the polynomial prefactor must use the ordered axes (n,k)",
            )
        if self.n_base.as_fraction() == 0 or self.k_base.as_fraction() == 0:
            raise _error(
                "proper_hypergeometric_zero_base",
                "exponential bases must be nonzero rational numbers",
            )
        keys = tuple(factor.affine_key for factor in self.factorial_factors)
        if keys != tuple(sorted(keys)):
            raise _error(
                "proper_hypergeometric_factor_order",
                "factorial factors must be sorted by their affine arguments",
            )
        if len(keys) != len(set(keys)):
            raise _error(
                "proper_hypergeometric_duplicate_factor",
                "combine powers of each identical affine factorial argument",
            )
        if not self.polynomial.polynomial.terms and (
            self.factorial_factors
            or self.n_base.as_fraction() != 1
            or self.k_base.as_fraction() != 1
        ):
            raise _error(
                "proper_hypergeometric_zero_normal_form",
                "the zero term has no factorial factors and unit exponential bases",
            )
        return self

    @property
    def is_zero(self) -> bool:
        """Whether the polynomial prefactor makes this term identically zero."""
        return not self.polynomial.polynomial.terms

    @property
    def support_factors(self) -> tuple[IntegerAffineFactorial, ...]:
        """Reciprocal-factorial inequalities defining the support region."""
        return tuple(factor for factor in self.factorial_factors if factor.power < 0)

    @property
    def domain_factors(self) -> tuple[IntegerAffineFactorial, ...]:
        """Numerator-factorial inequalities defining where the term is defined."""
        return tuple(factor for factor in self.factorial_factors if factor.power > 0)
