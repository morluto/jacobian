"""Typed request and result for proper-hypergeometric Ore action."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


class ProperHypergeometricOperatorActionRequest(StrictModel):
    operator: ShiftOreOperator
    term: ProperHypergeometricTerm


class ProperHypergeometricOperatorActionResult(StrictModel):
    """Source-bound relative multiplier for the action on a term."""

    operator: ShiftOreOperator
    term: ProperHypergeometricTerm
    relative_multiplier: RationalFunction

    @model_validator(mode="after")
    def require_bivariate_multiplier(self) -> Self:
        if self.relative_multiplier.variables != ("n", "k"):
            raise PydanticCustomError(
                "ore_algebra.hypergeometric_action_axes",
                "the relative multiplier must use ordered axes (n,k)",
            )
        return self


def _zero_rational_function() -> RationalFunction:
    zero = SparseRationalPolynomial(terms=())
    one = SparseRationalPolynomial(
        terms=(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1), exponents=(0, 0)
            ),
        )
    )
    return RationalFunction._from_kernel(
        variables=("n", "k"), numerator=zero, denominator=one
    )


__all__ = [
    "ProperHypergeometricOperatorActionRequest",
    "ProperHypergeometricOperatorActionResult",
]
