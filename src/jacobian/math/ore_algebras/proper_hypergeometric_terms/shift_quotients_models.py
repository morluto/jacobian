"""Request and exact result values for proper hypergeometric shift quotients."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)
from jacobian.math.polynomials.values import RationalFunction


class ProperHypergeometricShiftQuotientsRequest(StrictModel):
    term: ProperHypergeometricTerm


class ProperHypergeometricShiftQuotientsResult(StrictModel):
    """Formal generic-locus quotients with exact axes (n,k)."""

    n_ratio: RationalFunction
    k_ratio: RationalFunction

    @model_validator(mode="after")
    def require_bivariate_axes(self) -> Self:
        if self.n_ratio.variables != ("n", "k") or self.k_ratio.variables != (
            "n",
            "k",
        ):
            raise PydanticCustomError(
                "ore_algebra.hypergeometric_quotient_axes",
                "both shift quotients must use ordered axes (n,k)",
            )
        return self
