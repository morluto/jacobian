"""Typed contract for applying an additive-group coaction to a polynomial."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.values import RationalPolynomial

MAX_GA_ORBIT_SOURCE_TERMS = 128
MAX_GA_ORBIT_SOURCE_DEGREE = 64
MAX_GA_ORBIT_EXPANSIONS = 4_096
MAX_GA_ORBIT_WORK = 4_000_000
MAX_GA_ORBIT_INTERMEDIATE_BYTES = 8_000_000
MAX_GA_ORBIT_OUTPUT_BYTES = 2_000_000
MAX_GA_ORBIT_COEFFICIENT_DIGITS = 128


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial_ga_orbit.{reason}", message)


class GaPolynomialOrbitRequest(StrictModel):
    """One polynomial acted on by a checked action in the same ordered ring."""

    action: PolynomialGaAction = Field(
        description=(
            "A polynomial Ga action; its counit and additive composition law are "
            "checked before orbit expansion."
        )
    )
    polynomial: RationalPolynomial = Field(
        description="A QQ polynomial in the action's exact ordered source-variable axis."
    )

    @model_validator(mode="after")
    def require_same_ordered_ring(self) -> Self:
        if self.polynomial.variables != self.action.source_variables:
            raise _error(
                "ordered_ring_mismatch",
                "the polynomial must use the action's exact ordered source ring",
            )
        return self
