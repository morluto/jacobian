"""Structural presentations of numerical semigroups.

Minimality and the numerical-semigroup property are claims checked by consumers.
"""

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.numerical_semigroups._models import MAX_GENERATORS


class NumericalSemigroup(StrictModel):
    """A positive, increasing generator axis claimed to be minimal and coprime."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_ordered_axis(self) -> Self:
        values = tuple(map(parse_canonical_integer, self.minimal_generators))
        if values[0] <= 0 or values != tuple(sorted(set(values))):
            raise ValueError(
                "minimal_generators must be positive and strictly increasing"
            )
        return self
