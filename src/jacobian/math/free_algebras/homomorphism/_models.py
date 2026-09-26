"""Typed contracts for free-algebra homomorphism evaluation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import (
    FreeAlgebraPolynomial,
    FreeAlgebraPolynomialHomomorphism,
)


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"free_algebra.{code}", message)


class FreeAlgebraHomomorphismApplyRequest(StrictModel):
    """One polynomial and the canonical map that acts on its source algebra."""

    homomorphism: FreeAlgebraPolynomialHomomorphism = Field(
        description=(
            "The canonical map specified by source and target alphabets and one "
            "target polynomial image per source generator. Each image admits "
            "at most 64 terms, with words of at most 32 letters."
        )
    )
    polynomial: FreeAlgebraPolynomial = Field(
        description=(
            "The source polynomial, limited at execution to 64 terms with "
            "words of at most 32 letters."
        )
    )

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if self.polynomial.alphabet != self.homomorphism.source_alphabet:
            raise _error(
                "homomorphism_source_axis",
                "input polynomial must use the homomorphism source alphabet",
            )
        return self
