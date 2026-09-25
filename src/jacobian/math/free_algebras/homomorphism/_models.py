"""Typed contracts for free-algebra homomorphism evaluation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import (
    MAX_FREE_ALGEBRA_GENERATORS,
    FreeAlgebraLetter,
    FreeAlgebraPolynomial,
)


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"free_algebra.{code}", message)


class FreeAlgebraHomomorphism(StrictModel):
    """The unique algebra map specified by ordered generator images over QQ."""

    source_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    target_alphabet: tuple[FreeAlgebraLetter, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS
    )
    generator_images: tuple[FreeAlgebraPolynomial, ...] = Field(
        max_length=MAX_FREE_ALGEBRA_GENERATORS,
        description="One target polynomial image for each source generator, in axis order.",
    )

    @model_validator(mode="after")
    def require_bound_axes(self) -> Self:
        if len(set(self.source_alphabet)) != len(self.source_alphabet):
            raise _error(
                "alphabet_letters_not_distinct", "source letters must be distinct"
            )
        if len(set(self.target_alphabet)) != len(self.target_alphabet):
            raise _error(
                "alphabet_letters_not_distinct", "target letters must be distinct"
            )
        if len(self.generator_images) != len(self.source_alphabet):
            raise _error(
                "homomorphism_image_count",
                "one image is required for every source generator",
            )
        if any(
            image.alphabet != self.target_alphabet for image in self.generator_images
        ):
            raise _error(
                "homomorphism_target_axis",
                "every image must use the declared target alphabet",
            )
        return self


class FreeAlgebraHomomorphismApplyRequest(StrictModel):
    homomorphism: FreeAlgebraHomomorphism
    polynomial: FreeAlgebraPolynomial

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if self.polynomial.alphabet != self.homomorphism.source_alphabet:
            raise _error(
                "homomorphism_source_axis",
                "input polynomial must use the homomorphism source alphabet",
            )
        return self


class FreeAlgebraHomomorphismApplyResult(StrictModel):
    homomorphism: FreeAlgebraHomomorphism
    polynomial: FreeAlgebraPolynomial
    image: FreeAlgebraPolynomial

    @model_validator(mode="after")
    def require_result_axis(self) -> Self:
        if self.polynomial.alphabet != self.homomorphism.source_alphabet:
            raise _error(
                "homomorphism_source_axis",
                "retained polynomial must use the source alphabet",
            )
        if self.image.alphabet != self.homomorphism.target_alphabet:
            raise _error(
                "homomorphism_target_axis", "image must use the target alphabet"
            )
        return self
