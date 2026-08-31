"""Typed wire contracts for number field operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    field: SimpleNumberFieldPresentation


class NumberFieldDiscriminantResult(StrictModel):
    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    discriminant: str | None = None
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_outcome(self) -> Self:
        if self.status == "COMPLETE" and self.discriminant is None:
            raise _validation_error(
                "complete_discriminant_requires_value",
                "a complete number-field discriminant requires its exact value",
            )
        if self.status == "UNKNOWN" and (
            self.discriminant is not None or self.detail is None
        ):
            raise _validation_error(
                "unknown_discriminant_shape",
                "an unknown number-field computation requires detail and no value",
            )
        return self


class NumberFieldEmbeddingsRequest(StrictModel):
    """Request every Archimedean embedding of one bounded presented field."""

    field: SimpleNumberFieldPresentation


class NumberFieldRealEmbeddingOrderRequest(StrictModel):
    """Compare two field elements at one selected real embedding record."""

    left: SimpleNumberFieldRealEmbeddingBinding
    right: SimpleNumberFieldRealEmbeddingBinding


__all__ = [
    "NumberFieldDiscriminantResult",
    "NumberFieldEmbeddingsRequest",
    "NumberFieldRealEmbeddingOrderRequest",
    "NumberFieldRequest",
]
