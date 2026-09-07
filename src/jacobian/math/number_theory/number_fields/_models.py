"""Typed wire contracts for number field operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldDiscriminantInteger,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    field: SimpleNumberFieldPresentation


class NumberFieldDiscriminantResult(StrictModel):
    """A field-discriminant claim retaining its exact field presentation."""

    field: SimpleNumberFieldPresentation
    discriminant: NumberFieldDiscriminantInteger


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
