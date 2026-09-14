"""Typed wire contracts for number field operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BeforeValidator, WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldDiscriminantInteger,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)

MAX_INTEGRAL_BASIS_DEGREE = 31


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"number_field.{reason}", message)


class NumberFieldRequest(StrictModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    field: SimpleNumberFieldPresentation


def _require_integral_basis_degree(value: Any) -> Any:
    """Apply the ring-of-integers degree envelope before shared parsing."""

    coefficients = (
        value.coefficients_descending
        if isinstance(value, SimpleNumberFieldPresentation)
        else value.get("coefficients_descending")
        if isinstance(value, Mapping)
        else None
    )
    if isinstance(coefficients, (list, tuple)) and len(coefficients) > (
        MAX_INTEGRAL_BASIS_DEGREE + 1
    ):
        raise _validation_error(
            "ring_of_integers_degree_bound",
            "ring-of-integers computation admits degree at most "
            f"{MAX_INTEGRAL_BASIS_DEGREE}",
        )
    if isinstance(value, Mapping):
        return canonicalize_json_containers(value)
    return value


def _integral_basis_field_schema() -> dict[str, Any]:
    schema = SimpleNumberFieldPresentation.model_json_schema(mode="validation")
    schema["properties"]["coefficients_descending"]["maxItems"] = (
        MAX_INTEGRAL_BASIS_DEGREE + 1
    )
    schema["description"] = (
        "A canonical simple number-field presentation of degree at most "
        f"{MAX_INTEGRAL_BASIS_DEGREE} for ring-of-integers computation."
    )
    return schema


_IntegralBasisField = Annotated[
    SimpleNumberFieldPresentation,
    BeforeValidator(_require_integral_basis_degree),
    WithJsonSchema(_integral_basis_field_schema()),
]


class NumberFieldRingOfIntegersRequest(StrictModel):
    """A simple number field within the integral-basis worker envelope."""

    field: _IntegralBasisField


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
    "MAX_INTEGRAL_BASIS_DEGREE",
    "NumberFieldDiscriminantResult",
    "NumberFieldEmbeddingsRequest",
    "NumberFieldRealEmbeddingOrderRequest",
    "NumberFieldRequest",
    "NumberFieldRingOfIntegersRequest",
]
