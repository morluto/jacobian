"""Requests/results for finite modular q-series transforms."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormSpace,
    ModularQExpansion,
)


class NamedQExpansionRequest(StrictModel):
    space: ModularFormSpace
    form: str
    precision: StrictInt = Field(
        ge=1,
        le=4096,
        description="Number of coefficients in the bounded transform envelope.",
    )


class SturmBoundRequest(StrictModel):
    space: ModularFormSpace


class SturmBoundResult(StrictModel):
    space: ModularFormSpace
    index: StrictInt = Field(ge=1)
    bound: StrictInt = Field(ge=0)


class QExpansionTransformRequest(StrictModel):
    expansion: ModularQExpansion


class HeckeRequest(QExpansionTransformRequest):
    index: StrictInt = Field(
        ge=1, le=256, description="Hecke index in the bounded exact transform envelope."
    )
    output_precision: StrictInt = Field(
        ge=1, le=4096, description="Number of output coefficients."
    )


class URequest(QExpansionTransformRequest):
    prime: StrictInt = Field(
        ge=2,
        le=256,
        description="Prime U index in the bounded exact transform envelope.",
    )
    output_precision: StrictInt = Field(
        ge=1, le=4096, description="Number of output coefficients."
    )


class VRequest(QExpansionTransformRequest):
    prime: StrictInt = Field(
        ge=2,
        le=256,
        description="Prime V index in the bounded exact transform envelope.",
    )
    output_precision: StrictInt = Field(
        ge=1, le=4096, description="Number of output coefficients."
    )


__all__ = [
    "HeckeRequest",
    "QExpansionTransformRequest",
    "SturmBoundRequest",
    "SturmBoundResult",
    "URequest",
    "VRequest",
]
