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
    precision: StrictInt = Field(ge=1)


class SturmBoundRequest(StrictModel):
    space: ModularFormSpace


class SturmBoundResult(StrictModel):
    space: ModularFormSpace
    index: StrictInt = Field(ge=1)
    bound: StrictInt = Field(ge=0)


class QExpansionTransformRequest(StrictModel):
    expansion: ModularQExpansion


class HeckeRequest(QExpansionTransformRequest):
    index: StrictInt = Field(ge=1)
    output_precision: StrictInt = Field(ge=1)


class URequest(QExpansionTransformRequest):
    prime: StrictInt = Field(ge=2)
    output_precision: StrictInt = Field(ge=1)


class VRequest(QExpansionTransformRequest):
    prime: StrictInt = Field(ge=2)
    output_precision: StrictInt = Field(ge=1)


__all__ = [
    "HeckeRequest",
    "QExpansionTransformRequest",
    "SturmBoundRequest",
    "SturmBoundResult",
    "URequest",
    "VRequest",
]
