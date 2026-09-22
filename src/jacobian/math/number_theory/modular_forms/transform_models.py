"""Requests/results for finite modular q-series transforms."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
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
    index: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_SOURCE_ORDER)
    output_precision: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_OUTPUT_PRECISION)


class URequest(QExpansionTransformRequest):
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)
    output_precision: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_OUTPUT_PRECISION)


class VRequest(QExpansionTransformRequest):
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)
    output_precision: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_OUTPUT_PRECISION)


__all__ = [
    "HeckeRequest",
    "QExpansionTransformRequest",
    "SturmBoundRequest",
    "SturmBoundResult",
    "URequest",
    "VRequest",
]
