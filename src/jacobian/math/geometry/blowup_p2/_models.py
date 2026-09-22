"""Canonical labelled finite blow-up P2 values and wire contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)

MAX_BLOWUP_POINTS = 16
MAX_BLOWUP_LABEL = 64
MAX_BLOWUP_INTEGER_DIGITS = 256


def _err(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"blowup_p2.{code}", message)


class BlowupPoint(StrictModel):
    label: str = Field(min_length=1, max_length=MAX_BLOWUP_LABEL)
    point: RationalProjectivePoint


class BlowupP2Surface(StrictModel):
    """Blow-up of P2 at one exact ordered labelled rational point set."""

    points: tuple[BlowupPoint, ...] = Field(max_length=MAX_BLOWUP_POINTS)

    @model_validator(mode="after")
    def canonical_labels(self) -> Self:
        labels = tuple(p.label for p in self.points)
        if labels != tuple(sorted(labels)) or len(set(labels)) != len(labels):
            raise _err("point_labels", "blow-up point labels must be unique and sorted")
        return self


class BlowupSurfaceRequest(StrictModel):
    points: tuple[BlowupPoint, ...] = Field(max_length=MAX_BLOWUP_POINTS)


class BlowupDivisorClass(StrictModel):
    surface: BlowupP2Surface
    degree: ExactInteger
    multiplicities: tuple[ExactInteger, ...] = Field(max_length=MAX_BLOWUP_POINTS)

    @model_validator(mode="after")
    def require_axis(self) -> Self:
        if len(self.multiplicities) != len(self.surface.points):
            raise _err(
                "multiplicity_axis",
                "one multiplicity is required for every labelled point",
            )
        return self


class DivisorClassRequest(StrictModel):
    surface: BlowupP2Surface
    degree: ExactInteger
    multiplicities: tuple[ExactInteger, ...] = Field(max_length=MAX_BLOWUP_POINTS)


class IntersectionContribution(StrictModel):
    label: str
    product: ExactInteger


class IntersectionRequest(StrictModel):
    left: BlowupDivisorClass
    right: BlowupDivisorClass


class IntersectionResult(StrictModel):
    left: BlowupDivisorClass
    right: BlowupDivisorClass
    value: ExactInteger
    degree_product: ExactInteger
    exceptional_subtractions: tuple[IntersectionContribution, ...]


class CanonicalClassRequest(StrictModel):
    surface: BlowupP2Surface


class AdjunctionRequest(StrictModel):
    divisor: BlowupDivisorClass


class AdjunctionProfile(StrictModel):
    divisor: BlowupDivisorClass
    canonical: BlowupDivisorClass
    self_intersection: ExactInteger
    canonical_intersection: ExactInteger
    adjunction_pairing: ExactInteger
    arithmetic_genus: ExactInteger
    parity: Literal["INTEGRAL"] = "INTEGRAL"


__all__ = [
    "AdjunctionProfile",
    "AdjunctionRequest",
    "BlowupDivisorClass",
    "BlowupP2Surface",
    "BlowupPoint",
    "BlowupSurfaceRequest",
    "CanonicalClassRequest",
    "DivisorClassRequest",
    "IntersectionContribution",
    "IntersectionRequest",
    "IntersectionResult",
]
