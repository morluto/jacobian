"""Canonical points of the Gaussian-rational projective line."""

from fractions import Fraction
from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields import GaussianRational


def _divide(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    a, b = left
    c, d = right
    norm = c * c + d * d
    if not norm:
        raise ValueError("cannot divide by zero in Q(i)")
    return (a * c + b * d) / norm, (b * c - a * d) / norm


class GaussianProjectiveLinePoint(StrictModel):
    """A first-nonzero-coordinate-normalized point of P^1(Q(i))."""

    coordinates: tuple[GaussianRational, GaussianRational]

    @model_validator(mode="after")
    def normalize_homogeneous_coordinates(self) -> Self:
        values = tuple(coordinate.as_fractions() for coordinate in self.coordinates)
        pivot = next((value for value in values if value[0] or value[1]), None)
        if pivot is None:
            raise ValueError(
                "a projective point needs a nonzero homogeneous coordinate"
            )
        normalized = tuple(
            GaussianRational.from_fractions(*_divide(value, pivot)) for value in values
        )
        object.__setattr__(self, "coordinates", normalized)
        return self


class GaussianCrossRatioSource(StrictModel):
    first: GaussianProjectiveLinePoint
    second: GaussianProjectiveLinePoint
    third: GaussianProjectiveLinePoint
    fourth: GaussianProjectiveLinePoint


__all__ = ["GaussianCrossRatioSource", "GaussianProjectiveLinePoint"]
