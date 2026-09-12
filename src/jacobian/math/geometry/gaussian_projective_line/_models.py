"""Canonical points of the Gaussian-rational projective line."""

from fractions import Fraction
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.number_theory.number_fields.values import (
    MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
)

PROJECTIVE_LINE_FIELD = "Q(i)"
PROJECTIVE_LINE_AXES = ("X", "Y")
CROSS_RATIO_ORDER = "(X1-X3)(X2-X4) / ((X1-X4)(X2-X3))"


def _divide(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    a, b = left
    c, d = right
    norm = c * c + d * d
    if not norm:
        raise ValueError("cannot divide by zero in Q(i)")
    return (a * c + b * d) / norm, (b * c - a * d) / norm


def _fraction_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


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
        normalized_values = tuple(_divide(value, pivot) for value in values)
        if any(
            _fraction_digits(component) > MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
            for value in normalized_values
            for component in value
        ):
            raise PydanticCustomError(
                "geometry.projective_point_normalization_height_bound",
                "projective-point normalization exceeds the Gaussian-rational component bound",
            )
        normalized = tuple(
            GaussianRational.from_fractions(*value) for value in normalized_values
        )
        object.__setattr__(self, "coordinates", normalized)
        return self


class GaussianCrossRatioSource(StrictModel):
    first: GaussianProjectiveLinePoint
    second: GaussianProjectiveLinePoint
    third: GaussianProjectiveLinePoint
    fourth: GaussianProjectiveLinePoint

    @model_validator(mode="after")
    def require_pairwise_distinct_points(self) -> Self:
        points = (self.first, self.second, self.third, self.fourth)
        for left_index, left in enumerate(points):
            for right in points[left_index + 1 :]:
                if left.coordinates == right.coordinates:
                    raise PydanticCustomError(
                        "geometry.projective_points_distinct",
                        "cross-ratio inputs must be pairwise projectively distinct",
                    )
        return self


__all__ = [
    "CROSS_RATIO_ORDER",
    "PROJECTIVE_LINE_AXES",
    "PROJECTIVE_LINE_FIELD",
    "GaussianCrossRatioSource",
    "GaussianProjectiveLinePoint",
]
