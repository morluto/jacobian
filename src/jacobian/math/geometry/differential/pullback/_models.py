"""Typed values for exact rational metric pullbacks."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    _polynomial_key,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import SparseRationalPolynomial


class RationalMetricPullbackRequest(StrictModel):
    metric: RationalCoordinateMetric
    map: RationalFunctionMap

    @model_validator(mode="after")
    def require_target_axis(self) -> Self:
        if self.map.target_coordinates != self.metric.tensor.coordinate_axis:
            raise ValueError("map target coordinates must equal metric coordinate axis")
        return self


class RationalMetricPullbackProfile(StrictModel):
    metric: RationalCoordinateMetric
    map: RationalFunctionMap
    pullback: RationalCoordinateTensor
    pullback_locus_guard: tuple[SparseRationalPolynomial, ...] = Field(max_length=768)

    @model_validator(mode="after")
    def require_profile_contract(self) -> Self:
        source_axis = self.map.source_variables
        if self.pullback.coordinate_axis != source_axis:
            raise ValueError("pullback must use the map source coordinate axis")
        if self.pullback.variance != ("COVARIANT", "COVARIANT"):
            raise ValueError("pullback must be a covariant rank-two tensor")
        if self.map.target_coordinates != self.metric.tensor.coordinate_axis:
            raise ValueError("map target coordinates must equal metric coordinate axis")
        keys = [_polynomial_key(value) for value in self.pullback_locus_guard]
        if keys != sorted(set(keys)):
            raise ValueError("pullback guards must be unique and canonically ordered")
        if any(
            any(len(term.exponents) != len(source_axis) for term in value.terms)
            for value in self.pullback_locus_guard
        ):
            raise ValueError("pullback guards must use the map source axis")
        guard_keys = set(keys)
        tensor_keys = {
            _polynomial_key(value)
            for value in self.pullback.retained_nonzero_denominators
        }
        if not tensor_keys <= guard_keys:
            raise ValueError(
                "pullback tensor denominators must be retained by its locus"
            )
        return self


__all__ = ["RationalMetricPullbackProfile", "RationalMetricPullbackRequest"]
