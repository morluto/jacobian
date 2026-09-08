"""Contracts for exact rational coordinate covariant derivatives."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.values import RationalCoordinateTensor


class RationalCovariantDerivativeRequest(StrictModel):
    """A metric and one rational tensor over the same ordered coordinate chart."""

    metric: RationalCoordinateMetric
    tensor: RationalCoordinateTensor

    @model_validator(mode="after")
    def require_shared_chart(self) -> Self:
        if self.tensor.coordinate_axis != self.metric.tensor.coordinate_axis:
            raise ValueError("metric and tensor must use the same coordinate axis")
        return self


__all__ = [
    "RationalCovariantDerivativeRequest",
]
