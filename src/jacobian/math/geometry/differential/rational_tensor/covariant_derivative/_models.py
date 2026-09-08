"""Contracts for exact rational coordinate covariant derivatives."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    _polynomial_key,
    canonical_locus_guards,
)


class RationalCovariantDerivativeRequest(StrictModel):
    """A metric and one rational tensor over the same ordered coordinate chart."""

    metric: RationalCoordinateMetric
    tensor: RationalCoordinateTensor

    @model_validator(mode="after")
    def require_shared_chart(self) -> Self:
        if self.tensor.coordinate_axis != self.metric.tensor.coordinate_axis:
            raise ValueError("metric and tensor must use the same coordinate axis")
        return self


class RationalCovariantDerivativeProfile(StrictModel):
    """The exact Levi-Civita covariant derivative of a source tensor.

    The leading result index is covariant and represents the derivative
    coordinate. Remaining indices retain the source tensor's variance and
    lexicographic component order.
    """

    metric: RationalCoordinateMetric
    source: RationalCoordinateTensor
    covariant_derivative: RationalCoordinateTensor

    @model_validator(mode="after")
    def require_profile_binding(self) -> Self:
        axis = self.metric.tensor.coordinate_axis
        if self.source.coordinate_axis != axis:
            raise ValueError("source tensor must use the metric coordinate axis")
        if self.covariant_derivative.coordinate_axis != axis:
            raise ValueError("derivative must use the metric coordinate axis")
        expected_variance = ("COVARIANT", *self.source.variance)
        if self.covariant_derivative.variance != expected_variance:
            raise ValueError("derivative must prepend one covariant index")
        expected_guards = canonical_locus_guards(
            self.metric.tensor.retained_nonzero_denominators,
            self.source.retained_nonzero_denominators,
            component_denominators=tuple(
                value.denominator for value in self.covariant_derivative.components
            ),
            variable_count=len(axis),
        )
        expected_keys = set(map(_polynomial_key, expected_guards))
        actual_keys = set(
            map(
                _polynomial_key, self.covariant_derivative.retained_nonzero_denominators
            )
        )
        if not expected_keys <= actual_keys:
            raise ValueError("derivative must retain source and component locus guards")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        metric: RationalCoordinateMetric,
        source: RationalCoordinateTensor,
        covariant_derivative: RationalCoordinateTensor,
    ) -> Self:
        """Construct after owner-local exact arithmetic and locus assembly."""

        return cls.model_construct(
            metric=metric,
            source=source,
            covariant_derivative=covariant_derivative,
        )


__all__ = [
    "RationalCovariantDerivativeProfile",
    "RationalCovariantDerivativeRequest",
]
