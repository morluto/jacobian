"""Contracts for exact rational Laplace--Beltrami values."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateMetric,
)
from jacobian.math.geometry.differential.values import (
    _is_unit_polynomial,
    _polynomial_key,
    canonical_locus_guards,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    SparseRationalPolynomial,
    require_sparse_polynomial_budget,
)


class RationalLaplaceBeltramiRequest(StrictModel):
    """A rational metric and scalar field on the same ordered chart."""

    metric: RationalCoordinateMetric
    scalar: RationalFunction

    @model_validator(mode="after")
    def require_shared_chart(self) -> Self:
        if self.scalar.variables != self.metric.tensor.coordinate_axis:
            raise ValueError("metric and scalar must use the same coordinate axis")
        return self


class RationalLaplaceBeltramiResult(StrictModel):
    """The exact scalar Laplace--Beltrami value and its retained sources."""

    metric: RationalCoordinateMetric
    scalar: RationalFunction
    value: RationalFunction
    retained_nonzero_denominators: tuple[SparseRationalPolynomial, ...] = Field(
        default=(), max_length=768
    )

    @model_validator(mode="after")
    def require_result_axes(self) -> Self:
        axis = self.metric.tensor.coordinate_axis
        if self.scalar.variables != axis or self.value.variables != axis:
            raise ValueError("metric, scalar, and value must share the coordinate axis")
        for guard in self.retained_nonzero_denominators:
            if (
                not guard.terms
                or any(len(term.exponents) != len(axis) for term in guard.terms)
                or guard.terms[0].coefficient.as_fraction() != 1
                or _is_unit_polynomial(guard, len(axis))
            ):
                raise ValueError(
                    "result locus guards must be monic nonconstant polynomials"
                )
            require_sparse_polynomial_budget(
                guard,
                maximum_terms=256,
                maximum_exponent=64,
                maximum_coefficient_digits=128,
                label="Laplace--Beltrami locus guard",
            )
        expected = canonical_locus_guards(
            self.metric.tensor.retained_nonzero_denominators,
            component_denominators=(self.value.denominator,),
            variable_count=len(axis),
        )
        expected_keys = {_polynomial_key(guard) for guard in expected}
        actual_order = tuple(
            _polynomial_key(guard) for guard in self.retained_nonzero_denominators
        )
        actual_keys = set(actual_order)
        if (
            actual_order != tuple(sorted(set(actual_order)))
            or not expected_keys <= actual_keys
        ):
            raise ValueError("result must retain the metric and value locus guards")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        metric: RationalCoordinateMetric,
        scalar: RationalFunction,
        value: RationalFunction,
        retained_nonzero_denominators: tuple[SparseRationalPolynomial, ...],
    ) -> Self:
        """Construct after owner-local exact arithmetic and locus assembly."""

        return cls.model_construct(
            metric=metric,
            scalar=scalar,
            value=value,
            retained_nonzero_denominators=retained_nonzero_denominators,
        )


__all__ = ["RationalLaplaceBeltramiRequest", "RationalLaplaceBeltramiResult"]
