"""Rational coordinate metrics and their complete curvature profiles."""

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    _is_unit_polynomial,
    _polynomial_key,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalFunction,
    SparseRationalPolynomial,
    require_sparse_polynomial_budget,
)


class RationalCoordinateMetric(StrictModel):
    """A symmetric covariant tensor on its generic nondegenerate locus.

    Nondegeneracy is established by the consuming operation. No signature,
    positivity, global chart, or completeness is asserted by this value.
    """

    tensor: RationalCoordinateTensor
    chart_semantics: Literal["GENERIC_NONDEGENERATE_LOCUS"] = (
        "GENERIC_NONDEGENERATE_LOCUS"
    )

    @model_validator(mode="after")
    def require_metric_shape(self) -> Self:
        n = len(self.tensor.coordinate_axis)
        if n > 4 or self.tensor.variance != ("COVARIANT", "COVARIANT"):
            raise ValueError("metric must be a covariant rank-two tensor on 1..4 axes")
        if any(
            self.tensor.components[i * n + j] != self.tensor.components[j * n + i]
            for i in range(n)
            for j in range(i)
        ):
            raise ValueError("metric components must be symmetric")
        return self


class RationalCoordinateConnection(StrictModel):
    """Connection coefficients Gamma^k_ij in the ordered coordinate frame.

    Components follow (k,i,j), last index fastest. A connection is not a
    tensor; its coordinates and retained chart locus travel with its values.
    """

    coordinate_axis: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    components: tuple[RationalFunction, ...] = Field(min_length=1, max_length=64)
    retained_nonzero_denominators: tuple[SparseRationalPolynomial, ...] = Field(
        max_length=768
    )

    @model_validator(mode="after")
    def require_connection_shape(self) -> Self:
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise ValueError("connection coordinate axis must be unique")
        if len(self.components) != len(self.coordinate_axis) ** 3:
            raise ValueError("connection requires exactly n cubed components")
        if any(value.variables != self.coordinate_axis for value in self.components):
            raise ValueError(
                "connection components must use its complete ordered field"
            )
        keys = []
        for guard in self.retained_nonzero_denominators:
            if not guard.terms or any(
                len(term.exponents) != len(self.coordinate_axis) for term in guard.terms
            ):
                raise ValueError(
                    "connection guards must be nonzero on the coordinate field"
                )
            if guard.terms[0].coefficient.as_fraction() != 1 or _is_unit_polynomial(
                guard, len(self.coordinate_axis)
            ):
                raise ValueError("connection guards must be monic and nonconstant")
            require_sparse_polynomial_budget(
                guard,
                maximum_terms=256,
                maximum_exponent=64,
                maximum_coefficient_digits=128,
                label="connection locus guard",
            )
            keys.append(_polynomial_key(guard))
        if keys != sorted(set(keys)):
            raise ValueError("connection guards must be unique and canonically ordered")
        key_set = set(keys)
        if any(
            not _is_unit_polynomial(value.denominator, len(self.coordinate_axis))
            and _polynomial_key(value.denominator) not in key_set
            for value in self.components
        ):
            raise ValueError("connection locus must retain every component denominator")
        return self


class RationalMetricCurvatureRequest(StrictModel):
    metric: RationalCoordinateMetric


class RationalMetricCurvatureProfile(StrictModel):
    """Exact source-bound curvature on the retained nondegenerate locus.

    Riemann components use (l,k,i,j): [nabla_i,nabla_j]v^l=R^l_kij v^k.
    Ricci uses (k,j): Ric_kj=sum_i R^i_kij. Scalar curvature contracts
    inverse_metric with ricci. Component decoding does not replay these
    operation-owned identities.
    """

    metric: RationalCoordinateMetric
    inverse_metric: RationalCoordinateTensor
    connection: RationalCoordinateConnection
    riemann: RationalCoordinateTensor
    ricci: RationalCoordinateTensor
    scalar_curvature: RationalCoordinateTensor

    @model_validator(mode="after")
    def require_profile_axes(self) -> Self:
        axis = self.metric.tensor.coordinate_axis
        guards = self.connection.retained_nonzero_denominators
        if self.connection.coordinate_axis != axis:
            raise ValueError("connection must retain the metric coordinate axis")
        for value, variance in (
            (self.inverse_metric, ("CONTRAVARIANT", "CONTRAVARIANT")),
            (self.riemann, ("CONTRAVARIANT", "COVARIANT", "COVARIANT", "COVARIANT")),
            (self.ricci, ("COVARIANT", "COVARIANT")),
            (self.scalar_curvature, ()),
        ):
            if (
                value.coordinate_axis != axis
                or value.variance != variance
                or value.retained_nonzero_denominators != guards
            ):
                raise ValueError(
                    "profile tensors must retain their declared axes, variance and locus"
                )
        if not set(
            map(_polynomial_key, self.metric.tensor.retained_nonzero_denominators)
        ) <= set(map(_polynomial_key, guards)):
            raise ValueError("profile must retain every source locus guard")
        return self
