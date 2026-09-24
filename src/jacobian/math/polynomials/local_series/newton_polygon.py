"""Exact Newton polygons for polynomials over a local Laurent series prefix."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)
from jacobian.math.polynomials.values import PolynomialVariable

MAX_LOCAL_POLYNOMIAL_ROWS = 256
MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS = 8192
MAX_NEWTON_POLYGON_Y_DEGREE = 32_768
MAX_NEWTON_POLYGON_SCALAR_DIGITS = 256
MAX_NEWTON_POLYGON_OUTPUT_BYTES = CanonicalLimits().max_output_bytes


class LocalPolynomialCoefficient(StrictModel):
    """One coefficient of y^degree; None denotes the exact zero coefficient."""

    y_degree: StrictInt = Field(ge=0, le=MAX_NEWTON_POLYGON_Y_DEGREE)
    series: TruncatedLaurentWindow | None = Field(
        description="None means exact zero; a finite prefix alone cannot prove zero."
    )


class LocalPolynomialInSeries(StrictModel):
    """Sparse polynomial in y with coefficients in one local series parent."""

    variable: PolynomialVariable
    place: Literal["FINITE", "INFINITY"] = "FINITE"
    center: CanonicalRational = CanonicalRational(num=0, den=1)
    coefficients: tuple[LocalPolynomialCoefficient, ...]

    @model_validator(mode="after")
    def require_canonical_rows(self) -> LocalPolynomialInSeries:
        degrees = tuple(row.y_degree for row in self.coefficients)
        if degrees != tuple(sorted(set(degrees))):
            raise ValueError(
                "local polynomial rows must have unique increasing y degrees"
            )
        if self.place == "INFINITY" and self.center.as_fraction() != 0:
            raise ValueError("an infinity local polynomial has center zero")
        for row in self.coefficients:
            if row.series is None:
                continue
            series = row.series
            if (series.variable, series.place, series.center) != (
                self.variable,
                self.place,
                self.center,
            ):
                raise ValueError(
                    "all coefficient series must share the declared local parent"
                )
        return self


class NewtonPolygonPoint(StrictModel):
    y_degree: StrictInt = Field(ge=0)
    valuation: StrictInt


class NewtonPolygonEdge(StrictModel):
    left: NewtonPolygonPoint
    right: NewtonPolygonPoint
    slope: CanonicalRational
    horizontal_length: StrictInt = Field(gt=0)
    source_y_degrees: tuple[StrictInt, ...]


class LocalPolynomialNewtonPolygonResult(StrictModel):
    """Source-bound lower Newton hull; all valuations are exact over QQ."""

    source: LocalPolynomialInSeries
    coefficient_valuations: tuple[tuple[StrictInt, StrictInt | None], ...]
    points: tuple[NewtonPolygonPoint, ...]
    vertices: tuple[NewtonPolygonPoint, ...]
    edges: tuple[NewtonPolygonEdge, ...]


def _admit(
    source: LocalPolynomialInSeries,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not isinstance(source, LocalPolynomialInSeries):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="local_series.newton_polynomial_type",
            message="polynomial must be a local polynomial in Laurent series",
        )
    if len(source.coefficients) > MAX_LOCAL_POLYNOMIAL_ROWS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.newton_rows_bound",
            message=f"local polynomial exceeds {MAX_LOCAL_POLYNOMIAL_ROWS} coefficient rows",
        )
    slots = sum(
        len(row.series.coefficients)
        for row in source.coefficients
        if row.series is not None
    )
    if slots > MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.newton_slots_bound",
            message=f"local polynomial exceeds {MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS} retained series slots",
        )
    valuations: list[tuple[int, int]] = []
    points: list[tuple[int, int]] = []
    for row_index, row in enumerate(source.coefficients):
        if row.series is None:
            continue
        if len(row.series.coefficients) > MAX_LOCAL_SERIES_TERMS:
            raise OperationResourceAdmissionError(
                location=("polynomial", "coefficients", row_index),
                code="local_series.newton_row_width_bound",
                message="coefficient series exceeds the local-series term bound",
            )
        valuation = None
        for offset, coefficient in enumerate(row.series.coefficients):
            value = coefficient.as_fraction()
            if max(len(str(abs(value.numerator))), len(str(value.denominator))) > min(
                MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
                MAX_NEWTON_POLYGON_SCALAR_DIGITS,
            ):
                raise OperationResourceAdmissionError(
                    location=("polynomial", "coefficients", row_index),
                    code="local_series.newton_scalar_bound",
                    message="coefficient exceeds the local-series scalar bound",
                )
            if value and valuation is None:
                valuation = row.series.valuation_lower + offset
        if valuation is None:
            raise OperationDomainValidationError(
                location=("polynomial", "coefficients", row_index, "series"),
                code="local_series.newton_valuation_unknown",
                message="a zero finite prefix does not determine the coefficient valuation; use exact zero (null) or provide a nonzero term",
            )
        valuations.append((row.y_degree, valuation))
        points.append((row.y_degree, valuation))
    center = source.center.as_fraction()
    center_digits = max(len(str(abs(center.numerator))), len(str(center.denominator)))
    output_bound = (
        512
        + len(source.coefficients) * 192
        + slots * (2 * MAX_NEWTON_POLYGON_SCALAR_DIGITS + 96)
        + center_digits * 2
    )
    if output_bound > MAX_NEWTON_POLYGON_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="local_series.newton_output_bound",
            message="local Newton polygon source and result exceed the canonical output envelope",
        )
    return valuations, points


def local_polynomial_newton_polygon(
    source: LocalPolynomialInSeries,
) -> LocalPolynomialNewtonPolygonResult:
    """Compute the exact lower convex hull of (y-degree, local valuation)."""
    valuation_rows, point_rows = _admit(source)
    hull: list[tuple[int, int]] = []
    for point in point_rows:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (b[0] - a[0]) * (point[1] - b[1]) - (b[1] - a[1]) * (
                point[0] - b[0]
            )
            if cross > 0:
                break
            hull.pop()
        hull.append(point)

    def wire(point: tuple[int, int]) -> NewtonPolygonPoint:
        return NewtonPolygonPoint.model_construct(y_degree=point[0], valuation=point[1])

    edges = []
    for left, right in pairwise(hull):
        collinear = tuple(
            degree
            for degree, value in point_rows
            if left[0] <= degree <= right[0]
            and (degree - left[0]) * (right[1] - left[1])
            == (value - left[1]) * (right[0] - left[0])
        )
        edges.append(
            NewtonPolygonEdge.model_construct(
                left=wire(left),
                right=wire(right),
                slope=CanonicalRational.from_fraction(
                    Fraction(right[1] - left[1], right[0] - left[0])
                ),
                horizontal_length=right[0] - left[0],
                source_y_degrees=collinear,
            )
        )
    return LocalPolynomialNewtonPolygonResult.model_construct(
        source=source,
        coefficient_valuations=tuple(
            (row.y_degree, dict(valuation_rows).get(row.y_degree))
            for row in source.coefficients
        ),
        points=tuple(wire(point) for point in point_rows),
        vertices=tuple(wire(point) for point in hull),
        edges=tuple(edges),
    )


__all__ = [
    "LocalPolynomialCoefficient",
    "LocalPolynomialInSeries",
    "LocalPolynomialNewtonPolygonResult",
    "NewtonPolygonEdge",
    "NewtonPolygonPoint",
    "local_polynomial_newton_polygon",
]
