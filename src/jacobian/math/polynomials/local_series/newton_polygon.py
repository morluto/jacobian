"""Exact Newton polygons for polynomials over a local Laurent series prefix."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal

from pydantic import Field, StrictInt, TypeAdapter, ValidationError, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.arithmetic import (
    _check as _check_laurent,
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
MAX_NEWTON_POLYGON_RESULT_DIGITS = (
    MAX_LOCAL_POLYNOMIAL_ROWS * 64
    + MAX_LOCAL_POLYNOMIAL_SERIES_SLOTS * (2 * MAX_NEWTON_POLYGON_SCALAR_DIGITS + 64)
    + 1024
)
_VARIABLE_ADAPTER = TypeAdapter(PolynomialVariable)


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


def _admit_parent(source: LocalPolynomialInSeries) -> None:
    if not isinstance(source.coefficients, tuple):
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients"),
            code="local_series.newton_rows",
            message="local polynomial coefficients must be a tuple",
        )
    if len(source.coefficients) > MAX_LOCAL_POLYNOMIAL_ROWS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients"),
            code="local_series.newton_rows_bound",
            message=f"local polynomial exceeds {MAX_LOCAL_POLYNOMIAL_ROWS} coefficient rows",
        )
    if type(source.variable) is not str:
        raise OperationDomainValidationError(
            location=("polynomial", "variable"),
            code="local_series.newton_variable",
            message="local polynomial variable must be a strict identifier",
        )
    try:
        _VARIABLE_ADAPTER.validate_python(source.variable, strict=True)
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("polynomial", "variable"),
            code="local_series.newton_variable",
            message="local polynomial variable must match the polynomial identifier grammar",
        ) from error
    if source.place not in ("FINITE", "INFINITY"):
        raise OperationDomainValidationError(
            location=("polynomial", "place"),
            code="local_series.newton_place",
            message="local polynomial expansion place must be FINITE or INFINITY",
        )
    if not isinstance(source.center, CanonicalRational):
        raise OperationDomainValidationError(
            location=("polynomial", "center"),
            code="local_series.newton_center",
            message="local polynomial center must be a canonical rational",
        )
    # Native callers can bypass the Pydantic validator with model_construct(),
    # so re-establish the center's reduced components, denominator
    # positivity, and scalar bound before the hull consumes it, mirroring the
    # nested Laurent and Puiseux window admission.
    center_num = getattr(source.center, "num", None)
    center_den = getattr(source.center, "den", None)
    if type(center_num) is not int or type(center_den) is not int:
        raise OperationDomainValidationError(
            location=("polynomial", "center"),
            code="local_series.newton_center",
            message="local polynomial center components must be strict integers",
        )
    try:
        center = source.center.as_fraction()
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OperationDomainValidationError(
            location=("polynomial", "center"),
            code="local_series.newton_center",
            message="local polynomial center must be a valid canonical rational",
        ) from error
    if center_den <= 0 or (center_num, center_den) != (
        center.numerator,
        center.denominator,
    ):
        raise OperationDomainValidationError(
            location=("polynomial", "center"),
            code="local_series.newton_center",
            message=(
                "local polynomial center must be reduced with a positive denominator"
            ),
        )
    try:
        require_bounded_rational(
            source.center,
            max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
            label="local polynomial center",
        )
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("polynomial", "center"),
            code="local_series.newton_center_bound",
            message=str(error),
        ) from error
    if source.place == "INFINITY" and center != 0:
        raise OperationDomainValidationError(
            location=("polynomial", "center"),
            code="local_series.newton_infinity_center",
            message="an infinity local polynomial has center zero",
        )
    # Native callers can bypass the Pydantic validator with model_construct(),
    # so validate row values before reading degrees, then re-establish unique
    # increasing ordering before the hull loop.
    for index, row in enumerate(source.coefficients):
        if not isinstance(row, LocalPolynomialCoefficient):
            raise OperationDomainValidationError(
                location=("polynomial", "coefficients", index),
                code="local_series.newton_row_type",
                message="local polynomial rows must be coefficient values",
            )
    degrees = tuple(row.y_degree for row in source.coefficients)
    if degrees != tuple(sorted(set(degrees))):
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients"),
            code="local_series.newton_row_order",
            message="local polynomial rows must have unique increasing y degrees",
        )


def _admit_row(
    source: LocalPolynomialInSeries,
    row_index: int,
) -> None:
    row = source.coefficients[row_index]
    if (
        type(row.y_degree) is not int
        or not 0 <= row.y_degree <= MAX_NEWTON_POLYGON_Y_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients", row_index, "y_degree"),
            code="local_series.newton_row_degree",
            message="local polynomial row degree is outside its domain",
        )
    if row.series is None:
        return
    try:
        _check_laurent(row.series)
    except OperationResourceAdmissionError as error:
        raise OperationResourceAdmissionError(
            location=("polynomial", "coefficients", row_index, "series"),
            code="local_series.newton_series_bound",
            message=f"coefficient series exceeds its admitted envelope: {error}",
        ) from error
    except OperationDomainValidationError as error:
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients", row_index, "series"),
            code="local_series.newton_series_parent",
            message=f"coefficient series failed structural admission: {error}",
        ) from error
    if (row.series.variable, row.series.place, row.series.center) != (
        source.variable,
        source.place,
        source.center,
    ):
        raise OperationDomainValidationError(
            location=("polynomial", "coefficients", row_index, "series"),
            code="local_series.newton_parent_mismatch",
            message="all coefficient series must share the declared local parent",
        )


def _admit(
    source: LocalPolynomialInSeries,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not isinstance(source, LocalPolynomialInSeries):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="local_series.newton_polynomial_type",
            message="polynomial must be a local polynomial in Laurent series",
        )
    _admit_parent(source)
    for row_index in range(len(source.coefficients)):
        _admit_row(source, row_index)
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
            if max(
                len(format_canonical_integer(abs(value.numerator))),
                len(format_canonical_integer(value.denominator)),
            ) > min(
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
    center_digits = max(
        len(format_canonical_integer(abs(center.numerator))),
        len(format_canonical_integer(center.denominator)),
    )
    component_digits = sum(
        max(
            len(format_canonical_integer(abs(value.numerator))),
            len(format_canonical_integer(value.denominator)),
        )
        for row in source.coefficients
        if row.series is not None
        for coefficient in row.series.coefficients
        for value in (coefficient.as_fraction(),)
    )
    result_digits = (
        512 + len(source.coefficients) * 192 + component_digits * 2 + center_digits * 2
    )
    if result_digits > MAX_NEWTON_POLYGON_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="local_series.newton_result_envelope",
            message="local Newton polygon source and result exceed the admitted digit envelope",
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
