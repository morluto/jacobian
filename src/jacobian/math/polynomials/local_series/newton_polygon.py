"""Exact Newton polygons for polynomials over a local Laurent series prefix."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from math import gcd, isqrt, lcm
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers.complex import (
    MAX_COMPLEX_ALGEBRAIC_COEFFICIENT_DIGITS,
    ComplexAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

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


class NewtonEdgeCharacteristicRequest(StrictModel):
    """Select a lower edge of a local polynomial Newton polygon."""

    polynomial: LocalPolynomialInSeries
    edge_index: StrictInt = Field(ge=0)


class NewtonEdgeCharacteristicTerm(StrictModel):
    """Transport a source coefficient's leading term to the edge polynomial."""

    y_degree: StrictInt = Field(ge=0)
    characteristic_exponent: StrictInt = Field(ge=0)
    leading_coefficient: CanonicalRational


class NewtonEdgeCharacteristicResult(StrictModel):
    """Exact rational edge polynomial, with its source coefficients retained."""

    source: LocalPolynomialInSeries
    edge_index: StrictInt = Field(ge=0)
    edge: NewtonPolygonEdge
    terms: tuple[NewtonEdgeCharacteristicTerm, ...]
    characteristic_polynomial: RationalPolynomial


class NewtonEdgeCharacteristicRoot(StrictModel):
    """One exact root of a supported Newton edge characteristic polynomial."""

    value: CanonicalRational | RealAlgebraicValue | ComplexAlgebraicValue
    multiplicity: StrictInt = Field(ge=1, le=2)


class NewtonEdgeCharacteristicRootsResult(StrictModel):
    """Exact roots of a degree-at-most-two edge polynomial over QQ.

    Algebraic roots use Jacobian's canonical indexed-root values; rational roots
    remain rationals. This is an exact leading-coefficient slice and carries no
    assertion that a later Newton-Puiseux lift exists or has been computed.
    """

    characteristic: NewtonEdgeCharacteristicResult
    roots: tuple[NewtonEdgeCharacteristicRoot, ...] = Field(max_length=2)

    @model_validator(mode="after")
    def require_root_multiplicities(self) -> NewtonEdgeCharacteristicRootsResult:
        degree = max(
            (
                term.exponents[0]
                for term in self.characteristic.characteristic_polynomial.polynomial.terms
            ),
            default=0,
        )
        if sum(root.multiplicity for root in self.roots) != degree:
            raise ValueError(
                "root multiplicities must reconstruct the edge polynomial degree"
            )
        return self


def newton_edge_characteristic_polynomial(
    request: NewtonEdgeCharacteristicRequest,
) -> NewtonEdgeCharacteristicResult:
    """Return the edge polynomial in the leading coefficient variable ``c``.

    Its terms are ``lc(a_j) * c**(j-j_left)`` for source coefficients whose
    valuation points lie on the selected lower edge. Roots describe possible
    nonzero leading coefficients after the edge's valuation substitution; this
    operation does not select or lift roots.
    """
    if not isinstance(request, NewtonEdgeCharacteristicRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="local_series.newton_characteristic_request_type",
            message="request must select an edge of a local polynomial",
        )
    polygon = local_polynomial_newton_polygon(request.polynomial)
    if request.edge_index >= len(polygon.edges):
        raise OperationDomainValidationError(
            location=("edge_index",),
            code="local_series.newton_edge_index",
            message="edge_index must select an edge in the exact lower Newton polygon",
        )
    edge = polygon.edges[request.edge_index]
    left_degree = edge.left.y_degree
    source_rows = {row.y_degree: row for row in request.polynomial.coefficients}
    valuations = dict(polygon.coefficient_valuations)
    transported = []
    polynomial_terms = []
    for degree in edge.source_y_degrees:
        row = source_rows.get(degree)
        valuation = valuations.get(degree)
        if row is None or row.series is None or valuation is None:
            raise OperationDomainValidationError(
                location=("polynomial", "coefficients"),
                code="local_series.newton_source_transport",
                message="the selected edge could not be transported to its source coefficients",
            )
        offset = valuation - row.series.valuation_lower
        coefficient = row.series.coefficients[offset]
        exponent = degree - left_degree
        transported.append(
            NewtonEdgeCharacteristicTerm.model_construct(
                y_degree=degree,
                characteristic_exponent=exponent,
                leading_coefficient=coefficient,
            )
        )
        polynomial_terms.append(
            RationalPolynomialTerm.model_construct(
                coefficient=coefficient,
                exponents=(exponent,),
            )
        )
    characteristic = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("c",),
        polynomial=SparseRationalPolynomial.model_construct(
            terms=tuple(
                sorted(polynomial_terms, key=lambda term: term.exponents, reverse=True)
            )
        ),
    )
    return NewtonEdgeCharacteristicResult.model_construct(
        source=request.polynomial,
        edge_index=request.edge_index,
        edge=edge,
        terms=tuple(transported),
        characteristic_polynomial=characteristic,
    )


def newton_edge_characteristic_roots(
    request: NewtonEdgeCharacteristicRequest,
) -> NewtonEdgeCharacteristicRootsResult:
    """Solve the Newton edge equation exactly when its degree is at most two.

    The quadratic formula is performed over QQ after clearing denominators and
    primitive normalization. Irrational real and nonreal roots are represented
    by the existing canonical algebraic-root carriers. Higher-degree edges are
    rejected before root computation; no numerical root selection is used.
    """
    characteristic = newton_edge_characteristic_polynomial(request)
    terms = characteristic.characteristic_polynomial.polynomial.terms
    degree = max((term.exponents[0] for term in terms), default=0)
    if degree > 2:
        raise OperationResourceAdmissionError(
            location=("characteristic_polynomial",),
            code="local_series.newton_edge_root_degree_bound",
            message="exact edge-root extraction currently admits degree at most two",
        )
    coefficient_by_degree = {
        term.exponents[0]: term.coefficient.as_fraction() for term in terms
    }
    denominator = 1
    for coefficient in coefficient_by_degree.values():
        denominator = lcm(denominator, coefficient.denominator)
    integer_coefficients = [
        int(coefficient_by_degree.get(exponent, Fraction(0)) * denominator)
        for exponent in range(degree, -1, -1)
    ]
    content = 0
    for integer_coefficient in integer_coefficients:
        content = gcd(content, abs(integer_coefficient))
    integer_coefficients = [
        coefficient // content for coefficient in integer_coefficients
    ]
    if integer_coefficients[0] < 0:
        integer_coefficients = [-coefficient for coefficient in integer_coefficients]
    coefficient_digit_bound = min(
        MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
        MAX_COMPLEX_ALGEBRAIC_COEFFICIENT_DIGITS,
    )
    if any(
        len(str(abs(coefficient))) > coefficient_digit_bound
        for coefficient in integer_coefficients
    ):
        raise OperationResourceAdmissionError(
            location=("characteristic_polynomial",),
            code="local_series.newton_edge_root_coefficient_bound",
            message=(
                "primitive edge-root polynomial exceeds the "
                f"{coefficient_digit_bound}-digit algebraic-root carrier bound"
            ),
        )
    if degree == 0:
        roots: tuple[NewtonEdgeCharacteristicRoot, ...] = ()
    elif degree == 1:
        a, b = integer_coefficients
        roots = (
            NewtonEdgeCharacteristicRoot(
                value=CanonicalRational.from_fraction(Fraction(-b, a)),
                multiplicity=1,
            ),
        )
    else:
        a, b, c = integer_coefficients
        discriminant = b * b - 4 * a * c
        if discriminant >= 0 and isqrt(discriminant) ** 2 == discriminant:
            square_root = isqrt(discriminant)
            values = sorted(
                {Fraction(-b - square_root, 2 * a), Fraction(-b + square_root, 2 * a)}
            )
            roots = tuple(
                NewtonEdgeCharacteristicRoot(
                    value=CanonicalRational.from_fraction(value),
                    multiplicity=2 if discriminant == 0 else 1,
                )
                for value in values
            )
        elif discriminant > 0:
            roots = tuple(
                NewtonEdgeCharacteristicRoot(
                    value=RealAlgebraicValue._from_admitted_polynomial(
                        polynomial=tuple(integer_coefficients), real_root_index=index
                    ),
                    multiplicity=1,
                )
                for index in range(2)
            )
        else:
            roots = tuple(
                NewtonEdgeCharacteristicRoot(
                    value=ComplexAlgebraicValue._from_admitted_polynomial(
                        polynomial=tuple(integer_coefficients), root_index=index
                    ),
                    multiplicity=1,
                )
                for index in range(2)
            )
    return NewtonEdgeCharacteristicRootsResult(
        characteristic=characteristic,
        roots=roots,
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
    "NewtonEdgeCharacteristicRequest",
    "NewtonEdgeCharacteristicResult",
    "NewtonEdgeCharacteristicRoot",
    "NewtonEdgeCharacteristicRootsResult",
    "NewtonEdgeCharacteristicTerm",
    "NewtonPolygonEdge",
    "NewtonPolygonPoint",
    "local_polynomial_newton_polygon",
    "newton_edge_characteristic_polynomial",
    "newton_edge_characteristic_roots",
]
