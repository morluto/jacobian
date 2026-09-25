"""Exact Newton polygons for polynomials over a local Laurent series prefix."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from math import gcd, isqrt, lcm
from typing import Literal

from pydantic import Field, StrictInt, TypeAdapter, ValidationError, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
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
from jacobian.math.polynomials.local_series.arithmetic import (
    _check as _check_laurent,
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
            raise ValueError("root multiplicities must reconstruct the edge polynomial degree")
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
    integers = [
        coefficient_by_degree.get(exponent, Fraction(0)) * denominator
        for exponent in range(degree, -1, -1)
    ]
    integer_coefficients = [int(value) for value in integers]
    content = 0
    for coefficient in integer_coefficients:
        content = gcd(content, abs(coefficient))
    integer_coefficients = [coefficient // content for coefficient in integer_coefficients]
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



def _admit_parent(source: LocalPolynomialInSeries) -> None:
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
    # so re-establish unique increasing row ordering before the hull loop.
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
    result_digits = (
        512
        + len(source.coefficients) * 192
        + slots * (2 * MAX_NEWTON_POLYGON_SCALAR_DIGITS + 96)
        + center_digits * 2
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
