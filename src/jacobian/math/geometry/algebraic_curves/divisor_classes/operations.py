"""Exact plane-curve multiplicities and strict-transform classes."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import NoReturn

from jacobian._exact import CanonicalRational
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.algebraic_curves.divisor_classes._models import (
    MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS,
    MAX_CURVE_DIVISOR_DEGREE,
    MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS,
    MAX_CURVE_DIVISOR_OUTPUT_BYTES,
    MAX_CURVE_DIVISOR_POINT_DIGITS,
    MAX_CURVE_DIVISOR_TERMS,
    MAX_CURVE_DIVISOR_WORK,
    PlaneCurveStrictTransformRequest,
)
from jacobian.math.geometry.blowup_p2._models import (
    BlowupDivisorClass,
    BlowupP2Surface,
    BlowupPoint,
)
from jacobian.math.geometry.blowup_p2.operations import (
    construct_divisor_class,
    construct_surface,
)
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.polynomials.values import RationalPolynomial

MAX_BLOWUP_LABEL_CHARS = 64


def _domain(reason: str, message: str, location: tuple[str | int, ...]) -> NoReturn:
    raise OperationDomainValidationError(
        location=location, code=f"plane_curve_divisor.{reason}", message=message
    )


def _admit_polynomial(polynomial: RationalPolynomial) -> int:
    if len(polynomial.variables) != 3 or not polynomial.polynomial.terms:
        _domain(
            "polynomial_axis",
            "a plane curve requires a nonzero ternary polynomial",
            ("polynomial",),
        )
    if len(polynomial.polynomial.terms) > MAX_CURVE_DIVISOR_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_curve_divisor.term_bound",
            message="plane-curve source admits at most 64 terms",
        )
    degrees = {sum(term.exponents) for term in polynomial.polynomial.terms}
    if len(degrees) != 1:
        _domain(
            "inhomogeneous", "the curve polynomial must be homogeneous", ("polynomial",)
        )
    degree = next(iter(degrees))
    if not 1 <= degree <= MAX_CURVE_DIVISOR_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_curve_divisor.degree_bound",
            message="the plane-curve degree must be 1..12",
        )
    for index, term in enumerate(polynomial.polynomial.terms):
        if len(term.exponents) != 3 or any(
            exponent < 0 or exponent > degree for exponent in term.exponents
        ):
            _domain(
                "term_shape",
                "homogeneous curve terms must use three bounded exponents",
                ("polynomial", index),
            )
        if not isinstance(term.coefficient, CanonicalRational):
            _domain(
                "coefficient_type",
                "curve coefficients must be canonical rationals",
                ("polynomial", index),
            )
        if (
            max(
                decimal_digit_width(term.coefficient.num),
                decimal_digit_width(term.coefficient.den),
            )
            > MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("polynomial", index),
                code="plane_curve_divisor.coefficient_bound",
                message="curve coefficients are limited to 32 decimal digits",
            )
    return degree


def _admit_surface_points(surface: BlowupP2Surface) -> int:
    if not isinstance(surface.points, tuple) or len(surface.points) > 16:
        raise OperationResourceAdmissionError(
            location=("surface",),
            code="plane_curve_divisor.point_bound",
            message="at most 16 blow-up points are admitted",
        )
    # Validate every point's raw arithmetic before canonicalizing the surface.
    for point_index, row in enumerate(surface.points):
        if not isinstance(row, BlowupPoint) or not isinstance(
            row.point, RationalProjectivePoint
        ):
            _domain(
                "point_shape",
                "surface entries must be labelled rational projective points",
                ("surface", point_index),
            )
        if (
            type(row.label) is not str
            or not 1 <= len(row.label) <= MAX_BLOWUP_LABEL_CHARS
        ):
            _domain(
                "point_label",
                "blow-up labels must be 1..64 characters",
                ("surface", point_index),
            )
        coordinates = row.point.coordinates
        if not isinstance(coordinates, tuple) or len(coordinates) != 3:
            _domain(
                "point_axis",
                "blow-up points must use three coordinates",
                ("surface", point_index),
            )
        for coordinate_index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, CanonicalRational):
                _domain(
                    "point_scalar",
                    "point coordinates must be canonical rationals",
                    ("surface", point_index, coordinate_index),
                )
            if (
                max(
                    decimal_digit_width(coordinate.num),
                    decimal_digit_width(coordinate.den),
                )
                > MAX_CURVE_DIVISOR_POINT_DIGITS
            ):
                raise OperationResourceAdmissionError(
                    location=("surface", point_index, coordinate_index),
                    code="plane_curve_divisor.point_height",
                    message="projective point components are limited to 16 decimal digits",
                )
        if all(coordinate.as_fraction() == 0 for coordinate in coordinates):
            _domain(
                "zero_point",
                "blow-up points must be nonzero projective points",
                ("surface", point_index),
            )
    return len(surface.points)


def _reject_duplicate_points(surface: BlowupP2Surface) -> None:
    labels = tuple(point.label for point in surface.points)
    if len(set(labels)) != len(labels):
        _domain("duplicate_label", "blow-up point labels must be unique", ("surface",))
    for first, second in combinations(surface.points, 2):
        left = first.point.coordinates
        right = second.point.coordinates
        if all(
            left[i].as_fraction() * right[j].as_fraction()
            == left[j].as_fraction() * right[i].as_fraction()
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            _domain(
                "duplicate_point",
                "blow-up points must be distinct projective points",
                ("surface",),
            )


def _admit(request: PlaneCurveStrictTransformRequest) -> tuple[int, int, int]:
    """Admit the complete polynomial, point family, and retained result first."""
    if not isinstance(request, PlaneCurveStrictTransformRequest):
        _domain(
            "request_type",
            "expected a plane-curve strict-transform request",
            ("request",),
        )
    polynomial, surface = request.polynomial, request.surface
    if not isinstance(polynomial, RationalPolynomial) or not isinstance(
        surface, BlowupP2Surface
    ):
        _domain(
            "request_shape",
            "request must retain canonical polynomial and surface values",
            ("request",),
        )
    degree = _admit_polynomial(polynomial)
    if tuple(sorted(request.projective_coordinate_variables)) != tuple(
        sorted(polynomial.variables)
    ):
        _domain(
            "coordinate_transport",
            "projective coordinate variables must permute the polynomial axis",
            ("projective_coordinate_variables",),
        )
    point_count = _admit_surface_points(surface)
    derivative_slots = (degree + 1) * (degree + 2) // 2
    work = (
        6 * point_count * len(polynomial.polynomial.terms) * derivative_slots
        + point_count * point_count * 9
        + point_count * 128
    )
    point_component_digits = 2 * MAX_CURVE_DIVISOR_POINT_DIGITS
    intermediate_digits = (
        len(polynomial.polynomial.terms) * MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS
        + 2 * degree * point_component_digits
        + 4 * degree
        + 16
    )
    output_bytes = 4096 + point_count * (12 * MAX_BLOWUP_LABEL_CHARS + 3 * 192)
    if (
        work > MAX_CURVE_DIVISOR_WORK
        or intermediate_digits > MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS
        or output_bytes > MAX_CURVE_DIVISOR_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="plane_curve_divisor.work_bound",
            message="plane-curve divisor computation exceeds its admitted work, growth, or output envelope",
        )
    _reject_duplicate_points(surface)
    return degree, work, intermediate_digits


def _multiplicity_at_point(
    polynomial: RationalPolynomial,
    coordinates: tuple[CanonicalRational, CanonicalRational, CanonicalRational],
    degree: int,
) -> int:
    """Return the least nonzero Taylor degree in a chart containing the point."""
    fractions = tuple(value.as_fraction() for value in coordinates)
    chart = next(index for index, value in enumerate(fractions) if value)
    affine = tuple(
        value / fractions[chart]
        for index, value in enumerate(fractions)
        if index != chart
    )
    powers = tuple(
        tuple(value**exponent for exponent in range(degree + 1)) for value in affine
    )
    terms = tuple(
        (
            term.coefficient.as_fraction(),
            tuple(term.exponents[index] for index in range(3) if index != chart),
        )
        for term in polynomial.polynomial.terms
    )
    for order in range(degree + 1):
        for first_derivative in range(order + 1):
            second_derivative = order - first_derivative
            coefficient = Fraction(0)
            for scalar, exponents in terms:
                first_exp, second_exp = exponents
                if first_derivative > first_exp or second_derivative > second_exp:
                    continue
                coefficient += (
                    scalar
                    * comb(first_exp, first_derivative)
                    * comb(second_exp, second_derivative)
                    * powers[0][first_exp - first_derivative]
                    * powers[1][second_exp - second_derivative]
                )
            if coefficient:
                return order
    raise AssertionError("a nonzero homogeneous polynomial has nonzero chart expansion")


def plane_curve_strict_transform_class(
    request: PlaneCurveStrictTransformRequest,
) -> BlowupDivisorClass:
    """Compute ``dH - sum m_i E_i`` on the request's labelled P2 blow-up."""
    degree, _work, _height = _admit(request)
    canonical_surface = construct_surface(request.surface.points)
    variable_index = {
        variable: index
        for index, variable in enumerate(request.projective_coordinate_variables)
    }
    multiplicities = []
    for blowup_point in canonical_surface.points:
        ordered_coordinates = tuple(
            blowup_point.point.coordinates[variable_index[variable]]
            for variable in request.polynomial.variables
        )
        assert len(ordered_coordinates) == 3
        multiplicities.append(
            _multiplicity_at_point(request.polynomial, ordered_coordinates, degree)
        )
    return construct_divisor_class(canonical_surface, degree, tuple(multiplicities))


__all__ = ["plane_curve_strict_transform_class"]
