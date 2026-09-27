"""Exact plane-curve multiplicities and strict-transform classes."""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import combinations
from math import comb, gcd
from typing import NoReturn, TypeGuard

from jacobian._exact import CanonicalRational
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.algebraic_curves.divisor_classes._models import (
    MAX_CURVE_DIVISOR_ALLOCATION_UNITS,
    MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS,
    MAX_CURVE_DIVISOR_DEGREE,
    MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS,
    MAX_CURVE_DIVISOR_POINT_DIGITS,
    MAX_CURVE_DIVISOR_TERMS,
    MAX_CURVE_DIVISOR_WORK,
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
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_BLOWUP_LABEL_CHARS = 64
_VARIABLE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}\Z")


def _domain(reason: str, message: str, location: tuple[str | int, ...]) -> NoReturn:
    raise OperationDomainValidationError(
        location=location, code=f"plane_curve_divisor.{reason}", message=message
    )


def _admit_polynomial(polynomial: RationalPolynomial) -> int:  # noqa: C901
    if not isinstance(polynomial, RationalPolynomial):
        _domain(
            "polynomial_type",
            "expected a canonical rational plane polynomial",
            ("polynomial",),
        )
    domain = getattr(polynomial, "domain", None)
    variables = getattr(polynomial, "variables", None)
    sparse = getattr(polynomial, "polynomial", None)
    if (
        domain != "QQ"
        or not isinstance(variables, tuple)
        or len(variables) != 3
        or any(
            type(variable) is not str or _VARIABLE_PATTERN.fullmatch(variable) is None
            for variable in variables
        )
        or len(set(variables)) != 3
        or not isinstance(sparse, SparseRationalPolynomial)
        or not isinstance(sparse.terms, tuple)
        or not sparse.terms
    ):
        _domain(
            "polynomial_axis",
            "a plane curve requires a nonzero ternary polynomial",
            ("polynomial",),
        )
    if len(sparse.terms) > MAX_CURVE_DIVISOR_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_curve_divisor.term_bound",
            message="plane-curve source admits at most 64 terms",
        )
    for index, term in enumerate(sparse.terms):
        if not isinstance(term, RationalPolynomialTerm):
            _domain(
                "term_shape",
                "homogeneous curve terms must be canonical rational terms",
                ("polynomial", index),
            )
        term_exponents = getattr(term, "exponents", None)
        if not isinstance(term_exponents, tuple) or len(term_exponents) != 3:
            _domain(
                "term_shape",
                "homogeneous curve terms must use three bounded exponents",
                ("polynomial", index),
            )
        if any(
            type(exponent) is not int or exponent < 0 for exponent in term_exponents
        ):
            _domain(
                "term_shape",
                "homogeneous curve exponents must be nonnegative integers",
                ("polynomial", index),
            )
        coefficient = getattr(term, "coefficient", None)
        if not isinstance(coefficient, CanonicalRational):
            _domain(
                "coefficient_type",
                "curve coefficients must be canonical rationals",
                ("polynomial", index),
            )
        if not _has_rational_components(coefficient):
            _domain(
                "coefficient_type",
                "curve coefficients must be canonical rationals",
                ("polynomial", index),
            )
        if (
            _rational_component_digits(coefficient)
            > MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("polynomial", index),
                code="plane_curve_divisor.coefficient_bound",
                message="curve coefficients are limited to 32 decimal digits",
            )
        if not _is_canonical_rational(coefficient):
            _domain(
                "coefficient_type",
                "curve coefficients must be canonical rationals",
                ("polynomial", index),
            )
        if coefficient.num == 0:
            _domain(
                "zero_term",
                "zero polynomial terms must be omitted",
                ("polynomial", index),
            )
    exponents = tuple(term.exponents for term in sparse.terms)
    if exponents != tuple(sorted(exponents, reverse=True)) or len(
        set(exponents)
    ) != len(exponents):
        _domain(
            "term_order",
            "curve terms must have unique exponents in descending lexicographic order",
            ("polynomial",),
        )
    degrees = {sum(term.exponents) for term in sparse.terms}
    if len(degrees) != 1:
        _domain(
            "inhomogeneous", "the curve polynomial must be homogeneous", ("polynomial",)
        )
    degree = next(iter(degrees))
    if degree > MAX_CURVE_DIVISOR_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_curve_divisor.degree_bound",
            message="the plane-curve degree must be at most 12",
        )
    if degree == 0:
        _domain("degree_zero", "a nonconstant polynomial is required", ("polynomial",))
    if degree > MAX_CURVE_DIVISOR_DEGREE:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="plane_curve_divisor.degree_bound",
            message="the plane-curve degree must be 1..12",
        )
    return degree


def _has_rational_components(value: CanonicalRational) -> bool:
    numerator = getattr(value, "num", None)
    denominator = getattr(value, "den", None)
    return type(numerator) is int and type(denominator) is int and denominator > 0


def _is_canonical_rational(value: object) -> TypeGuard[CanonicalRational]:
    if not isinstance(value, CanonicalRational) or not _has_rational_components(value):
        return False
    numerator = value.num
    denominator = value.den
    return gcd(abs(numerator), denominator) == 1 and (
        numerator != 0 or denominator == 1
    )


def _rational_component_digits(value: CanonicalRational) -> int:
    limit = 10**MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS
    # Canonical carriers can be forged by native callers; reject magnitude
    # before exact formatting, which can be expensive for enormous integers.
    if abs(value.num) >= limit or value.den >= limit:
        return MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS + 1
    return max(decimal_digit_width(value.num), decimal_digit_width(value.den))


def _admit_surface_points(surface: BlowupP2Surface) -> int:
    points = getattr(surface, "points", None)
    if not isinstance(points, tuple):
        _domain("point_shape", "surface points must be a tuple", ("surface",))
    if len(points) > 16:
        raise OperationResourceAdmissionError(
            location=("surface",),
            code="plane_curve_divisor.point_bound",
            message="at most 16 blow-up points are admitted",
        )
    # Validate every point's raw arithmetic before canonicalizing the surface.
    for point_index, row in enumerate(points):
        row_label = getattr(row, "label", None)
        row_point = getattr(row, "point", None)
        if not isinstance(row, BlowupPoint) or not isinstance(
            row_point, RationalProjectivePoint
        ):
            _domain(
                "point_shape",
                "surface entries must be labelled rational projective points",
                ("surface", point_index),
            )
        if (
            type(row_label) is not str
            or not 1 <= len(row_label) <= MAX_BLOWUP_LABEL_CHARS
        ):
            _domain(
                "point_label",
                "blow-up labels must be 1..64 characters",
                ("surface", point_index),
            )
        coordinates = getattr(row_point, "coordinates", None)
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
            if not _has_rational_components(coordinate):
                _domain(
                    "point_scalar",
                    "point coordinates must be canonical rationals",
                    ("surface", point_index, coordinate_index),
                )
            if _rational_component_digits(coordinate) > MAX_CURVE_DIVISOR_POINT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("surface", point_index, coordinate_index),
                    code="plane_curve_divisor.point_height",
                    message="projective point components are limited to 16 decimal digits",
                )
            if not _is_canonical_rational(coordinate):
                _domain(
                    "point_scalar",
                    "point coordinates must be canonical rationals",
                    ("surface", point_index, coordinate_index),
                )
        if all(coordinate.as_fraction() == 0 for coordinate in coordinates):
            _domain(
                "zero_point",
                "blow-up points must be nonzero projective points",
                ("surface", point_index),
            )
    return len(points)


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


def _admit(
    polynomial: RationalPolynomial,
    surface: BlowupP2Surface,
    projective_coordinate_variables: tuple[str, str, str],
) -> int:
    """Admit the complete polynomial, point family, and retained result first."""
    if not isinstance(surface, BlowupP2Surface):
        _domain(
            "surface_type",
            "expected a canonical labelled blow-up surface",
            ("surface",),
        )
    degree = _admit_polynomial(polynomial)
    polynomial_variables = getattr(polynomial, "variables", ())
    if (
        not isinstance(projective_coordinate_variables, tuple)
        or len(projective_coordinate_variables) != 3
        or any(
            type(variable) is not str for variable in projective_coordinate_variables
        )
        or len(set(projective_coordinate_variables)) != 3
        or tuple(sorted(projective_coordinate_variables))
        != tuple(sorted(polynomial_variables))
    ):
        _domain(
            "coordinate_transport",
            "projective coordinate variables must permute the polynomial axis",
            ("projective_coordinate_variables",),
        )
    point_count = _admit_surface_points(surface)
    labels = tuple(point.label for point in surface.points)
    if labels != tuple(sorted(labels)):
        _domain(
            "surface_labels",
            "blow-up point labels must be in canonical sorted order",
            ("surface",),
        )
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
    allocation_units = (
        8 * point_count + 4 * len(polynomial.polynomial.terms) + 4 * (degree + 1)
    )
    if work > MAX_CURVE_DIVISOR_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="plane_curve_divisor.work_bound",
            message="plane-curve divisor computation exceeds its admitted work bound",
        )
    if intermediate_digits > MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="plane_curve_divisor.intermediate_growth_bound",
            message="plane-curve divisor computation exceeds its admitted coefficient-growth bound",
        )
    if allocation_units > MAX_CURVE_DIVISOR_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="plane_curve_divisor.allocation_bound",
            message="plane-curve divisor computation exceeds its admitted allocation bound",
        )
    _reject_duplicate_points(surface)
    return degree


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
    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)


def plane_curve_strict_transform_class(
    polynomial: RationalPolynomial,
    surface: BlowupP2Surface,
    projective_coordinate_variables: tuple[str, str, str],
) -> BlowupDivisorClass:
    """Compute ``dH - sum m_i E_i`` on the request's labelled P2 blow-up."""
    degree = _admit(polynomial, surface, projective_coordinate_variables)
    canonical_surface = construct_surface(surface.points)
    variable_index = {
        variable: index
        for index, variable in enumerate(projective_coordinate_variables)
    }
    multiplicities = []
    for blowup_point in canonical_surface.points:
        ordered_coordinates = tuple(
            blowup_point.point.coordinates[variable_index[variable]]
            for variable in polynomial.variables
        )
        if len(ordered_coordinates) != 3:
            _domain(
                "coordinate_transport",
                "projective coordinate transport did not produce three coordinates",
                ("projective_coordinate_variables",),
            )
        multiplicities.append(
            _multiplicity_at_point(polynomial, ordered_coordinates, degree)
        )
    return construct_divisor_class(canonical_surface, degree, tuple(multiplicities))


__all__ = ["plane_curve_strict_transform_class"]
