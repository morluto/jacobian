"""Exact arithmetic for divisor classes on finite blow-ups of P2."""

from __future__ import annotations

from typing import NoReturn

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.blowup_p2._models import (
    MAX_BLOWUP_INTEGER_DIGITS,
    AdjunctionProfile,
    BlowupDivisorClass,
    BlowupP2Surface,
    BlowupPoint,
    IntersectionResult,
)
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.geometry.projective.coordinates.operations import (
    rational_projective_point,
)


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("surface",), code=f"blowup_p2.{code}", message=message
    )


def _integer_digits(value: int) -> int:
    """Count decimal digits without relying on Python's int-to-string guard."""

    return len(format_canonical_integer(abs(value)))


def _admit_surface_point(row: BlowupPoint) -> None:
    if (
        not isinstance(row.point, RationalProjectivePoint)
        or not isinstance(row.point.coordinates, tuple)
        or any(
            not isinstance(coordinate, CanonicalRational)
            for coordinate in row.point.coordinates
        )
    ):
        _reject(
            "point_shape",
            "blow-up points must retain canonical projective coordinates",
        )
    try:
        canonical_point = RationalProjectivePoint.model_validate(
            row.point.model_dump(mode="python")
        )
    except Exception:
        _reject(
            "point_shape", "blow-up points must retain valid projective coordinates"
        )
    if canonical_point.model_dump(mode="python") != row.point.model_dump(mode="python"):
        _reject("point_shape", "blow-up points must use canonical coordinates")
    coords = row.point.coordinates
    if all(c.as_fraction() == 0 for c in coords):
        _reject(
            "zero_projective_point",
            "blow-up points must be nonzero projective points",
        )
    canonical = rational_projective_point(coords).point
    if canonical != row.point:
        _reject(
            "point_not_canonical",
            "blow-up points must use first-nonzero-coordinate normalization",
        )


def _admit_surface(surface: BlowupP2Surface) -> None:
    if not isinstance(surface, BlowupP2Surface):
        _reject("surface_type", "expected a labelled BlowupP2Surface")
    if not isinstance(surface.points, tuple):
        _reject(
            "surface_points", "surface points must be a tuple of BlowupPoint values"
        )
    if len(surface.points) > 16:
        raise OperationResourceAdmissionError(
            location=("surface",),
            code="blowup_p2.point_count",
            message="too many blown-up points",
        )
    for row in surface.points:
        if not isinstance(row, BlowupPoint):
            _reject("point_type", "surface points must be BlowupPoint values")
        if type(row.label) is not str or not 1 <= len(row.label) <= 64:
            _reject("point_label", "blow-up labels must be bounded strings")
    labels = tuple(row.label for row in surface.points)
    if labels != tuple(sorted(labels)):
        _reject("point_labels", "blow-up point labels must be unique and sorted")
    try:
        payload = surface.model_dump(mode="python")
        canonical_surface = BlowupP2Surface.model_validate(payload)
    except Exception:
        _reject(
            "surface_shape",
            "surface and nested points must be canonically validated",
        )
    if canonical_surface.model_dump(mode="python") != payload:
        _reject(
            "surface_shape",
            "surface and nested points must be canonical carriers",
        )
    seen = set()
    for row in surface.points:
        if not isinstance(row, BlowupPoint):
            _reject("point_type", "surface points must be BlowupPoint values")
        if row.label in seen:
            _reject("duplicate_label", "blow-up labels must be unique")
        seen.add(row.label)
        _admit_surface_point(row)
    if len({p.point for p in surface.points}) != len(surface.points):
        _reject("duplicate_point", "blow-up points must be distinct projective points")


def construct_surface(points: tuple[BlowupPoint, ...]) -> BlowupP2Surface:
    if not isinstance(points, tuple) or any(
        not isinstance(point, BlowupPoint) for point in points
    ):
        _reject("points_type", "points must be a tuple of BlowupPoint values")
    if len(points) > 16:
        raise OperationResourceAdmissionError(
            location=("points",),
            code="blowup_p2.point_count",
            message="too many blown-up points",
        )
    canonical = []
    for row in points:
        # Native callers can forge a BlowupPoint with model_construct; validate
        # its nested point and label before dereferencing coordinates.
        if not isinstance(row, BlowupPoint):
            _reject("point_type", "surface points must be BlowupPoint values")
        if type(row.label) is not str or not 1 <= len(row.label) <= 64:
            _reject("point_label", "blow-up labels must be bounded strings")
        _admit_surface_point(row)
        try:
            result = rational_projective_point(row.point.coordinates)
        except (OperationDomainValidationError, OperationResourceAdmissionError):
            raise
        except Exception:
            _reject(
                "point_shape", "blow-up points must retain valid projective coordinates"
            )
        canonical.append(BlowupPoint(label=row.label, point=result.point))
    surface = BlowupP2Surface(points=tuple(sorted(canonical, key=lambda p: p.label)))
    _admit_surface(surface)
    return surface


def _admit_class(value: BlowupDivisorClass) -> None:
    if not isinstance(value, BlowupDivisorClass):
        _reject("class_type", "expected a labelled BlowupDivisorClass")
    if type(value.degree) is not int or not isinstance(value.multiplicities, tuple):
        _reject("class_shape", "divisor coefficients must be exact integer tuples")
    if any(type(entry) is not int for entry in value.multiplicities):
        _reject("class_shape", "divisor coefficients must be exact integers")
    _admit_surface(value.surface)
    coefficient_digits = (
        _integer_digits(value.degree),
        *(_integer_digits(entry) for entry in value.multiplicities),
    )
    if any(digits > MAX_BLOWUP_INTEGER_DIGITS for digits in coefficient_digits):
        raise OperationResourceAdmissionError(
            location=("divisor",),
            code="blowup_p2.integer_digits",
            message=(
                "divisor coefficients must fit the "
                f"{MAX_BLOWUP_INTEGER_DIGITS}-digit operation envelope"
            ),
        )
    if len(value.multiplicities) != len(value.surface.points):
        _reject(
            "multiplicity_axis",
            "multiplicity vector does not match the labelled point axis",
        )


def construct_divisor_class(
    surface: BlowupP2Surface, degree: int, multiplicities: tuple[int, ...]
) -> BlowupDivisorClass:
    _admit_surface(surface)
    if (
        type(degree) is not int
        or not isinstance(multiplicities, tuple)
        or any(type(entry) is not int for entry in multiplicities)
    ):
        _reject("class_shape", "degree and multiplicities must be exact integers")
    if _integer_digits(degree) > MAX_BLOWUP_INTEGER_DIGITS or any(
        _integer_digits(entry) > MAX_BLOWUP_INTEGER_DIGITS for entry in multiplicities
    ):
        raise OperationResourceAdmissionError(
            location=("divisor",),
            code="blowup_p2.integer_digits",
            message=(
                "divisor coefficients must fit the "
                f"{MAX_BLOWUP_INTEGER_DIGITS}-digit operation envelope"
            ),
        )
    if len(multiplicities) != len(surface.points):
        _reject(
            "multiplicity_axis", "one multiplicity is required for every labelled point"
        )
    return BlowupDivisorClass(
        surface=surface, degree=degree, multiplicities=multiplicities
    )


def _same_surface(left: BlowupDivisorClass, right: BlowupDivisorClass) -> None:
    _admit_class(left)
    _admit_class(right)
    if left.surface != right.surface:
        _reject(
            "parent_mismatch", "divisor classes belong to different labelled blow-ups"
        )


def _admit_intersection_growth(
    left: BlowupDivisorClass, right: BlowupDivisorClass
) -> None:
    # Products have additive decimal widths; summing at most 16 exceptional
    # terms needs two carry digits.  Check this before multiplying anything.
    product_digits = _integer_digits(left.degree) + _integer_digits(right.degree)
    for index in range(len(left.multiplicities)):
        product_digits = max(
            product_digits,
            _integer_digits(left.multiplicities[index])
            + _integer_digits(right.multiplicities[index]),
        )
    contribution_count = len(left.multiplicities)
    carry_digits = 0 if contribution_count == 0 else len(str(contribution_count))
    if product_digits + carry_digits > MAX_BLOWUP_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("intersection",),
            code="blowup_p2.intersection_growth",
            message=(
                "intersection intermediates exceed the "
                f"{MAX_BLOWUP_INTEGER_DIGITS}-digit operation envelope"
            ),
        )


def intersect_classes(
    left: BlowupDivisorClass, right: BlowupDivisorClass
) -> IntersectionResult:
    _same_surface(left, right)
    _admit_intersection_growth(left, right)
    products = tuple(
        (p.label, left.multiplicities[i] * right.multiplicities[i])
        for i, p in enumerate(left.surface.points)
    )
    value = left.degree * right.degree - sum(product for _, product in products)
    from jacobian.math.geometry.blowup_p2._models import IntersectionContribution

    return IntersectionResult(
        left=left,
        right=right,
        value=value,
        degree_product=left.degree * right.degree,
        exceptional_subtractions=tuple(
            IntersectionContribution(label=label, product=product)
            for label, product in products
        ),
    )


def canonical_class(surface: BlowupP2Surface) -> BlowupDivisorClass:
    _admit_surface(surface)
    return BlowupDivisorClass(
        surface=surface, degree=-3, multiplicities=tuple(-1 for _ in surface.points)
    )


def adjunction_profile(divisor: BlowupDivisorClass) -> AdjunctionProfile:
    _admit_class(divisor)
    canonical = canonical_class(divisor.surface)
    self_value = intersect_classes(divisor, divisor).value
    canonical_value = intersect_classes(divisor, canonical).value
    pairing = self_value + canonical_value
    if pairing % 2:
        _reject(
            "nonintegral_genus",
            "adjunction pairing does not yield an integral arithmetic genus",
        )
    return AdjunctionProfile(
        divisor=divisor,
        canonical=canonical,
        self_intersection=self_value,
        canonical_intersection=canonical_value,
        adjunction_pairing=pairing,
        arithmetic_genus=1 + pairing // 2,
    )


__all__ = [
    "adjunction_profile",
    "canonical_class",
    "construct_divisor_class",
    "construct_surface",
    "intersect_classes",
]
