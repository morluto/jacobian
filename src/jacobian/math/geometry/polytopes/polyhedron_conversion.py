"""Exact rational H-to-V conversion for general polyhedra."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    MAX_DD_WEIGHTED_HEIGHT_WORK,
    PolyhedralConversionAdmissionError,
    _admit_halfspaces_to_generators,
    _HalfspaceConversionAdmission,
    _halfspaces_to_generators_from_admission,
)
from jacobian.math.geometry.polytopes._rational_geometry import (
    PolyhedralConversionError,
)
from jacobian.math.geometry.polytopes.values import (
    MAX_RATIONAL_POLYHEDRON_GENERATORS,
    MAX_RATIONAL_POLYHEDRON_INEQUALITIES,
    MAX_RATIONAL_POLYTOPE_DIMENSION,
    RationalAffineHalfspace,
    RationalHPolyhedron,
    RationalPolyhedronVPresentation,
)

MAX_POLYHEDRON_V_RESULT_BYTES = CanonicalLimits().max_output_bytes
"""Canonical JSON transport egress ceiling used for conservative preflight."""

MAX_POLYHEDRON_INPUT_COMPONENT_DIGITS = 1_024
"""Per-numerator/denominator input ceiling for practical exact DD work."""


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _refuse(code: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("polyhedron",), code=code, message=message
    )


def _empty(source: RationalHPolyhedron) -> RationalPolyhedronVPresentation:
    return RationalPolyhedronVPresentation(
        space=source.space,
        points=(),
        rays=(),
        lineality=(),
        empty=True,
        affine_dimension=-1,
    )


def _decimal_digits(value: int) -> int:
    """Conservative decimal digit bound without formatting a large integer."""

    bits = abs(value).bit_length()
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _validate_source(source: RationalHPolyhedron) -> None:
    if not isinstance(source, RationalHPolyhedron):
        _reject(
            "polyhedron",
            "polyhedron.h_to_v.invalid_source",
            "source must be a canonical rational H-polyhedron",
        )
    try:
        payload = source.model_dump(mode="python")
        canonical = RationalHPolyhedron.model_validate(payload)
    except Exception:
        _reject(
            "polyhedron",
            "polyhedron.h_to_v.malformed_source",
            "H-presentation must contain canonical axes and rational rows",
        )
    if canonical.model_dump(mode="python") != payload:
        _reject(
            "polyhedron",
            "polyhedron.h_to_v.malformed_source",
            "H-presentation is not canonical",
        )


def _collect_rows(
    source: RationalHPolyhedron,
) -> tuple[list[tuple[list[Fraction], Fraction]], bool]:
    """Convert admitted scalar values and classify constant inequalities."""

    rows: list[tuple[list[Fraction], Fraction]] = []
    component_digits = 1
    has_inconsistent_constant = False
    for inequality in source.inequalities:
        if not isinstance(inequality, RationalAffineHalfspace):
            _reject(
                "polyhedron",
                "polyhedron.h_to_v.malformed_row",
                "every inequality must be a canonical affine halfspace",
            )
        coefficients = [component.as_fraction() for component in inequality.normal]
        bound = inequality.bound.as_fraction()
        for component in (*inequality.normal, inequality.bound):
            component_digits = max(
                component_digits,
                _decimal_digits(component.num),
                _decimal_digits(component.den),
            )
            if component_digits > MAX_POLYHEDRON_INPUT_COMPONENT_DIGITS:
                _refuse(
                    "polyhedron.h_to_v.coefficient_height",
                    "input coefficient exceeds the admitted 1,024-digit envelope",
                )
        if all(coefficient == 0 for coefficient in coefficients):
            has_inconsistent_constant |= bound < 0
            continue
        rows.append((coefficients, bound))
    return rows, has_inconsistent_constant


def _admit_expansion(
    rows: list[tuple[list[Fraction], Fraction]], dimension: int
) -> _HalfspaceConversionAdmission:
    """Preflight DD once, then prove output growth and wire-size limits."""

    try:
        admission = _admit_halfspaces_to_generators(
            rows,
            dimension,
            maximum_result_minor_digits=32_768,
        )
        minor_digits = admission.minor_digits
        maximum_rays = admission.maximum_rays
        # The kernel's generator count is bounded by the section's facet bound;
        # lineality adds at most d vectors. Each rational coordinate has a
        # numerator and denominator of at most minor_digits decimal digits.
        total_vectors = maximum_rays + dimension
        per_component_bytes = 2 * minor_digits + 32
        estimated_bytes = (
            total_vectors * max(1, dimension) * per_component_bytes
            + total_vectors * (dimension + 2)
            + 4_096
        )
        if estimated_bytes > MAX_POLYHEDRON_V_RESULT_BYTES:
            _refuse(
                "polyhedron.h_to_v.result_bytes",
                f"conservative V-result bound {estimated_bytes} exceeds {MAX_POLYHEDRON_V_RESULT_BYTES} bytes",
            )
        if maximum_rays > MAX_RATIONAL_POLYHEDRON_GENERATORS:
            _refuse(
                "polyhedron.h_to_v.generator_count",
                "worst-case generator count exceeds the serializable value bound",
            )
        # Difference rows can combine two generator-coordinate denominators;
        # rowwise denominator clearing across d axes is therefore bounded by
        # 2*d*minor_digits plus the numerator and determinant carries.
        rank_digits = 2 * max(1, dimension) * minor_digits + dimension + 2
        rank_rows = maximum_rays + dimension
        rank_weight = max(1, rank_rows) * max(1, dimension) ** 3 * rank_digits**2
        if rank_weight > MAX_DD_WEIGHTED_HEIGHT_WORK:
            _refuse(
                "polyhedron.h_to_v.rank_work",
                "affine-dimension rank work exceeds the admitted height-weighted bound",
            )
    except PolyhedralConversionAdmissionError as exc:
        _refuse("polyhedron.h_to_v.kernel_admission", str(exc))
    return admission


def _admit_source(
    source: RationalHPolyhedron,
) -> _HalfspaceConversionAdmission | None:
    _validate_source(source)
    dimension = len(source.space.axes)
    if dimension > MAX_RATIONAL_POLYTOPE_DIMENSION:
        _refuse(
            "polyhedron.h_to_v.dimension",
            "ambient dimension exceeds the admitted envelope",
        )
    if len(source.inequalities) > MAX_RATIONAL_POLYHEDRON_INEQUALITIES:
        _refuse(
            "polyhedron.h_to_v.inequality_count",
            "inequality count exceeds the admitted envelope",
        )

    rows, has_inconsistent_constant = _collect_rows(source)
    if has_inconsistent_constant:
        # This is a direct exact contradiction, with a constant-size result.
        return None
    return _admit_expansion(rows, dimension)


def halfspaces_to_v_presentation(
    source: RationalHPolyhedron,
) -> RationalPolyhedronVPresentation:
    """Return exact finite points, oriented recession rays, and lineality."""

    admission = _admit_source(source)
    if admission is None:
        return _empty(source)
    try:
        converted = _halfspaces_to_generators_from_admission(admission)
    except (PolyhedralConversionAdmissionError, PolyhedralConversionError) as exc:
        _refuse("polyhedron.h_to_v.kernel_failure", str(exc))
    if converted.empty:
        return _empty(source)
    points = tuple(
        tuple(CanonicalRational.from_fraction(value) for value in point)
        for point in converted.vertices
    )
    rays = tuple(
        tuple(CanonicalRational.from_fraction(Fraction(value)) for value in ray)
        for ray in converted.recession_rays
    )
    lineality = tuple(
        tuple(CanonicalRational.from_fraction(Fraction(value)) for value in basis)
        for basis in converted.lineality_basis
    )
    return RationalPolyhedronVPresentation._from_kernel(
        space=source.space,
        points=points,
        rays=rays,
        lineality=lineality,
        empty=converted.empty,
        affine_dimension=converted.affine_dimension,
    )


__all__ = ["MAX_POLYHEDRON_V_RESULT_BYTES", "halfspaces_to_v_presentation"]
