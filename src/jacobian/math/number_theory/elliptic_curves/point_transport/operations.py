"""Exact point transport for short-Weierstrass isomorphisms over finite fields."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import FiniteFieldElement
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldIsomorphismResult,
    _coordinates,
    _curve_admit,
    _element,
    _multiply,
    _point_admit_with_curve,
)
from jacobian.math.number_theory.elliptic_curves.point_transport._models import (
    FiniteFieldPointTransportResult,
)

MAX_POINT_TRANSPORT_WORK = 2_000_000


def transport_point(
    isomorphism: FiniteFieldIsomorphismResult,
    point: FiniteFieldEllipticPoint,
) -> FiniteFieldPointTransportResult:
    """Apply ``(x,y) -> (u^2*x,u^3*y)`` to one point on an isomorphic model.

    The request's isomorphism is caller supplied. Recheck its common field,
    nonsingular endpoint curves, nonzero ``u``, and exact coefficient
    identities ``A_target=u^4*A_source`` and ``B_target=u^6*B_source`` before
    using it. Infinity maps to infinity. For affine points, the source point
    is re-admitted before applying the coordinate map.
    """

    if not isinstance(isomorphism, FiniteFieldIsomorphismResult):
        raise OperationDomainValidationError(
            location=("isomorphism",),
            code="elliptic_curve.finite_field.point_transport_isomorphism_type",
            message="isomorphism must be a finite-field short-model result value",
        )
    if not isinstance(point, FiniteFieldEllipticPoint):
        raise OperationDomainValidationError(
            location=("point",),
            code="elliptic_curve.finite_field.point_transport_point_type",
            message="point must be a finite-field elliptic point value",
        )
    source = _curve_admit(isomorphism.source)
    target = _curve_admit(isomorphism.target)
    if source.field != target.field:
        raise OperationDomainValidationError(
            location=("isomorphism", "target", "field"),
            code="elliptic_curve.finite_field.point_transport_field_mismatch",
            message="point transport requires one exact finite-field presentation",
        )
    if not isomorphism.isomorphic or isomorphism.scaling is None:
        raise OperationDomainValidationError(
            location=("isomorphism",),
            code="elliptic_curve.finite_field.point_transport_requires_isomorphism",
            message="point transport requires an isomorphic result with a scaling witness",
        )
    field = source.field
    degree = field.degree
    work = degree**3 * 64
    if work > MAX_POINT_TRANSPORT_WORK:
        raise OperationResourceAdmissionError(
            location=("isomorphism", "source", "field"),
            code="elliptic_curve.finite_field.point_transport_work_bound",
            message="coordinate transport exceeds its finite-field arithmetic envelope",
        )
    scaling = isomorphism.scaling
    try:
        scaling = FiniteFieldElement.model_validate(scaling.model_dump())
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("isomorphism", "scaling"),
            code="elliptic_curve.finite_field.point_transport_scaling_invalid",
            message="scaling witness must be a valid finite-field element",
        ) from exc
    if (
        scaling.presentation != field
        or any(
            type(value) is not int or not 0 <= value < field.characteristic
            for value in scaling.coordinates
        )
        or len(scaling.coordinates) != degree
    ):
        raise OperationDomainValidationError(
            location=("isomorphism", "scaling"),
            code="elliptic_curve.finite_field.point_transport_scaling_parent",
            message="scaling witness must be a canonical element of the common field",
        )
    u = _coordinates(scaling)
    zero = (0,) * degree
    if u == zero:
        raise OperationDomainValidationError(
            location=("isomorphism", "scaling"),
            code="elliptic_curve.finite_field.point_transport_zero_scaling",
            message="an isomorphism scaling must be nonzero",
        )
    u2 = _multiply(field, u, u)
    u3 = _multiply(field, u2, u)
    u4 = _multiply(field, u2, u2)
    u6 = _multiply(field, u4, u2)
    if _multiply(field, u4, _coordinates(source.coefficient_a)) != _coordinates(
        target.coefficient_a
    ) or _multiply(field, u6, _coordinates(source.coefficient_b)) != _coordinates(
        target.coefficient_b
    ):
        raise OperationDomainValidationError(
            location=("isomorphism", "scaling"),
            code="elliptic_curve.finite_field.point_transport_invalid_witness",
            message="scaling does not transport the source curve coefficients to the target",
        )

    source_point = _point_admit_with_curve(source, point)
    if source_point.at_infinity:
        target_point = FiniteFieldEllipticPoint.infinity(target)
    else:
        if source_point.x is None or source_point.y is None:
            raise RuntimeError("admitted affine point lost its coordinates")
        x = _multiply(field, u2, _coordinates(source_point.x))
        y = _multiply(field, u3, _coordinates(source_point.y))
        target_point = FiniteFieldEllipticPoint.affine(
            target, _element(field, x), _element(field, y)
        )
    admitted_isomorphism = FiniteFieldIsomorphismResult.model_construct(
        source=source,
        target=target,
        isomorphic=True,
        scaling=_element(field, u),
    )
    return FiniteFieldPointTransportResult._from_kernel(
        isomorphism=admitted_isomorphism,
        source_point=source_point,
        target_point=target_point,
    )


__all__ = ["transport_point"]
