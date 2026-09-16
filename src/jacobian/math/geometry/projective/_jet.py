"""Exact first jets of homogeneous plane curves at rational points."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"projective_geometry.{reason}", message)


MAX_JET_DEGREE = 64
"""Maximum total degree of an admitted homogeneous plane polynomial."""

MAX_JET_TERMS = 256
"""Maximum monomial terms of an admitted homogeneous plane polynomial."""

MAX_JET_COEFFICIENT_DIGITS = 1_024
"""Per-component digit bound for polynomial and point entries."""


JetStatus = Literal["ON_CURVE_SIMPLE", "ON_CURVE_SINGULAR_OR_HIGHER", "OFF_CURVE"]


class PlaneCurveJetRequest(StrictModel):
    """Compute the exact first jet of a plane curve at a projective point.

    The polynomial must be homogeneous of positive degree in exactly three
    variables over QQ, and the point must be a nonzero rational projective
    triple. Nonhomogeneous input and zero points are rejected before any
    evaluation.
    """

    polynomial: RationalPolynomial = Field(
        description=(
            "Homogeneous polynomial in exactly three variables over QQ with "
            f"positive degree at most {MAX_JET_DEGREE} and at most "
            f"{MAX_JET_TERMS} terms."
        )
    )
    point: RationalProjectivePoint = Field(
        description=(
            "Nonzero rational projective triple; nonzero rescalings denote "
            "the same point and preserve the result."
        )
    )


class PlaneCurveJetResult(StrictModel):
    """Exact value, gradient, chart replay, and incidence state of one jet."""

    value: CanonicalRational
    partials: tuple[CanonicalRational, CanonicalRational, CanonicalRational]
    chart_index: int = Field(ge=0, le=2)
    affine_point: tuple[CanonicalRational, CanonicalRational]
    status: JetStatus

    @model_validator(mode="after")
    def require_status_consistency(self) -> Self:
        is_zero = self.value.as_fraction() == 0
        gradient_zero = all(p.as_fraction() == 0 for p in self.partials)
        if self.status == "OFF_CURVE":
            if is_zero:
                raise _validation_error(
                    "jet_status_value",
                    "OFF_CURVE requires a nonzero polynomial value",
                )
        elif is_zero is False:
            raise _validation_error(
                "jet_status_value",
                "on-curve states require a zero polynomial value",
            )
        elif (self.status == "ON_CURVE_SIMPLE") == gradient_zero:
            raise _validation_error(
                "jet_status_gradient",
                "ON_CURVE_SIMPLE requires a nonzero gradient and "
                "ON_CURVE_SINGULAR_OR_HIGHER a zero gradient",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalRational,
        partials: tuple[CanonicalRational, CanonicalRational, CanonicalRational],
        chart_index: int,
        affine_point: tuple[CanonicalRational, CanonicalRational],
        status: JetStatus,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its evaluation."""

        return cls.model_construct(
            value=value,
            partials=partials,
            chart_index=chart_index,
            affine_point=affine_point,
            status=status,
        )


def _reject(location: str, code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _refuse(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomial",),
        code=code,
        message=message,
    )


def _admit_plane_curve_jet(
    polynomial: RationalPolynomial, point: RationalProjectivePoint
) -> int:
    """Enforce the shared envelope; return the homogeneous degree."""

    if not isinstance(polynomial, RationalPolynomial):
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.polynomial_not_rational",
            "plane-curve input must be a sparse rational polynomial value",
        )
    if not isinstance(point, RationalProjectivePoint):
        _reject(
            "point",
            "projective_geometry.plane_curve.point_not_projective",
            "plane-curve point must be a rational projective point value",
        )
    assert isinstance(polynomial, RationalPolynomial)
    assert isinstance(point, RationalProjectivePoint)
    if polynomial.domain != "QQ":
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.coefficient_domain_not_qq",
            "plane-curve coefficients must lie over QQ",
        )
    if len(polynomial.variables) != 3:
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.not_a_plane_curve",
            "plane-curve polynomials use exactly three variables",
        )
    if len(point.coordinates) != 3:
        _reject(
            "point",
            "projective_geometry.plane_curve.point_not_a_plane_point",
            "plane-curve points are projective triples",
        )
    terms = polynomial.polynomial.terms
    if not terms:
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.zero_polynomial",
            "the zero polynomial has no plane-curve jet",
        )
    degrees = {sum(term.exponents) for term in terms}
    if len(degrees) != 1:
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.not_homogeneous",
            "plane-curve polynomials must be homogeneous; "
            "nonhomogeneous input is rejected",
        )
    degree = next(iter(degrees))
    if degree < 1:
        _reject(
            "polynomial",
            "projective_geometry.plane_curve.constant_polynomial",
            "plane-curve polynomials have positive degree",
        )
    try:
        for term in terms:
            require_bounded_rational(
                term.coefficient,
                max_digits=MAX_JET_COEFFICIENT_DIGITS,
                label="jet polynomial coefficient",
            )
        for coordinate in point.coordinates:
            require_bounded_rational(
                coordinate,
                max_digits=MAX_JET_COEFFICIENT_DIGITS,
                label="jet point coordinate",
            )
    except ValueError as exc:
        _refuse(
            "projective_geometry.plane_curve.coefficient_over_envelope",
            str(exc),
        )
    if len(terms) > MAX_JET_TERMS:
        _refuse(
            "projective_geometry.plane_curve.term_count_over_envelope",
            f"plane-curve terms exceed the {MAX_JET_TERMS}-term envelope",
        )
    if degree > MAX_JET_DEGREE:
        _refuse(
            "projective_geometry.plane_curve.degree_over_envelope",
            f"plane-curve degree exceeds the {MAX_JET_DEGREE} envelope",
        )
    coords = [c.as_fraction() for c in point.coordinates]
    if all(c == 0 for c in coords):
        _reject(
            "point",
            "projective_geometry.plane_curve.zero_point",
            "projective points must have a nonzero coordinate",
        )
    return degree


def plane_curve_first_jet(
    polynomial: RationalPolynomial, point: RationalProjectivePoint
) -> PlaneCurveJetResult:
    """Evaluate the exact value and gradient of a homogeneous plane curve.

    The chart is the first nonzero point coordinate; the affine replay
    divides the remaining coordinates by the chart coordinate and checks
    ``F(p) = chart^d * F_aff(affine)`` exactly. ``ON_CURVE_SIMPLE`` holds
    iff the value vanishes with nonzero gradient; vanishing value with zero
    gradient is the unresolved ``ON_CURVE_SINGULAR_OR_HIGHER`` branch.
    """

    degree = _admit_plane_curve_jet(polynomial, point)
    assert isinstance(polynomial, RationalPolynomial)
    assert isinstance(point, RationalProjectivePoint)
    coords = [c.as_fraction() for c in point.coordinates]
    monomials = [
        (term.coefficient.as_fraction(), tuple(term.exponents))
        for term in polynomial.polynomial.terms
    ]

    def evaluate(at: list[Fraction]) -> Fraction:
        total = Fraction(0)
        for coefficient, exponents in monomials:
            term = coefficient
            for base, exponent in zip(at, exponents, strict=True):
                term *= base**exponent
            total += term
        return total

    value = evaluate(coords)
    partials: list[Fraction] = []
    for axis in range(3):
        total = Fraction(0)
        for coefficient, exponents in monomials:
            if exponents[axis] == 0:
                continue
            term = coefficient * exponents[axis]
            for other in range(3):
                power = exponents[other] - (1 if other == axis else 0)
                term *= coords[other] ** power
            total += term
        partials.append(total)
    chart = next(i for i, c in enumerate(coords) if c != 0)
    scale = coords[chart]
    affine = [coords[j] / scale for j in range(3) if j != chart]
    # Dehomogenized replay: F(p) = chart^d * F_aff(affine).
    replay_at = list(affine)
    replay_at.insert(chart, Fraction(1))
    replay = evaluate(replay_at) * scale**degree
    if replay != value:
        _reject(
            "point",
            "projective_geometry.plane_curve.chart_replay_failed",
            "dehomogenized chart replay must reproduce the homogeneous value",
        )
    if value != 0:
        status: JetStatus = "OFF_CURVE"
    elif any(p != 0 for p in partials):
        status = "ON_CURVE_SIMPLE"
    else:
        status = "ON_CURVE_SINGULAR_OR_HIGHER"
    return PlaneCurveJetResult._from_kernel(
        value=CanonicalRational.from_fraction(value),
        partials=(
            CanonicalRational.from_fraction(partials[0]),
            CanonicalRational.from_fraction(partials[1]),
            CanonicalRational.from_fraction(partials[2]),
        ),
        chart_index=chart,
        affine_point=(
            CanonicalRational.from_fraction(affine[0]),
            CanonicalRational.from_fraction(affine[1]),
        ),
        status=status,
    )


def _run_plane_curve_first_jet(request: PlaneCurveJetRequest) -> PlaneCurveJetResult:
    return plane_curve_first_jet(request.polynomial, request.point)


FIRST_JET_OPERATION = MathTool(
    operation_id="projective_geometry.plane_curve.first_jet.compute",
    title="Compute the exact first jet of a plane curve at a rational point",
    description=(
        "For a bounded homogeneous polynomial in three variables over QQ and a "
        "nonzero rational projective triple, return the exact value, all three "
        "first partials, the first-nonzero-coordinate chart with "
        "dehomogenized replay, and incidence state ON_CURVE_SIMPLE, "
        "ON_CURVE_SINGULAR_OR_HIGHER, or OFF_CURVE. Nonhomogeneous input and "
        "zero points are rejected; nonzero rescalings of point or polynomial "
        "preserve the conclusion."
    ),
    request_type=PlaneCurveJetRequest,
    result_type=PlaneCurveJetResult,
    run=_run_plane_curve_first_jet,
    tags=("projective-geometry", "plane-curve", "jet", "exact"),
    discovery_terms=(
        "plane curve singularity jet",
        "homogeneous polynomial projective point",
        "simple point of a plane curve",
    ),
    examples=(
        OperationExample(
            name="cusp_curve_at_origin_chart",
            description=(
                "Evaluate the exact jet of the cuspidal cubic y^2 z - x^3 at "
                "[0:0:1]; the polynomial must be homogeneous and the point "
                "nonzero."
            ),
            input={
                "polynomial": {
                    "domain": "QQ",
                    "variables": ["x", "y", "z"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "-1", "den": "1"},
                                "exponents": [3, 0, 0],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0, 2, 1],
                            },
                        ]
                    },
                },
                "point": {
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ]
                },
            },
        ),
    ),
)

__all__ = [
    "FIRST_JET_OPERATION",
    "PlaneCurveJetRequest",
    "PlaneCurveJetResult",
    "plane_curve_first_jet",
]
