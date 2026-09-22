from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.dynamics.arithmetic.projective._kernel import (
    HomogeneousProjectiveMap,
    ProjectivePoint,
    apply_projective_map,
    compose_projective_maps,
    is_critical_point,
)
from jacobian.math.dynamics.arithmetic.projective._tools import (
    compute_critical_orbit,
    compute_projective_orbit,
)
from jacobian.math.dynamics.arithmetic.projective_models import (
    CriticalOrbitRequest,
    ProjectiveOrbitRequest,
)


def _r(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def _square_map() -> HomogeneousProjectiveMap:
    return HomogeneousProjectiveMap(
        degree=2,
        numerator=(_r(0), _r(0), _r(1)),
        denominator=(_r(1), _r(0), _r(0)),
    )


def test_homogeneous_application_transports_infinity_exactly() -> None:
    image = apply_projective_map(_square_map(), ProjectivePoint.infinity())
    assert image == ProjectivePoint.infinity()
    assert apply_projective_map(
        _square_map(), ProjectivePoint.from_fraction(Fraction(2))
    ).affine() == Fraction(4)


def test_homogeneous_composition_reconstructs_application() -> None:
    composed = compose_projective_maps(_square_map(), _square_map())
    point = ProjectivePoint.from_fraction(Fraction(2))
    assert apply_projective_map(composed, point).affine() == Fraction(16)


def test_composition_base_locus_uses_owner_error() -> None:
    outer = HomogeneousProjectiveMap(
        degree=1, numerator=(_r(0), _r(1)), denominator=(_r(0), _r(1))
    )
    inner = HomogeneousProjectiveMap(
        degree=1, numerator=(_r(0), _r(0)), denominator=(_r(0), _r(1))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        compose_projective_maps(outer, inner)
    assert (
        error.value.errors()[0]["type"] == "arithmetic_dynamics.projective_base_locus"
    )


def test_critical_derivative_admission_precedes_large_fraction_arithmetic() -> None:
    height = 10**8_191
    map_value = HomogeneousProjectiveMap(
        degree=2,
        numerator=(_r(1), _r(0), _r(0)),
        denominator=(
            CanonicalRational.from_integer_ratio(2 * height, 1),
            _r(-2),
            CanonicalRational.from_integer_ratio(1, height),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        is_critical_point(map_value, ProjectivePoint.from_fraction(Fraction(height)))


def test_critical_orbit_admits_the_aggregate_result_before_rows() -> None:
    height = 10**8_191
    map_value = HomogeneousProjectiveMap(
        degree=2,
        numerator=(_r(1), _r(0), _r(0)),
        denominator=(CanonicalRational.from_integer_ratio(height, 1), _r(0), _r(0)),
    )
    request = ProjectiveOrbitRequest(
        map=map_value, start=ProjectivePoint.from_fraction(Fraction(0)), max_steps=256
    )
    critical_request = CriticalOrbitRequest(
        map=request.map,
        critical_points=(request.start,) * 64,
        max_steps=request.max_steps,
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_critical_orbit(critical_request)


def test_step_bound_does_not_claim_nonperiodicity() -> None:
    result = compute_projective_orbit(
        ProjectiveOrbitRequest(
            map=HomogeneousProjectiveMap(
                degree=1,
                numerator=(_r(1), _r(1)),
                denominator=(_r(0), _r(1)),
            ),
            start=ProjectivePoint.from_fraction(Fraction(0)),
            max_steps=2,
        )
    )
    assert result.termination == "STEP_BOUND_REACHED"
    assert result.exact_period is False
    assert result.period is None


def test_orbit_growth_is_rejected_before_large_rational_construction() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        compute_projective_orbit(
            ProjectiveOrbitRequest(
                map=_square_map(),
                start=ProjectivePoint.from_fraction(Fraction(2)),
                max_steps=17,
            )
        )


def test_identity_orbit_is_admitted_at_the_step_boundary() -> None:
    result = compute_projective_orbit(
        ProjectiveOrbitRequest(
            map=HomogeneousProjectiveMap(
                degree=1,
                numerator=(_r(0), _r(1)),
                denominator=(_r(1), _r(0)),
            ),
            start=ProjectivePoint.from_fraction(Fraction(2)),
            max_steps=256,
        )
    )
    assert result.exact_period is True
    assert result.period == 1
    assert result.orbit[-1].affine() == Fraction(2)


def test_small_square_orbit_is_admitted_with_exact_value() -> None:
    result = compute_projective_orbit(
        ProjectiveOrbitRequest(
            map=_square_map(),
            start=ProjectivePoint.from_fraction(Fraction(2)),
            max_steps=5,
        )
    )
    assert result.termination == "STEP_BOUND_REACHED"
    assert result.orbit[-1].affine() == Fraction(2**32)


def test_degree_zero_constant_projective_map_is_closed() -> None:
    constant = HomogeneousProjectiveMap(
        degree=0, numerator=(_r(0),), denominator=(_r(1),)
    )
    assert (
        apply_projective_map(constant, ProjectivePoint.from_fraction(Fraction(3)))
        == ProjectivePoint.infinity()
    )


def test_projective_native_types_have_owner_errors() -> None:
    with pytest.raises(OperationDomainValidationError):
        apply_projective_map("not a map", ProjectivePoint.infinity())  # type: ignore[arg-type]
