"""Exact connected components of bounded plane semialgebraic sets."""

from __future__ import annotations

import shutil
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.real_algebra import (
    compute_plane_component_profile,
    verify_plane_component_profile,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    IsolatedRealPlanePoint,
    PlaneComponentProfileComputed,
    PlaneComponentProfileRequest,
    PlaneComponentProfileResult,
    PlaneSemialgebraicSet,
    PlaneSign,
    PlaneSignCondition,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_samples import (
    isolated_plane_point_coordinates,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

pytestmark = [
    pytest.mark.requires_backend("qepcad"),
    pytest.mark.skipif(
        shutil.which("qepcad") is None,
        reason="the exact plane-component backend is unavailable",
    ),
]


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial(
    terms: tuple[tuple[int, tuple[int, int]], ...],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(coefficient=_q(coefficient), exponents=exponents)
                for coefficient, exponents in terms
            )
        ),
    )


def _real_value(value: Fraction) -> RealAlgebraicValue:
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=(str(value.denominator), str(-value.numerator)),
        real_root_index=0,
    )


def _rational_point(x: int | Fraction, y: int | Fraction) -> IsolatedRealPlanePoint:
    x = Fraction(x)
    y = Fraction(y)
    return IsolatedRealPlanePoint(
        axis=("x", "y"),
        coordinates=(_real_value(x), _real_value(y)),
        isolating_box=RationalBox(
            domain="QQ",
            variables=("x", "y"),
            intervals=(
                ClosedRationalInterval(lower=_q(x), upper=_q(x)),
                ClosedRationalInterval(lower=_q(y), upper=_q(y)),
            ),
        ),
    )


def _set(
    polynomials: tuple[RationalPolynomial, ...],
    rows: tuple[tuple[PlaneSign, ...], ...],
) -> PlaneSemialgebraicSet:
    return PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=polynomials,
        sign_conditions=tuple(PlaneSignCondition(signs=row) for row in rows),
    )


def _computed(
    semialgebraic_set: PlaneSemialgebraicSet,
    samples: tuple[IsolatedRealPlanePoint, ...] = (),
) -> PlaneComponentProfileComputed:
    result = compute_plane_component_profile(semialgebraic_set, samples)
    assert result.outcome.status == "COMPUTED", result.outcome
    assert isinstance(result.outcome, PlaneComponentProfileComputed)
    assert (
        PlaneComponentProfileResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    representative_encodings = tuple(
        component.representative.model_dump_json()
        for component in result.outcome.components
    )
    assert len(representative_encodings) == len(set(representative_encodings))
    for component in result.outcome.components:
        _assert_point_satisfies_source(
            component.representative,
            semialgebraic_set,
        )
    return result.outcome


def _assert_point_satisfies_source(
    point: IsolatedRealPlanePoint,
    semialgebraic_set: PlaneSemialgebraicSet,
) -> None:
    coordinates = isolated_plane_point_coordinates(point)
    signs: list[PlaneSign] = []
    for polynomial in semialgebraic_set.polynomials:
        converted = rational_polynomial_to_sympy(polynomial)
        value = converted.as_expr().subs(
            dict(zip(converted.gens, coordinates, strict=True))
        )
        if value == 0 or value.is_zero is True:
            signs.append(PlaneSign.ZERO)
        elif value.is_positive is True:
            signs.append(PlaneSign.POSITIVE)
        else:
            assert value.is_negative is True
            signs.append(PlaneSign.NEGATIVE)
    assert tuple(signs) in {
        condition.signs for condition in semialgebraic_set.sign_conditions
    }


def _radial_level(radius_squared: int) -> RationalPolynomial:
    return _polynomial(((1, (2, 0)), (1, (0, 2)), (-radius_squared, (0, 0))))


def test_annulus_complement_has_two_components_and_binds_samples() -> None:
    inner = _radial_level(1)
    outer = _radial_level(4)
    # (inner * outer) > 0, expressed by its complete factor sign table.
    semialgebraic_set = _set(
        (inner, outer),
        (
            (PlaneSign.NEGATIVE, PlaneSign.NEGATIVE),
            (PlaneSign.POSITIVE, PlaneSign.POSITIVE),
        ),
    )
    baseline = _computed(semialgebraic_set)
    outcome = _computed(
        semialgebraic_set,
        (
            _rational_point(0, 0),
            _rational_point(Fraction(1, 2), 0),
            _rational_point(3, 0),
            _rational_point(Fraction(3, 2), 0),
        ),
    )

    assert len(outcome.components) == 2
    assert outcome.components == baseline.components
    origin, inner_sample, exterior, annulus = outcome.sample_dispositions
    assert origin.status == inner_sample.status == exterior.status == "INSIDE"
    assert origin.component_id == inner_sample.component_id
    assert origin.component_id != exterior.component_id
    assert annulus.status == "OUTSIDE"
    assert annulus.component_id is None


def test_disk_and_disjoint_disk_union_component_counts() -> None:
    unit = _radial_level(1)
    left = _polynomial(((1, (2, 0)), (4, (1, 0)), (1, (0, 2)), (3, (0, 0))))
    right = _polynomial(((1, (2, 0)), (-4, (1, 0)), (1, (0, 2)), (3, (0, 0))))

    disk = _computed(_set((unit,), ((PlaneSign.NEGATIVE,),)))
    disjoint_union = _computed(
        _set(
            (left, right),
            tuple(
                row
                for row in (
                    (PlaneSign.NEGATIVE, PlaneSign.NEGATIVE),
                    (PlaneSign.NEGATIVE, PlaneSign.ZERO),
                    (PlaneSign.NEGATIVE, PlaneSign.POSITIVE),
                    (PlaneSign.ZERO, PlaneSign.NEGATIVE),
                    (PlaneSign.POSITIVE, PlaneSign.NEGATIVE),
                )
            ),
        )
    )

    assert len(disk.components) == 1
    assert len(disjoint_union.components) == 2


def test_strict_and_closed_boundary_semantics() -> None:
    circle = _radial_level(1)
    boundary = _rational_point(1, 0)

    strict = _computed(
        _set((circle,), ((PlaneSign.NEGATIVE,),)),
        (boundary,),
    )
    closed = _computed(
        _set((circle,), ((PlaneSign.NEGATIVE,), (PlaneSign.ZERO,))),
        (boundary,),
    )

    assert strict.sample_dispositions[0].status == "OUTSIDE"
    assert closed.sample_dispositions[0].status == "INSIDE"


def test_circle_zero_set_is_one_component_with_an_exact_representative() -> None:
    outcome = _computed(_set((_radial_level(3),), ((PlaneSign.ZERO,),)))

    assert len(outcome.components) == 1
    representative = outcome.components[0].representative
    assert representative.axis == ("x", "y")
    assert all(
        polynomial.polynomial.terms
        for polynomial in representative.coordinate_polynomials
    )


def test_atom_order_does_not_change_the_canonical_profile() -> None:
    inner = _radial_level(1)
    outer = _radial_level(4)
    forward = _set(
        (inner, outer),
        (
            (PlaneSign.NEGATIVE, PlaneSign.NEGATIVE),
            (PlaneSign.POSITIVE, PlaneSign.POSITIVE),
        ),
    )
    reversed_request = _set(
        (outer, inner),
        (
            (PlaneSign.POSITIVE, PlaneSign.POSITIVE),
            (PlaneSign.NEGATIVE, PlaneSign.NEGATIVE),
        ),
    )

    assert compute_plane_component_profile(forward) == compute_plane_component_profile(
        reversed_request
    )


def test_closed_z_squared_minus_one_lemniscate_is_connected() -> None:
    # |(x+iy)^2 - 1|^2 - 1 = (x^2-y^2-1)^2 + 4x^2y^2 - 1.
    expanded = _polynomial(
        (
            (1, (4, 0)),
            (2, (2, 2)),
            (-2, (2, 0)),
            (1, (0, 4)),
            (2, (0, 2)),
        )
    )
    outcome = _computed(_set((expanded,), ((PlaneSign.NEGATIVE,), (PlaneSign.ZERO,))))

    assert len(outcome.components) == 1


def test_quartic_intersection_retains_degree_sixteen_representatives() -> None:
    # y^4 = x and x^4 = 2 have the two real solutions
    # (2^(1/4), +/- 2^(1/16)).  Both source polynomials lie on the operation's
    # degree-four boundary, while either y-coordinate has algebraic degree 16.
    y_fourth_minus_x = _polynomial(((-1, (1, 0)), (1, (0, 4))))
    x_fourth_minus_two = _polynomial(((1, (4, 0)), (-2, (0, 0))))

    semialgebraic_set = _set(
        (y_fourth_minus_x, x_fourth_minus_two),
        ((PlaneSign.ZERO, PlaneSign.ZERO),),
    )
    outcome = _computed(semialgebraic_set)

    assert len(outcome.components) == 2
    representatives = tuple(
        component.representative for component in outcome.components
    )
    assert {
        (
            representative.coordinates[0].polynomial,
            representative.coordinates[0].real_root_index,
        )
        for representative in representatives
    } == {(("1", "0", "0", "0", "-2"), 1)}
    assert {
        representative.coordinates[1].polynomial for representative in representatives
    } == {("1", *("0" for _ in range(15)), "-2")}
    assert {
        representative.coordinates[1].real_root_index
        for representative in representatives
    } == {0, 1}
    assert all(
        representative.isolating_box.intervals[0].lower.as_fraction() > 0
        for representative in representatives
    )
    assert (
        sum(
            representative.isolating_box.intervals[1].upper.as_fraction() < 0
            for representative in representatives
        )
        == 1
    )

    consumer_request = PlaneComponentProfileRequest.model_validate_json(
        PlaneComponentProfileRequest(
            semialgebraic_set=semialgebraic_set,
            samples=(representatives[0],),
        ).model_dump_json(),
        strict=True,
    )
    assert len(consumer_request.samples[0].coordinates[1].polynomial) == 17
    consumed = _computed(
        consumer_request.semialgebraic_set,
        consumer_request.samples,
    )
    assert consumed.components == outcome.components
    assert consumed.sample_dispositions[0].status == "INSIDE"
    assert consumed.sample_dispositions[0].component_id is not None
    assert (
        sum(
            representative.isolating_box.intervals[1].lower.as_fraction() > 0
            for representative in representatives
        )
        == 1
    )


def test_empty_and_whole_plane_degeneracies() -> None:
    empty_source = _set((), ())
    empty = _computed(empty_source)
    assert (
        verify_plane_component_profile(compute_plane_component_profile(empty_source))
        is True
    )
    whole = _computed(_set((), ((),)))

    assert empty.components == ()
    assert len(whole.components) == 1


def test_supplied_sample_must_select_one_root_per_coordinate() -> None:
    ambiguous_x = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=("1", "0", "-1"),
        real_root_index=0,
    )
    sample = IsolatedRealPlanePoint(
        axis=("x", "y"),
        coordinates=(ambiguous_x, _real_value(Fraction())),
        isolating_box=RationalBox(
            domain="QQ",
            variables=("x", "y"),
            intervals=(
                ClosedRationalInterval(lower=_q(-2), upper=_q(2)),
                ClosedRationalInterval(lower=_q(0), upper=_q(0)),
            ),
        ),
    )

    semialgebraic_sets = (
        _set((), ()),
        _set((), ((),)),
        _set((_radial_level(1),), ((PlaneSign.NEGATIVE,),)),
    )
    for semialgebraic_set in semialgebraic_sets:
        with pytest.raises(
            OperationDomainValidationError, match="select one exact real root"
        ):
            compute_plane_component_profile(
                semialgebraic_set,
                (sample,),
            )
