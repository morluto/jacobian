"""Public exact H-to-V polyhedron contract and independent small fixtures."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.polytopes import (
    RationalAffineHalfspace,
    RationalHPolyhedron,
    RationalPolyhedronSpace,
    Vertex,
    halfspaces_to_v_presentation,
    polyhedron_conversion,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _space(*axes: str) -> RationalPolyhedronSpace:
    return RationalPolyhedronSpace(axes=axes)


def _halfspace(normal: tuple[int, ...], bound: int) -> RationalAffineHalfspace:
    return RationalAffineHalfspace(
        normal=tuple(_q(value) for value in normal), bound=_q(bound)
    )


def _convert(axes: tuple[str, ...], *rows: RationalAffineHalfspace):
    return halfspaces_to_v_presentation(
        RationalHPolyhedron(space=_space(*axes), inequalities=rows)
    )


def test_ray_orientation_is_preserved_exactly() -> None:
    value = _convert(("x",), _halfspace((-1,), 0))
    assert value.empty is False
    assert value.points == ((_q(0),),)
    assert value.rays == ((_q(1),),)
    assert value.lineality == ()
    assert value.affine_dimension == 1


def test_unrestricted_line_has_point_and_lineality_without_boxing() -> None:
    value = _convert(("x",))
    assert value.points == ((_q(0),),)
    assert value.rays == ()
    assert value.lineality == ((_q(1),),)
    assert value.affine_dimension == 1


def test_zero_dimensional_affine_space_is_a_singleton() -> None:
    value = _convert(())
    assert value.points == ((),)
    assert value.rays == value.lineality == ()
    assert value.empty is False
    assert value.affine_dimension == 0


def test_bounded_square_has_four_exact_points_and_composes_with_facet_consumer() -> (
    None
):
    square = _convert(
        ("x", "y"),
        _halfspace((-1, 0), 0),
        _halfspace((1, 0), 1),
        _halfspace((0, -1), 0),
        _halfspace((0, 1), 1),
    )
    expected = {
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(1)),
    }
    actual = {
        tuple(component.as_fraction() for component in point) for point in square.points
    }
    assert actual == expected
    assert square.rays == square.lineality == ()
    assert square.affine_dimension == 2

    # H-to-V output feeds the existing full-dimensional bounded V-to-facets consumer.
    profile = facet_incidence(
        tuple(Vertex(coordinates=point) for point in square.points), 2
    )
    assert len(profile.facets) == 4


def test_inconsistent_halfspaces_return_the_empty_presentation() -> None:
    value = _convert(("x",), _halfspace((1,), 0), _halfspace((-1,), -1))
    assert value.empty is True
    assert value.points == value.rays == value.lineality == ()
    assert value.affine_dimension == -1


def test_lower_dimensional_line_keeps_its_affine_direction() -> None:
    value = _convert(("x", "y"), _halfspace((1, 0), 0), _halfspace((-1, 0), 0))
    assert value.points == ((_q(0), _q(0)),)
    assert value.rays == ()
    assert value.lineality == ((_q(0), _q(1)),)
    assert value.affine_dimension == 1


def test_v_presentation_round_trips_through_its_canonical_json() -> None:
    value = _convert(("x",), _halfspace((-1,), 0))
    wire = value.model_dump_json()
    restored = type(value).model_validate_json(wire)
    assert restored == value


def test_huge_constant_rows_are_classified_before_the_input_height_envelope() -> None:
    huge = 10**1_024
    # A true constant row never enters double-description expansion, so it is
    # discarded even when its bound exceeds the DD input height envelope.
    value = _convert(("x",), _halfspace((0,), huge), _halfspace((-1,), 0))
    assert value.empty is False
    assert value.points == ((_q(0),),)
    assert value.rays == ((_q(1),),)
    # A false constant row is a direct exact contradiction and returns the
    # empty presentation rather than refusing the oversized bound.
    contradictory = _convert(("x",), _halfspace((0,), -huge), _halfspace((-1,), 0))
    assert contradictory.empty is True
    assert contradictory.points == contradictory.rays == ()
    assert contradictory.affine_dimension == -1


def test_coefficient_growth_is_admitted_before_kernel_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("double-description conversion ran before admission")

    monkeypatch.setattr(
        polyhedron_conversion, "_halfspaces_to_generators_from_admission", forbidden
    )
    huge = 10**1_023
    source = RationalHPolyhedron(
        space=_space("x1", "x2", "x3", "x4", "x5", "x6", "x7"),
        inequalities=(
            RationalAffineHalfspace(
                normal=(_q(huge), *(_q(0) for _ in range(6))), bound=_q(0)
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="generator coefficients"):
        halfspaces_to_v_presentation(source)


def test_wire_size_bound_is_admitted_before_kernel_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("double-description conversion ran before byte admission")

    monkeypatch.setattr(
        polyhedron_conversion, "_halfspaces_to_generators_from_admission", forbidden
    )
    huge = _q(10**1_023)
    rows = [
        _halfspace((-1, 0, 0), 0),
        _halfspace((1, 0, 0), 1),
        _halfspace((0, -1, 0), 0),
        _halfspace((0, 1, 0), 1),
        _halfspace((0, 0, -1), 0),
        _halfspace((0, 0, 1), 1),
    ]
    rows.extend(
        RationalAffineHalfspace(
            normal=tuple(
                _q(value) for value in (index % 7 + 1, index % 5 + 1, index % 3 + 1)
            ),
            bound=huge,
        )
        for index in range(51)
    )
    source = RationalHPolyhedron(space=_space("x", "y", "z"), inequalities=tuple(rows))
    with pytest.raises(
        OperationResourceAdmissionError, match="conservative V-result bound"
    ):
        halfspaces_to_v_presentation(source)
