"""Correctness and scale regressions for the private exact DD kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from jacobian.math.geometry.polytopes import _polyhedral_conversion
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    dd_work_bound,
    halfspaces_to_generators,
    points_to_facets,
    primitive_homogeneous_halfspace,
    pulling_triangulation,
    require_dd_weighted_work_admissible,
    require_dd_work_admissible,
    require_pulling_work_admissible,
)


def test_homogeneous_halfspace_clears_all_denominators_and_content() -> None:
    assert primitive_homogeneous_halfspace(
        (Fraction(1, 2), Fraction(-3, 4)), Fraction(5, 6)
    ) == (10, -6, 9)


def test_dual_conversion_round_trips_a_degenerate_square_description() -> None:
    points = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (Fraction(1, 2), 0))
    hull = points_to_facets(points, 2)
    assert set(hull.facets) == {
        ((-1, 0), 0),
        ((1, 0), 1),
        ((0, -1), 0),
        ((0, 1), 1),
    }
    edge_point = 5
    assert sum(bool(bits & (1 << edge_point)) for bits in hull.facet_incidence) == 1

    converted = halfspaces_to_generators(hull.facets, 2)
    assert set(converted.vertices) == {
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
    }
    assert converted.bounded
    assert converted.affine_dimension == 2


def test_lineality_quotient_and_lower_dimensional_classification() -> None:
    strip = halfspaces_to_generators(
        [((1, 0), 1), ((-1, 0), 0)],
        2,
    )
    assert not strip.empty
    assert not strip.bounded
    assert strip.affine_dimension == 2
    assert set(strip.lineality_basis) == {(0, 1)}
    for basis in strip.cone.quotient.lineality_basis:
        assert basis[0] == 0

    segment = halfspaces_to_generators(
        [((1, 0), 0), ((-1, 0), 0), ((0, 1), 1), ((0, -1), 0)],
        2,
    )
    assert segment.bounded
    assert segment.affine_dimension == 1

    hull = points_to_facets(((0, 0), (1, 0)), 2)
    assert hull.affine_equalities == (((0, 1), 0),)


def test_empty_and_unbounded_polyhedra_are_distinguished_exactly() -> None:
    empty = halfspaces_to_generators([((1,), 0), ((-1,), -1)], 1)
    assert empty.empty
    assert not empty.bounded
    assert empty.affine_dimension == -1

    ray = halfspaces_to_generators([((-1,), 0)], 1)
    assert not ray.empty
    assert not ray.bounded
    assert ray.recession_rays == ((1,),)


def test_six_cube_uses_product_conversion_and_incidence_triangulation() -> None:
    points = tuple(product((0, 1), repeat=6))
    hull = points_to_facets(points, 6)
    assert len(hull.facets) == 12
    assert hull.cone.candidate_pairs == 0
    simplices = pulling_triangulation(len(points), 6, hull.facet_incidence)
    assert len(simplices) == 720
    assert all(len(simplex) == 7 for simplex in simplices)


def test_output_sensitive_admission_replaces_subset_count() -> None:
    rays, pairs = dd_work_bound(14, 8)
    assert (rays, pairs) == (240, 26333)
    require_dd_work_admissible(14, 8)
    with pytest.raises(ValueError, match="output-sensitive work bound"):
        require_dd_work_admissible(64, 7)
    with pytest.raises(ValueError, match="height-weighted work bound"):
        require_dd_weighted_work_admissible(31, 7, 100_000)
    require_pulling_work_admissible(12, 6)
    with pytest.raises(ValueError, match="pulling triangulation"):
        require_pulling_work_admissible(20, 6)


def test_coefficient_growth_is_rejected_before_cone_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_expansion(_rows: object) -> None:
        raise AssertionError("cone expansion ran before the height gate")

    monkeypatch.setattr(_polyhedral_conversion, "cone_generators", unexpected_expansion)
    huge = Fraction(10**200_000)
    with pytest.raises(ValueError, match="coefficient-growth bound"):
        points_to_facets(((0, 0), (huge, 0), (0, huge)), 2)
