"""Tests for circumradius profiles, forbidden patterns, and pinned distances."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    ForbiddenConfiguration,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import (
    circumradius_profile,
    forbidden_patterns,
)
from jacobian.math.geometry._pinned_distances import (
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _rational(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(x=_rational(x), y=_rational(y))


def _labelled(label: str, x: str, y: str) -> LabelledPoint2D:
    return LabelledPoint2D(label=label, point=_point(x, y))


class TestCatalogRegistration:
    def test_new_geometry_operations_are_registered(self):
        ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert "geometry.circumradius.profile.compute" in ids
        assert "geometry.configuration.forbidden_patterns.check" in ids
        assert "geometry.points.compute.pinned_distances" in ids


class TestPinnedDistances:
    def test_unit_square_has_six_distinct_lines(self):
        request = PinnedDistanceRequest(
            anchor=_point("0", "0"),
            points=(
                _point("0", "0"),
                _point("1", "0"),
                _point("1", "1"),
                _point("0", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 6
        assert len(result.lines) == 6
        assert result.min_squared_distance == result.lines[0]
        assert all(entry.squared_distance() >= 0 for entry in result.lines)

    def test_equidistant_parallel_lines_stay_separate_entries(self):
        request = PinnedDistanceRequest(
            anchor=_point("0", "0"),
            points=(
                _point("1", "0"),
                _point("1", "2"),
                _point("0", "1"),
                _point("2", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 6
        at_unit_distance = [
            entry for entry in result.lines if entry.squared_distance() == 1
        ]
        assert len(at_unit_distance) == 2
        assert all(len(entry.source_pairs) == 1 for entry in at_unit_distance)

    def test_min_entry_is_the_exact_minimum(self):
        request = PinnedDistanceRequest(
            anchor=_point("0", "0"),
            points=(
                _point("3", "0"),
                _point("0", "4"),
                _point("1", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        values = [entry.squared_distance() for entry in result.lines]
        assert result.min_squared_distance.squared_distance() == min(values)

    def test_source_bound_replay_rejects_tampered_ledger(self):
        anchor = _point("0", "0")
        points = (_point("1", "0"), _point("0", "1"), _point("1", "1"))
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        assert len(result.lines) == 3
        with pytest.raises(ValidationError, match="exact canonical ledger"):
            PinnedDistanceResult(
                anchor=anchor,
                points=points,
                lines=result.lines[:2],
                distinct_line_count=2,
                min_squared_distance=result.lines[0],
            )

    def test_huge_coordinates_rejected_at_request_boundary(self):
        big = format_canonical_integer(10**4999)
        with pytest.raises(ValidationError, match="pinned-distance bound"):
            PinnedDistanceRequest(
                anchor=RationalPoint2D(x=_rational("0"), y=_rational("1")),
                points=(
                    _point("0", "0"),
                    RationalPoint2D(x=_rational(big), y=_rational("1")),
                ),
            )

    def test_entry_distance_parses_beyond_int_str_limit(self):
        numerator = "9" + "0" * 5000
        entry = LineDistanceEntry(
            squared_distance_numerator=numerator,
            squared_distance_denominator="7",
            source_pairs=((0, 1),),
        )
        expected = Fraction(
            parse_canonical_integer(numerator),
            parse_canonical_integer("7"),
        )
        assert entry.squared_distance() == expected


class TestCircumradiusProfile:
    def test_unit_square_right_triangles(self):
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "1", "1"),
                _labelled("D", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.point_count == 4
        assert result.triple_count == 4
        half = CanonicalRational(num="1", den="2")
        assert all(entry.squared_circumradius == half for entry in result.entries)

    def test_collinear_triples_are_flagged_degenerate(self):
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "2", "0"),
                _labelled("D", "0", "1"),
            )
        )
        result = circumradius_profile(request)
        collinear = [e for e in result.entries if e.collinear]
        assert [(e.indices) for e in collinear] == [(0, 1, 2)]
        assert all(e.squared_circumradius is None for e in collinear)

    def test_result_validates_against_retained_sources(self):
        request = CircumradiusProfileRequest(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "2", "0"),
                _labelled("C", "0", "2"),
            )
        )
        result = circumradius_profile(request)
        forged_entries = (
            CircumradiusTripleEntry(
                labels=result.entries[0].labels,
                indices=(0, 1, 2),
                collinear=False,
                squared_circumradius=CanonicalRational(num="7", den="1"),
            ),
        )
        with pytest.raises(ValidationError, match="replayed"):
            CircumradiusProfileResult(
                point_count=3,
                triple_count=1,
                entries=forged_entries,
                points=request.points,
            )

    def test_oversized_coordinates_rejected_at_boundary(self):
        big = format_canonical_integer(10**129)
        with pytest.raises(ValidationError, match="128-digit"):
            CircumradiusProfileRequest(
                points=(
                    _labelled("A", "0", "0"),
                    _labelled("B", big, "0"),
                    _labelled("C", "0", "1"),
                )
            )


class TestForbiddenPatterns:
    def test_general_position_quadruple_has_neither_pattern(self):
        payload_points = [
            ("A", "0", "0"),
            ("B", "1", "0"),
            ("C", "0", "1"),
            ("D", "2", "3"),
        ]
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=tuple(_labelled(*item) for item in payload_points)
            )
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1

    def test_collinear_triple_carries_witness(self):
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=(
                    _labelled("A", "0", "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "2", "0"),
                    _labelled("D", "5", "5"),
                )
            )
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)
        assert result.checked_triples == 1

    def test_concyclic_quadruple_on_unit_square(self):
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=(
                    _labelled("A", "0", "0"),
                    _labelled("B", "1", "0"),
                    _labelled("C", "1", "1"),
                    _labelled("D", "0", "1"),
                )
            )
        )
        result = forbidden_patterns(request)
        assert result.has_concyclic_quadruple is True
        assert result.concyclic_quadruple.fourth == 3

    def test_fully_collinear_quadruple_is_not_concyclic(self):
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=(
                    _labelled("A", "5", "5"),
                    _labelled("B", "0", "0"),
                    _labelled("C", "1", "0"),
                    _labelled("D", "2", "0"),
                )
            )
        )
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is True
        assert result.has_concyclic_quadruple is False

    def test_result_replay_rejects_forged_decision(self):
        configuration = ForbiddenConfiguration(
            points=(
                _labelled("A", "0", "0"),
                _labelled("B", "1", "0"),
                _labelled("C", "2", "0"),
            )
        )
        with pytest.raises(ValidationError, match="exact screening"):
            ForbiddenPatternsResult(
                configuration=configuration,
                point_count=3,
                has_collinear_triple=False,
                has_concyclic_quadruple=False,
                checked_triples=1,
                checked_quadruples=0,
            )
