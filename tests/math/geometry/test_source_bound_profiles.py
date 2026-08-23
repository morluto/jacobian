"""Source-bound profile admission and replay tests for planar geometry."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import parse_canonical_integer
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
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
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _cr(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


def _point(x: int, y: int) -> RationalPoint2D:
    return RationalPoint2D(x=_cr(x, 1), y=_cr(y, 1))


def _entry_distance(entry) -> Fraction:
    return Fraction(
        parse_canonical_integer(entry.squared_distance_numerator),
        parse_canonical_integer(entry.squared_distance_denominator),
    )


def _labelled(
    label: str, x: CanonicalRational, y: CanonicalRational
) -> LabelledPoint2D:
    return LabelledPoint2D(label=label, point=RationalPoint2D(x=x, y=y))


UNIT_SQUARE = (
    _point(0, 0),
    _point(1, 0),
    _point(1, 1),
    _point(0, 1),
)

GENERAL_POSITION = (
    _point(0, 0),
    _point(1, 0),
    _point(0, 1),
    _point(2, 3),
)


class TestCircumradiusAdmission:
    def test_unit_square_known_values(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("a", _cr(0, 1), _cr(0, 1)),
                _labelled("b", _cr(1, 1), _cr(0, 1)),
                _labelled("c", _cr(1, 1), _cr(1, 1)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].squared_circumradius == _cr(1, 2)

    def test_collinear_triple_is_degenerate_entry(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                _labelled("a", _cr(0, 1), _cr(0, 1)),
                _labelled("b", _cr(1, 1), _cr(0, 1)),
                _labelled("c", _cr(2, 1), _cr(0, 1)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear is True
        assert result.entries[0].squared_circumradius is None

    def test_cross_multiplying_denominators_rejected(self) -> None:
        """Coordinates within the input cap can still overflow the result limit."""
        d = 10**4095
        points = (
            _labelled("a", _cr(1, d + 19), _cr(1, d + 33)),
            _labelled("b", _cr(1, d + 57), _cr(1, d + 91)),
            _labelled("c", _cr(1, d + 123), _cr(1, d + 169)),
        )
        with pytest.raises(ValidationError, match="squared-circumradius height"):
            CircumradiusProfileRequest(points=points)

    def test_large_but_bounded_coordinates_admitted(self) -> None:
        e = 10**799
        request = CircumradiusProfileRequest(
            points=(
                _labelled("a", _cr(1, e), _cr(1, 2 * e)),
                _labelled("b", _cr(1, 3 * e), _cr(1, 4 * e)),
                _labelled("c", _cr(1, 5 * e), _cr(1, 6 * e)),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].collinear is False
        assert len(result.entries[0].squared_circumradius.num) < 32_768


class TestPinnedDistanceReplay:
    def test_unit_square_anchor(self) -> None:
        request = PinnedDistanceRequest(anchor=_point(0, 0), points=UNIT_SQUARE)
        result = compute_pinned_distances(request)
        distances = {_entry_distance(entry) for entry in result.lines}
        assert distances == {Fraction(0), Fraction(1), Fraction(1, 2)}
        assert result.distinct_line_count == 6
        assert _entry_distance(result.min_squared_distance) == Fraction(0)

    def test_genuine_result_round_trips(self) -> None:
        request = PinnedDistanceRequest(anchor=_point(0, 0), points=UNIT_SQUARE)
        result = compute_pinned_distances(request)
        assert PinnedDistanceResult.model_validate(result.model_dump()) == result

    def test_forged_distance_rejected(self) -> None:
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=_point(0, 0), points=UNIT_SQUARE)
        ).model_dump()
        result["lines"][0]["squared_distance_numerator"] = "77"
        with pytest.raises(ValidationError, match="recomputed"):
            PinnedDistanceResult.model_validate(result)

    def test_dropped_source_pair_rejected(self) -> None:
        collinear_row = (_point(0, 0), _point(1, 0), _point(2, 0), _point(0, 1))
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=_point(5, 5), points=collinear_row)
        ).model_dump()
        shared = next(
            entry for entry in result["lines"] if len(entry["source_pairs"]) > 1
        )
        shared["source_pairs"] = shared["source_pairs"][:1]
        with pytest.raises(ValidationError, match="recomputed"):
            PinnedDistanceResult.model_validate(result)

    def test_duplicate_source_pair_rejected(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point(0, 0),
            points=(UNIT_SQUARE[0], UNIT_SQUARE[1]),
        )
        result = compute_pinned_distances(request).model_dump()
        first_pair = result["lines"][0]["source_pairs"][0]
        result["lines"][0]["source_pairs"] = (first_pair, first_pair)
        with pytest.raises(ValidationError, match="recomputed"):
            PinnedDistanceResult.model_validate(result)

    def test_omitted_line_rejected(self) -> None:
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=_point(0, 0), points=UNIT_SQUARE)
        ).model_dump()
        result["lines"] = result["lines"][:-1]
        result["distinct_line_count"] = len(result["lines"])
        with pytest.raises(ValidationError, match="recomputed"):
            PinnedDistanceResult.model_validate(result)

    def test_equal_distance_lines_stay_distinct(self) -> None:
        """x=1 and y=1 are distinct lines despite identical anchor distance."""
        request = PinnedDistanceRequest(
            anchor=_point(0, 0),
            points=(UNIT_SQUARE[1], UNIT_SQUARE[2], UNIT_SQUARE[3]),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 3


class TestCircumradiusSourceBinding:
    def _square_request(self) -> CircumradiusProfileRequest:
        return CircumradiusProfileRequest(
            points=(
                _labelled("a", _cr(0, 1), _cr(0, 1)),
                _labelled("b", _cr(1, 1), _cr(0, 1)),
                _labelled("c", _cr(1, 1), _cr(1, 1)),
                _labelled("d", _cr(0, 1), _cr(1, 1)),
            )
        )

    def test_genuine_profile_round_trips(self) -> None:
        result = circumradius_profile(self._square_request())
        assert CircumradiusProfileResult.model_validate(result.model_dump()) == result

    def test_forged_radius_rejected(self) -> None:
        result = circumradius_profile(self._square_request()).model_dump()
        result["entries"][0]["squared_circumradius"] = {"num": "7", "den": "3"}
        with pytest.raises(ValidationError, match="replayed"):
            CircumradiusProfileResult.model_validate(result)

    def test_forged_labels_rejected(self) -> None:
        result = circumradius_profile(self._square_request()).model_dump()
        result["entries"][0]["labels"] = ("x", "y", "z")
        with pytest.raises(ValidationError, match="labels"):
            CircumradiusProfileResult.model_validate(result)

    def test_flipped_collinearity_rejected(self) -> None:
        result = circumradius_profile(self._square_request()).model_dump()
        nondegenerate = next(
            entry for entry in result["entries"] if not entry["collinear"]
        )
        nondegenerate["collinear"] = True
        nondegenerate["squared_circumradius"] = None
        with pytest.raises(ValidationError, match="flags"):
            CircumradiusProfileResult.model_validate(result)

    def test_detached_points_rejected(self) -> None:
        result = circumradius_profile(self._square_request())
        forged = result.model_dump()
        forged["points"] = [
            {
                "label": "a",
                "point": {"x": {"num": "5", "den": "1"}, "y": {"num": "5", "den": "1"}},
            },
            *forged["points"][1:],
        ]
        with pytest.raises(ValidationError, match="replayed"):
            CircumradiusProfileResult.model_validate(forged)

    def test_point_count_mismatch_rejected(self) -> None:
        result = circumradius_profile(self._square_request()).model_dump()
        result["point_count"] = 5
        with pytest.raises(ValidationError, match="point_count"):
            CircumradiusProfileResult.model_validate(result)


class TestForbiddenPatternBinding:
    def _configuration(self, points) -> ForbiddenPatternsRequest:
        return ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=tuple(
                    ForbiddenLabelledPoint(label=f"p{index}", point=point)
                    for index, point in enumerate(points)
                )
            )
        )

    def test_square_reports_concyclic_witness(self) -> None:
        result = forbidden_patterns(self._configuration(UNIT_SQUARE))
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is True
        witness = result.concyclic_quadruple
        assert (witness.first, witness.second, witness.third, witness.fourth) == (
            0,
            1,
            2,
            3,
        )
        revalidated = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert revalidated == result

    def test_general_position_reports_neither(self) -> None:
        result = forbidden_patterns(self._configuration(GENERAL_POSITION))
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        assert result.checked_triples == 4
        assert result.checked_quadruples == 1
        revalidated = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert revalidated == result

    def test_false_negative_on_square_rejected(self) -> None:
        result = forbidden_patterns(self._configuration(UNIT_SQUARE)).model_dump()
        result["has_concyclic_quadruple"] = False
        result["concyclic_quadruple"] = None
        with pytest.raises(ValidationError, match="enumeration"):
            ForbiddenPatternsResult.model_validate(result)

    def test_wrong_enumeration_count_rejected(self) -> None:
        result = forbidden_patterns(self._configuration(UNIT_SQUARE)).model_dump()
        result["checked_triples"] = 3
        with pytest.raises(ValidationError, match="enumeration"):
            ForbiddenPatternsResult.model_validate(result)

    def test_detached_point_count_rejected(self) -> None:
        result = forbidden_patterns(self._configuration(GENERAL_POSITION)).model_dump()
        result["point_count"] = 5
        with pytest.raises(ValidationError, match="point_count"):
            ForbiddenPatternsResult.model_validate(result)


class TestForbiddenScreeningWorkBound:
    @staticmethod
    def _parabola_points(count: int) -> tuple[RationalPoint2D, ...]:
        return tuple(
            RationalPoint2D(x=_cr(t, 1), y=_cr(t * t, 1)) for t in range(1, count + 1)
        )

    def _request(self, points) -> ForbiddenPatternsRequest:
        return ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=tuple(
                    ForbiddenLabelledPoint(label=f"p{index}", point=point)
                    for index, point in enumerate(points)
                )
            )
        )

    def test_128_point_configuration_rejected_by_work_budget(self) -> None:
        """The complete enumeration must stay inside its declared budget."""
        with pytest.raises(ValidationError, match="screening work"):
            self._request(self._parabola_points(128))

    def test_heavy_denominator_configuration_rejected_by_work_budget(self) -> None:
        points = tuple(
            RationalPoint2D(x=_cr(t, 1), y=_cr(1, 10**300 + 2 * t + 1))
            for t in range(1, 41)
        )
        with pytest.raises(ValidationError, match="screening work"):
            self._request(points)

    def test_coordinate_component_cap_rejects_extreme_height(self) -> None:
        points = (
            RationalPoint2D(x=_cr(0, 1), y=_cr(1, 10**4096 + 1)),
            _point(1, 0),
            _point(0, 1),
        )
        with pytest.raises(ValidationError, match="configuration point y"):
            self._request(points)

    def test_pattern_free_boundary_configuration_computes_exactly(self) -> None:
        from math import comb

        request = self._request(self._parabola_points(48))
        result = forbidden_patterns(request)
        assert result.has_collinear_triple is False
        assert result.has_concyclic_quadruple is False
        assert result.checked_triples == comb(48, 3)
        assert result.checked_quadruples == comb(48, 4)
        revalidated = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert revalidated == result

    def test_cleared_denominators_preserve_decisions_and_witnesses(self) -> None:
        integer_points = self._parabola_points(6)
        scaled = tuple(
            RationalPoint2D(x=_cr(t, 3), y=_cr(t * t, 3)) for t in range(1, 7)
        )
        integer_result = forbidden_patterns(self._request(integer_points))
        fractional_result = forbidden_patterns(self._request(scaled))
        assert (
            integer_result.has_collinear_triple,
            integer_result.has_concyclic_quadruple,
        ) == (
            fractional_result.has_collinear_triple,
            fractional_result.has_concyclic_quadruple,
        )
        assert integer_result.checked_triples == fractional_result.checked_triples
        assert integer_result.checked_quadruples == fractional_result.checked_quadruples

    def test_unit_square_witness_unchanged_under_scaling(self) -> None:
        base = forbidden_patterns(self._request(UNIT_SQUARE))
        scaled = tuple(
            RationalPoint2D(
                x=CanonicalRational.from_fraction(p.x.as_fraction() / 5),
                y=CanonicalRational.from_fraction(p.y.as_fraction() / 5),
            )
            for p in UNIT_SQUARE
        )
        result = forbidden_patterns(self._request(scaled))
        assert (result.has_collinear_triple, result.has_concyclic_quadruple) == (
            base.has_collinear_triple,
            base.has_concyclic_quadruple,
        )
        assert (
            result.concyclic_quadruple.first,
            result.concyclic_quadruple.second,
            result.concyclic_quadruple.third,
            result.concyclic_quadruple.fourth,
        ) == (
            base.concyclic_quadruple.first,
            base.concyclic_quadruple.second,
            base.concyclic_quadruple.third,
            base.concyclic_quadruple.fourth,
        )
