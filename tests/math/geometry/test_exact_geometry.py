"""Tests for exact geometry operations."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    ForbiddenConfiguration,
    ForbiddenLabelledPoint,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile, forbidden_patterns
from jacobian.math.geometry._pinned_distances import (
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceProfileRequest,
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact._operations import (
    compute_distance_graph,
    compute_distance_profile,
)


def _make_point(label: str, coords: list[tuple[str, str]]) -> LabelledRationalPoint:
    return LabelledRationalPoint(
        label=label,
        coordinates=tuple({"num": n, "den": d} for n, d in coords),
    )


class TestDistanceProfile:
    def test_unit_square(self):
        pts = (
            _make_point("a", [("0", "1"), ("0", "1")]),
            _make_point("b", [("1", "1"), ("0", "1")]),
            _make_point("c", [("0", "1"), ("1", "1")]),
            _make_point("d", [("1", "1"), ("1", "1")]),
        )
        req = DistanceProfileRequest(
            configuration=PointConfiguration(points=pts),
        )
        result = compute_distance_profile(req)
        entries = {e.squared_distance: e.pair_count for e in result.entries}
        one = CanonicalRational(num="1", den="1")
        two = CanonicalRational(num="2", den="1")
        assert entries.get(one) == 4  # 4 unit-distance pairs
        assert entries.get(two) == 2  # 2 diagonal pairs

    def test_collinear(self):
        pts = (
            _make_point("a", [("0", "1"), ("0", "1")]),
            _make_point("b", [("1", "1"), ("0", "1")]),
            _make_point("c", [("2", "1"), ("0", "1")]),
        )
        req = DistanceProfileRequest(
            configuration=PointConfiguration(points=pts),
        )
        result = compute_distance_profile(req)
        entries = {e.squared_distance: e.pair_count for e in result.entries}
        one = CanonicalRational(num="1", den="1")
        four = CanonicalRational(num="4", den="1")
        assert entries.get(one) == 2  # a-b and b-c
        assert entries.get(four) == 1  # a-c


class TestDistanceGraph:
    def test_unit_square_distance_1(self):
        pts = (
            _make_point("a", [("0", "1"), ("0", "1")]),
            _make_point("b", [("1", "1"), ("0", "1")]),
            _make_point("c", [("0", "1"), ("1", "1")]),
            _make_point("d", [("1", "1"), ("1", "1")]),
        )
        req = DistanceGraphRequest(
            configuration=PointConfiguration(points=pts),
            target_squared_distance=CanonicalRational(num="1", den="1"),
        )
        result = compute_distance_graph(req)
        assert len(result.edges) == 4

    def test_rejects_single_point_configuration(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PointConfiguration(points=(_make_point("a", [("0", "1")]),))

    def test_rejects_negative_squared_distance(self):
        import pytest
        from pydantic import ValidationError

        configuration = PointConfiguration(
            points=(
                _make_point("a", [("0", "1")]),
                _make_point("b", [("1", "1")]),
            )
        )
        with pytest.raises(ValidationError, match="nonnegative"):
            DistanceGraphRequest(
                configuration=configuration,
                target_squared_distance=CanonicalRational(num="-1", den="1"),
            )


class TestCircumradiusCoordinateBound:
    """The coordinate cap derives from rational result growth, not N^2."""

    def _point(self, label: str, xn: str, xd: str, yn: str = "0", yd: str = "1"):
        return LabelledPoint2D(
            label=label,
            point=RationalPoint2D(
                x=CanonicalRational(num=xn, den=xd),
                y=CanonicalRational(num=yn, den=yd),
            ),
        )

    def test_reviewer_construction_rejected(self):
        # (0,0), (1/N, 0), (0, 1/M) with 8192-digit odd N, M produced a
        # 2N^2M^2 denominator beyond the canonical limit after admission.
        n = "9" * 8192
        m = "9" * 8191 + "7"
        with pytest.raises(ValidationError, match="digit"):
            CircumradiusProfileRequest(
                points=(
                    self._point("o", "0", "1", "0", "1"),
                    self._point("a", "1", n),
                    self._point("b", "0", "1", "1", m),
                )
            )

    def test_bounded_rational_triangle_admitted(self):
        request = CircumradiusProfileRequest(
            points=(
                self._point("o", "0", "1", "0", "1"),
                self._point("a", "3", "1", "0", "1"),
                self._point("b", "0", "1", "4", "1"),
            )
        )
        result = circumradius_profile(request)
        assert result.entries[0].squared_circumradius == CanonicalRational(
            num="25", den="4"
        )


class TestPinnedDistanceReplay:
    """Ledger entries replay against the retained anchor and points."""

    def _result(self):
        request = PinnedDistanceRequest(
            anchor=RationalPoint2D(
                x=CanonicalRational(num="0", den="1"),
                y=CanonicalRational(num="0", den="1"),
            ),
            points=(
                RationalPoint2D(
                    x=CanonicalRational(num="1", den="1"),
                    y=CanonicalRational(num="0", den="1"),
                ),
                RationalPoint2D(
                    x=CanonicalRational(num="2", den="1"),
                    y=CanonicalRational(num="0", den="1"),
                ),
            ),
        )
        return compute_pinned_distances(request)

    def test_genuine_result_round_trips(self):
        result = self._result()
        assert PinnedDistanceResult(
            anchor=result.anchor,
            points=result.points,
            lines=result.lines,
            distinct_line_count=result.distinct_line_count,
            min_squared_distance=result.min_squared_distance,
        )

    def test_forged_source_pairs_rejected(self):
        genuine = self._result()
        forged_entry = LineDistanceEntry(
            squared_distance=genuine.lines[0].squared_distance,
            source_pairs=((8, 9),),
        )
        with pytest.raises(ValidationError):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=(forged_entry,),
                distinct_line_count=1,
                min_squared_distance=forged_entry,
            )

    def test_forged_distance_rejected(self):
        genuine = self._result()
        forged_entry = LineDistanceEntry(
            squared_distance=CanonicalRational(num="123", den="7"),
            source_pairs=genuine.lines[0].source_pairs,
        )
        with pytest.raises(ValidationError):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=(forged_entry,),
                distinct_line_count=1,
                min_squared_distance=forged_entry,
            )


class TestForbiddenPatternsSourceBinding:
    @staticmethod
    def _point(label: str, x_num: str, y_num: str) -> ForbiddenLabelledPoint:
        return ForbiddenLabelledPoint(
            label=label,
            point=RationalPoint2D(
                x=CanonicalRational(num=x_num, den="1"),
                y=CanonicalRational(num=y_num, den="1"),
            ),
        )

    @classmethod
    def _request(cls) -> ForbiddenPatternsRequest:
        return ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=(
                    cls._point("a", "0", "0"),
                    cls._point("b", "1", "0"),
                    cls._point("c", "2", "0"),
                    cls._point("d", "0", "1"),
                )
            )
        )

    def test_producer_retains_and_replays_source(self):
        request = self._request()
        result = forbidden_patterns(request)
        assert result.configuration == request.configuration
        assert result.has_collinear_triple
        assert (
            result.collinear_triple.first,
            result.collinear_triple.second,
            result.collinear_triple.third,
        ) == (0, 1, 2)
        replayed = ForbiddenPatternsResult.model_validate(result.model_dump())
        assert replayed.checked_triples == result.checked_triples

    def test_forged_negative_screen_rejected(self):
        genuine = forbidden_patterns(self._request())
        with pytest.raises(ValidationError):
            ForbiddenPatternsResult(
                configuration=genuine.configuration,
                point_count=4,
                has_collinear_triple=False,
                has_concyclic_quadruple=False,
                collinear_triple=None,
                concyclic_quadruple=None,
                checked_triples=0,
                checked_quadruples=0,
            )

    def test_forged_witness_rejected(self):
        genuine = forbidden_patterns(self._request())
        assert genuine.collinear_triple is not None
        with pytest.raises(ValidationError):
            ForbiddenPatternsResult(
                configuration=genuine.configuration,
                point_count=4,
                has_collinear_triple=True,
                has_concyclic_quadruple=False,
                collinear_triple=type(genuine.collinear_triple)(
                    first=0, second=1, third=3
                ),
                concyclic_quadruple=None,
                checked_triples=4,
                checked_quadruples=0,
            )

    def test_concyclic_square_binds_its_quadruple(self):
        request = ForbiddenPatternsRequest(
            configuration=ForbiddenConfiguration(
                points=(
                    self._point("a", "0", "0"),
                    self._point("b", "1", "0"),
                    self._point("c", "1", "1"),
                    self._point("d", "0", "1"),
                )
            )
        )
        result = forbidden_patterns(request)
        assert result.has_concyclic_quadruple
        assert not result.has_collinear_triple
        assert (
            result.concyclic_quadruple.first,
            result.concyclic_quadruple.second,
            result.concyclic_quadruple.third,
            result.concyclic_quadruple.fourth,
        ) == (0, 1, 2, 3)
        ForbiddenPatternsResult.model_validate(result.model_dump())
