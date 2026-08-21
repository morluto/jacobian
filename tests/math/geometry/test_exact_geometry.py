"""Tests for exact geometry operations."""

from jacobian._exact import CanonicalRational
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


class TestCircumradiusProfile:
    def _parabola(self):
        return tuple(
            _make_point(f"t{t}", [(str(t), "1"), (str(t * t), "1")])
            for t in (1, 2, 4, 19, 29)
        )

    def test_parabola_collision(self):
        from fractions import Fraction

        from jacobian.math.geometry.exact._models import CircumradiusProfileRequest
        from jacobian.math.geometry.exact._operations import (
            compute_circumradius_profile,
        )

        req = CircumradiusProfileRequest(
            configuration=PointConfiguration(points=self._parabola()),
        )
        result = compute_circumradius_profile(req)
        assert result.dimension == 2
        assert result.point_count == 5
        assert result.degenerate_count == 0
        assert result.nondegenerate_count == 10
        collision = [
            entry.triple
            for entry in result.triples
            if entry.squared_radius is not None
            and entry.squared_radius.as_fraction() == Fraction(2166905)
        ]
        assert collision == [(0, 1, 4), (1, 2, 3)]
        # The shared radius appears with multiplicity 2.
        mult = {
            entry.squared_radius.as_fraction(): entry.triple_count
            for entry in result.multiplicities
        }
        assert mult[Fraction(2166905)] == 2

    def test_collinear_triple_is_degenerate(self):
        from jacobian.math.geometry.exact._models import (
            CircumradiusProfileRequest,
            CircumradiusTripleDisposition,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_circumradius_profile,
        )

        pts = (
            _make_point("a", [("0", "1"), ("0", "1")]),
            _make_point("b", [("1", "1"), ("0", "1")]),
            _make_point("c", [("2", "1"), ("0", "1")]),
            _make_point("d", [("0", "1"), ("1", "1")]),
        )
        result = compute_circumradius_profile(
            CircumradiusProfileRequest(configuration=PointConfiguration(points=pts)),
        )
        assert result.point_count == 4
        # C(4,3) = 4 triples; the collinear triple (a,b,c) is degenerate.
        assert result.degenerate_count == 1
        assert result.nondegenerate_count == 3
        degenerate = [
            entry.triple
            for entry in result.triples
            if entry.disposition is CircumradiusTripleDisposition.DEGENERATE
        ]
        assert degenerate == [(0, 1, 2)]
        for entry in result.triples:
            if entry.disposition is CircumradiusTripleDisposition.DEGENERATE:
                assert entry.squared_radius is None
            else:
                assert entry.squared_radius is not None
                assert entry.squared_radius.as_fraction() > 0

    def test_three_points_right_triangle(self):
        from fractions import Fraction

        from jacobian.math.geometry.exact._models import CircumradiusProfileRequest
        from jacobian.math.geometry.exact._operations import (
            compute_circumradius_profile,
        )

        pts = (
            _make_point("a", [("0", "1"), ("0", "1")]),
            _make_point("b", [("2", "1"), ("0", "1")]),
            _make_point("c", [("0", "1"), ("2", "1")]),
        )
        result = compute_circumradius_profile(
            CircumradiusProfileRequest(configuration=PointConfiguration(points=pts)),
        )
        assert result.nondegenerate_count == 1
        # Right triangle with legs 2,2: hypotenuse^2=8, R = hypotenuse/2 => R^2=2.
        assert result.triples[0].squared_radius.as_fraction() == Fraction(2)

    def test_multiplicity_total_matches(self):
        from jacobian.math.geometry.exact._models import CircumradiusProfileRequest
        from jacobian.math.geometry.exact._operations import (
            compute_circumradius_profile,
        )

        result = compute_circumradius_profile(
            CircumradiusProfileRequest(configuration=PointConfiguration(points=self._parabola())),
        )
        assert sum(e.triple_count for e in result.multiplicities) == result.nondegenerate_count

    def test_rejects_nonplanar(self):
        import pytest

        from jacobian.math.geometry.exact._models import CircumradiusProfileRequest

        pts = (
            _make_point("a", [("0", "1"), ("0", "1"), ("0", "1")]),
            _make_point("b", [("1", "1"), ("0", "1"), ("0", "1")]),
            _make_point("c", [("0", "1"), ("1", "1"), ("0", "1")]),
        )
        with pytest.raises(ValueError, match="planar"):
            CircumradiusProfileRequest(configuration=PointConfiguration(points=pts))
