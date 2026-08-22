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


class TestCollinearTriples:
    def _g(self, pts):
        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import (
            LabelledRationalPoint,
            PointConfiguration,
        )

        def cr(x):
            return CanonicalRational.from_fraction(__import__("fractions").Fraction(x))

        return PointConfiguration(
            points=tuple(
                LabelledRationalPoint(label=label, coordinates=(cr(x), cr(y)))
                for label, x, y in pts
            ),
        )

    def test_general_position_no_collinear(self):
        from jacobian.math.geometry.exact._models import CollinearTriplesRequest
        from jacobian.math.geometry.exact._operations import (
            compute_collinear_triples,
        )

        cfg = self._g([("a", -1, 0), ("b", 1, 0), ("c", 0, 2), ("d", 0, -2)])
        result = compute_collinear_triples(CollinearTriplesRequest(configuration=cfg))
        assert result.holds is False
        assert result.witnesses == ()
        assert result.point_count == 4

    def test_collinear_triple_present(self):
        from jacobian.math.geometry.exact._models import CollinearTriplesRequest
        from jacobian.math.geometry.exact._operations import (
            compute_collinear_triples,
        )

        cfg = self._g([("a", 0, 0), ("b", 2, 0), ("c", 0, 2), ("d", 0, -2)])
        result = compute_collinear_triples(CollinearTriplesRequest(configuration=cfg))
        assert result.holds is True
        # A=(0,0), C=(0,2), D=(0,-2) are collinear on x=0 (indices 0,2,3).
        assert (0, 2, 3) in result.witnesses

    def test_all_collinear_returns_all_triples(self):
        from itertools import combinations

        from jacobian.math.geometry.exact._models import CollinearTriplesRequest
        from jacobian.math.geometry.exact._operations import (
            compute_collinear_triples,
        )

        cfg = self._g([("a", 0, 0), ("b", 1, 0), ("c", 2, 0), ("d", 3, 0)])
        result = compute_collinear_triples(CollinearTriplesRequest(configuration=cfg))
        assert result.holds is True
        assert set(result.witnesses) == set(combinations(range(4), 3))


class TestConcyclicQuadruples:
    def _g(self, pts):
        from fractions import Fraction

        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import (
            LabelledRationalPoint,
            PointConfiguration,
        )

        def cr(x):
            return CanonicalRational.from_fraction(Fraction(x))

        return PointConfiguration(
            points=tuple(
                LabelledRationalPoint(label=label, coordinates=(cr(x), cr(y)))
                for label, x, y in pts
            ),
        )

    def test_unit_circle_concyclic(self):
        from jacobian.math.geometry.exact._models import ConcyclicQuadruplesRequest
        from jacobian.math.geometry.exact._operations import (
            compute_concyclic_quadruples,
        )

        cfg = self._g([("a", 1, 0), ("b", 0, 1), ("c", -1, 0), ("d", 0, -1)])
        result = compute_concyclic_quadruples(
            ConcyclicQuadruplesRequest(configuration=cfg),
        )
        assert result.holds is True
        assert (0, 1, 2, 3) in result.witnesses

    def test_general_position_no_concyclic(self):
        from jacobian.math.geometry.exact._models import ConcyclicQuadruplesRequest
        from jacobian.math.geometry.exact._operations import (
            compute_concyclic_quadruples,
        )

        cfg = self._g([("a", -1, 0), ("b", 1, 0), ("c", 0, 2), ("d", 0, -2)])
        result = compute_concyclic_quadruples(
            ConcyclicQuadruplesRequest(configuration=cfg),
        )
        assert result.holds is False
        assert result.witnesses == ()

    def test_rejects_nonplanar(self):
        import pytest

        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import (
            ConcyclicQuadruplesRequest,
            LabelledRationalPoint,
            PointConfiguration,
        )

        def cr(x):
            return CanonicalRational.from_fraction(__import__("fractions").Fraction(x))

        pts = (
            LabelledRationalPoint(label="a", coordinates=(cr(0), cr(0), cr(0))),
            LabelledRationalPoint(label="b", coordinates=(cr(1), cr(0), cr(0))),
            LabelledRationalPoint(label="c", coordinates=(cr(0), cr(1), cr(0))),
        )
        with pytest.raises(ValueError, match="planar"):
            ConcyclicQuadruplesRequest(configuration=PointConfiguration(points=pts))
