"""Tests for exact geometry operations."""


from jacobian.contracts.exact_geometry import (
    DistanceGraphRequest,
    DistanceProfileRequest,
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.domains.exact_geometry.operations import (
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
        assert entries.get("1") == 4  # 4 unit-distance pairs
        assert entries.get("2") == 2  # 2 diagonal pairs

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
        assert entries.get("1") == 2  # a-b and b-c
        assert entries.get("4") == 1  # a-c


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
            target_squared_distance="1",
        )
        result = compute_distance_graph(req)
        assert len(result.edges) == 4
