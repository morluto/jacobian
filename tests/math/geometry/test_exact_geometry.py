"""Tests for exact geometry operations."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile
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


class TestCircumradiusAdmission:
    @staticmethod
    def _labelled(label: str, x: CanonicalRational, y: CanonicalRational):
        return LabelledPoint2D(label=label, point=RationalPoint2D(x=x, y=y))

    def test_reciprocal_huge_denominators_rejected(self) -> None:
        """(1/a,1/b)-style points compound denominators past the limit."""
        big = "1" + "0" * 4090
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(
                points=(
                    self._labelled(
                        "o",
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="0", den="1"),
                    ),
                    self._labelled(
                        "p",
                        CanonicalRational(num="1", den=big),
                        CanonicalRational(num="1", den=big),
                    ),
                    self._labelled(
                        "q",
                        CanonicalRational(num="3", den=big),
                        CanonicalRational(num="2", den=big),
                    ),
                )
            )

    def test_right_triangle_known_answer(self) -> None:
        request = CircumradiusProfileRequest(
            points=(
                self._labelled(
                    "a",
                    CanonicalRational(num="0", den="1"),
                    CanonicalRational(num="0", den="1"),
                ),
                self._labelled(
                    "b",
                    CanonicalRational(num="4", den="1"),
                    CanonicalRational(num="0", den="1"),
                ),
                self._labelled(
                    "c",
                    CanonicalRational(num="0", den="1"),
                    CanonicalRational(num="3", den="1"),
                ),
            )
        )
        result = circumradius_profile(request)
        entry = result.entries[0]
        # Hypotenuse/2 = 5/2, squared = 25/4.
        assert entry.squared_circumradius == CanonicalRational(num="25", den="4")
