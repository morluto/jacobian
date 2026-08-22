"""Tests for exact geometry operations."""

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceProfileRequest,
    LabelledRationalPoint,
    PinnedLineDistanceRequest,
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


class TestPinnedLineDistance:
    def _cfg(self, pts):
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

    def _anchor(self, x, y):
        from fractions import Fraction

        from jacobian._exact import CanonicalRational

        return (
            CanonicalRational.from_fraction(Fraction(x)),
            CanonicalRational.from_fraction(Fraction(y)),
        )

    def test_inverted_orthocentric_equal_distance(self):
        from fractions import Fraction

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        cfg = self._cfg(
            [
                ("b", Fraction(1, 4), Fraction(0)),
                ("c", Fraction(1, 5), Fraction(2, 5)),
                ("h", Fraction(4, 13), Fraction(6, 13)),
            ],
        )
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg, anchor=self._anchor(0, 0)),
        )
        assert result.point_count == 3
        assert len(result.lines) == 3
        for entry in result.lines:
            assert entry.squared_distance.as_fraction() == Fraction(4, 65)
        mult = [(m[0].as_fraction(), m[1]) for m in result.distance_multiplicities]
        assert mult == [(Fraction(4, 65), 3)]

    def test_unit_square_anchor_origin(self):
        from fractions import Fraction

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 0, 1), ("d", 1, 1)])
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg, anchor=self._anchor(0, 0)),
        )
        # C(4,2)=6 pairs, but opposite sides span distinct lines; collinear pairs
        # a-b and c-d span parallel distinct lines. All 6 pairs give distinct
        # geometric lines here (no three collinear), so 6 lines.
        assert len(result.lines) == 6
        total_pairs = sum(len(entry.pairs) for entry in result.lines)
        assert total_pairs == 6
        # The anchor (0,0) lies on lines a-b (y=0) and a-c (x=0): distance 0.
        zero_lines = [
            e for e in result.lines if e.squared_distance.as_fraction() == Fraction(0)
        ]
        assert len(zero_lines) == 3

    def test_collinear_pairs_collapse_to_one_line(self):
        from fractions import Fraction

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        # Three collinear points on y=0: pairs (0,1),(0,2),(1,2) span ONE line.
        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 2, 0)])
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg, anchor=self._anchor(0, 1)),
        )
        assert len(result.lines) == 1
        entry = result.lines[0]
        assert len(entry.pairs) == 3  # all three source pairs retained
        assert set(entry.pairs) == {(0, 1), (0, 2), (1, 2)}
        # distance from (0,1) to y=0 is 1.
        assert entry.squared_distance.as_fraction() == Fraction(1, 1)

    def test_canonical_line_invariance_under_pair_order(self):
        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        # Same geometric set, different label order -> same line coefficients.
        cfg_a = self._cfg([("a", 0, 0), ("b", 2, 0), ("c", 1, 2)])
        cfg_b = self._cfg([("x", 1, 2), ("y", 2, 0), ("z", 0, 0)])
        ra = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg_a, anchor=self._anchor(0, 0)),
        )
        rb = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg_b, anchor=self._anchor(0, 0)),
        )
        coeffs_a = {
            tuple(c.as_fraction() for c in e.line_coefficients) for e in ra.lines
        }
        coeffs_b = {
            tuple(c.as_fraction() for c in e.line_coefficients) for e in rb.lines
        }
        assert coeffs_a == coeffs_b

    def test_rejects_nonplanar(self):
        import pytest

        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import (
            LabelledRationalPoint,
            PinnedLineDistanceRequest,
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
            PinnedLineDistanceRequest(
                configuration=PointConfiguration(points=pts),
                anchor=(cr(0), cr(0)),
            )

    def test_result_rejects_nonplanar_retained_configuration(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
            PinnedLineDistanceResult,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 0, 1), ("d", 1, 1)])
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg, anchor=self._anchor(0, 0))
        )
        cfg3d = PointConfiguration(
            points=tuple(
                LabelledRationalPoint(
                    label=p.label, coordinates=(*p.coordinates, _cr(7))
                )
                for p in cfg.points
            )
        )
        with pytest.raises(ValidationError, match="planar"):
            PinnedLineDistanceResult(
                configuration=cfg3d,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=4,
                lines=result.lines,
                distance_multiplicities=result.distance_multiplicities,
            )

    def test_result_rejects_one_dimensional_retained_configuration(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import PinnedLineDistanceResult

        cfg1d = PointConfiguration(
            points=(
                LabelledRationalPoint(label="a", coordinates=(_cr(0),)),
                LabelledRationalPoint(label="b", coordinates=(_cr(1),)),
            )
        )
        with pytest.raises(ValidationError, match="planar"):
            PinnedLineDistanceResult(
                configuration=cfg1d,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=2,
                lines=(),
                distance_multiplicities=(),
            )

    def test_result_rejects_3d_points_differing_only_outside_plane(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import PinnedLineDistanceResult

        # Same XY location, distinct third component: the XY-only replay would
        # divide by a zero direction norm, so planarity must be checked first.
        cfgz = PointConfiguration(
            points=(
                LabelledRationalPoint(label="a", coordinates=(_cr(0), _cr(0), _cr(0))),
                LabelledRationalPoint(label="b", coordinates=(_cr(0), _cr(0), _cr(1))),
            )
        )
        with pytest.raises(ValidationError, match="planar"):
            PinnedLineDistanceResult(
                configuration=cfgz,
                anchor=self._anchor(0, 1),
                dimension=2,
                point_count=2,
                lines=(),
                distance_multiplicities=(),
            )

    def test_anchor_arity_is_schema_expressed(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceRequest,
            PinnedLineDistanceResult,
        )

        for model in (PinnedLineDistanceRequest, PinnedLineDistanceResult):
            anchor_schema = model.model_json_schema()["properties"]["anchor"]
            assert anchor_schema["minItems"] == 2
            assert anchor_schema["maxItems"] == 2

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0)])
        for bad_anchor in (self._anchor(0, 0)[:1], (*self._anchor(0, 0), _cr(1))):
            with pytest.raises(ValidationError):
                PinnedLineDistanceRequest(configuration=cfg, anchor=bad_anchor)
            with pytest.raises(ValidationError):
                PinnedLineDistanceResult(
                    configuration=cfg,
                    anchor=bad_anchor,
                    dimension=2,
                    point_count=2,
                    lines=(),
                    distance_multiplicities=(),
                )

    def test_pair_ledger_is_schema_bounded_before_validation(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PAIRS,
            PinnedLineDistanceResult,
            PinnedLineEntry,
        )

        # Schema-visible bound derived from the mathematics: a bounded
        # configuration spans at most C(MAX_POINTS, 2) distinct pairs.
        assert MAX_PAIRS == 2016
        pairs_schema = PinnedLineEntry.model_json_schema()["properties"]["pairs"]
        assert pairs_schema["maxItems"] == MAX_PAIRS

        entry_schema = PinnedLineDistanceResult.model_json_schema()
        assert (
            entry_schema["$defs"]["PinnedLineEntry"]["properties"]["pairs"]["maxItems"]
            == MAX_PAIRS
        )

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0)])
        oversized = tuple((i, i + 1) for i in range(MAX_PAIRS + 1))
        with pytest.raises(ValidationError, match="at most 2016 items"):
            PinnedLineEntry(
                line_coefficients=(_cr(0), _cr(0), _cr(0)),
                squared_distance=_cr(1),
                pairs=oversized,
            )
        with pytest.raises(ValidationError, match="at most 2016 items"):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=2,
                lines=(
                    PinnedLineEntry(
                        line_coefficients=(_cr(0), _cr(1), _cr(0)),
                        squared_distance=_cr(1),
                        pairs=oversized,
                    ),
                ),
                distance_multiplicities=(),
            )

    def test_outer_result_ledgers_are_schema_bounded_before_validation(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PAIRS,
            PinnedLineDistanceResult,
            PinnedLineEntry,
        )

        result_schema = PinnedLineDistanceResult.model_json_schema()
        assert result_schema["properties"]["lines"]["maxItems"] == MAX_PAIRS
        assert (
            result_schema["properties"]["distance_multiplicities"]["maxItems"]
            == MAX_PAIRS
        )

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0)])
        cheap_entry = PinnedLineEntry(
            line_coefficients=(_cr(0), _cr(0), _cr(1)),
            squared_distance=_cr(0),
            pairs=((0, 1),),
        )
        with pytest.raises(ValidationError, match="at most 2016 items"):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=2,
                lines=tuple([cheap_entry] * (MAX_PAIRS + 1)),
                distance_multiplicities=(),
            )
        with pytest.raises(ValidationError, match="at most 2016 items"):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=2,
                lines=(),
                distance_multiplicities=tuple([(_cr(1), 1)] * (MAX_PAIRS + 1)),
            )

    def test_aggregate_output_budget_rejects_unencodable_profiles(self):
        from fractions import Fraction

        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PINNED_PROFILE_RESULT_BYTES,
            _maximum_pinned_profile_wire_bytes,
        )

        def big_pt(index: int) -> dict:
            fx = Fraction(1, 10**249 + index * 10**123 + 2 * index + 1)
            fy = Fraction(index + 1, 10**250 + index + 7)
            return {
                "label": f"p{index:02d}",
                "coordinates": [
                    {
                        "num": format_canonical_integer(fx.numerator),
                        "den": format_canonical_integer(fx.denominator),
                    },
                    {
                        "num": format_canonical_integer(fy.numerator),
                        "den": format_canonical_integer(fy.denominator),
                    },
                ],
            }

        cfg = PointConfiguration(
            points=tuple(
                LabelledRationalPoint.model_validate(big_pt(index))
                for index in range(64)
            )
        )
        anchor = (
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
        )
        assert (
            _maximum_pinned_profile_wire_bytes(cfg, anchor)
            > MAX_PINNED_PROFILE_RESULT_BYTES
        )
        with pytest.raises(ValidationError, match="aggregate result"):
            PinnedLineDistanceRequest(configuration=cfg, anchor=anchor)

    def test_estimate_dominates_actual_encoding_for_admitted_requests(self):
        from jacobian.canonical import encode_strict_json
        from jacobian.math.geometry.exact._models import (
            MAX_PINNED_PROFILE_RESULT_BYTES,
            _maximum_pinned_profile_wire_bytes,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 0, 1), ("d", 5, 3)])
        request = PinnedLineDistanceRequest(
            configuration=cfg,
            anchor=self._anchor(0, 0),
        )
        result = compute_pinned_line_distance_profile(request)
        actual = len(encode_strict_json(result.model_dump(mode="json")))
        assert actual <= MAX_PINNED_PROFILE_RESULT_BYTES
        assert actual <= _maximum_pinned_profile_wire_bytes(cfg, request.anchor)

    def test_result_budget_metadata_is_schema_published(self):
        from jacobian.math.geometry.exact._models import (
            COORDINATE_DIGITS,
            MAX_PINNED_PROFILE_RESULT_BYTES,
        )

        schema = PinnedLineDistanceRequest.model_json_schema()
        metadata = schema["properties"]["configuration"]
        assert metadata["coordinate_digit_bound"] == COORDINATE_DIGITS
        assert metadata["aggregate_result_budget_bytes"] == MAX_PINNED_PROFILE_RESULT_BYTES


def _cr(x):
    from fractions import Fraction

    from jacobian._exact import CanonicalRational

    return CanonicalRational.from_fraction(Fraction(x))
