"""Tests for exact geometry operations."""

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry.exact._models import (
    COORDINATE_DIGITS,
    DistanceGraphRequest,
    DistanceProfileRequest,
    LabelledRationalPoint,
    PinnedLineDistanceRequest,
    PinnedLineDistanceResult,
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
        # Each entry carries a source pair, so an oversized ledger now trips
        # the earlier aggregate pair-ledger cap before the schema maxItems
        # bound is reached during field validation.
        with pytest.raises(ValidationError, match="aggregate source-pair ledger"):
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
        assert (
            metadata["aggregate_result_budget_bytes"] == MAX_PINNED_PROFILE_RESULT_BYTES
        )


def _cr(x):
    from fractions import Fraction

    from jacobian._exact import CanonicalRational

    return CanonicalRational.from_fraction(Fraction(x))


class TestAggregatePairLedgerBound:
    def test_oversized_aggregate_pair_ledger_rejected_before_parsing(self):
        """2016 entries each carrying 2016 distinct pairs would amplify into
        over four million pair tuples during parsing; the aggregate ledger
        must be rejected at the mathematical MAX_PAIRS bound instead."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceResult,
        )

        entry = {
            "line_coefficients": [
                {"num": "0", "den": "1"},
                {"num": "1", "den": "1"},
                {"num": "0", "den": "1"},
            ],
            "squared_distance": {"num": "1", "den": "2"},
            "pairs": [[i, i + 1] for i in range(64)],
        }
        payload = {
            "configuration": {
                "points": [
                    {
                        "label": f"p{i}",
                        "coordinates": [
                            {"num": str(i), "den": "1"},
                            {"num": str(i * i), "den": "1"},
                        ],
                    }
                    for i in range(3)
                ]
            },
            "anchor": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
            "dimension": 2,
            "point_count": 3,
            "lines": [entry for _ in range(40)],
            "distance_multiplicities": [],
        }
        with pytest.raises(ValidationError, match="aggregate source-pair ledger"):
            PinnedLineDistanceResult.model_validate(payload)

    def test_prevalidated_entries_counted_in_aggregate_cap(self):
        """Pre-validated ``PinnedLineEntry`` instances bypass dict-only
        counting, so a native caller could supply individually valid entries
        whose aggregate pair total far exceeds MAX_PAIRS; the cap must count
        already-parsed entries too before any replay work runs."""
        from fractions import Fraction
        from itertools import combinations

        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PAIRS,
            PinnedLineConfiguration,
            PinnedLineDistanceResult,
            PinnedLineEntry,
            PinnedLinePoint,
        )

        def _cr(value):
            return CanonicalRational.from_fraction(Fraction(value))

        cfg = PinnedLineConfiguration(
            points=tuple(
                PinnedLinePoint(label=f"p{i}", coordinates=(_cr(i), _cr(i * i)))
                for i in range(64)
            )
        )
        full_ledger = tuple((i, j) for i, j in combinations(range(64), 2))
        assert len(full_ledger) == MAX_PAIRS
        entries = tuple(
            PinnedLineEntry(
                line_coefficients=(_cr(0), _cr(1), _cr(-k)),
                squared_distance=_cr(1),
                pairs=full_ledger,
            )
            for k in range(2)
        )
        assert sum(len(entry.pairs) for entry in entries) > MAX_PAIRS
        with pytest.raises(ValidationError, match="aggregate source-pair ledger"):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=(_cr(0), _cr(0)),
                dimension=2,
                point_count=64,
                lines=entries,
                distance_multiplicities=(),
            )

    def test_mixed_dict_and_instance_entries_counted_in_aggregate_cap(self):
        from fractions import Fraction
        from itertools import combinations

        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PAIRS,
            PinnedLineConfiguration,
            PinnedLineDistanceResult,
            PinnedLineEntry,
            PinnedLinePoint,
        )

        def _cr(value):
            return CanonicalRational.from_fraction(Fraction(value))

        cfg = PinnedLineConfiguration(
            points=tuple(
                PinnedLinePoint(label=f"p{i}", coordinates=(_cr(i), _cr(i * i)))
                for i in range(64)
            )
        )
        full_ledger = tuple((i, j) for i, j in combinations(range(64), 2))
        assert len(full_ledger) == MAX_PAIRS
        half = full_ledger[: MAX_PAIRS // 2]
        instance_entry = PinnedLineEntry(
            line_coefficients=(_cr(0), _cr(1), _cr(-1)),
            squared_distance=_cr(1),
            pairs=full_ledger,
        )
        dict_entry = {
            "line_coefficients": [
                {"num": "0", "den": "1"},
                {"num": "1", "den": "1"},
                {"num": "-2", "den": "1"},
            ],
            "squared_distance": {"num": "1", "den": "1"},
            "pairs": [list(pair) for pair in half],
        }
        assert len(instance_entry.pairs) + len(dict_entry["pairs"]) > MAX_PAIRS
        with pytest.raises(ValidationError, match="aggregate source-pair ledger"):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=(_cr(0), _cr(0)),
                dimension=2,
                point_count=64,
                lines=(instance_entry, dict_entry),
                distance_multiplicities=(),
            )

    def test_valid_profile_ledger_roundtrips(self):
        from jacobian.math.geometry.exact._models import (
            LabelledRationalPoint,
            PinnedLineDistanceRequest,
            PinnedLineDistanceResult,
            PointConfiguration,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        def _cr(value):
            return {"num": str(value), "den": "1"}

        cfg = PointConfiguration(
            points=tuple(
                LabelledRationalPoint(label=label, coordinates=(_cr(x), _cr(y)))
                for label, x, y in [("a", 0, 0), ("b", 1, 0), ("c", 0, 1), ("d", 1, 1)]
            )
        )
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(
                configuration=cfg,
                anchor=({"num": "0", "den": "1"}, {"num": "0", "den": "1"}),
            )
        )
        total_pairs = sum(len(line.pairs) for line in result.lines)
        assert total_pairs <= 2016
        assert (
            PinnedLineDistanceResult.model_validate(result.model_dump(mode="json"))
            == result
        )


class TestSortedPairLedger:
    def _collinear_profile(self):
        from jacobian.math.geometry.exact._models import (
            LabelledRationalPoint,
            PinnedLineDistanceRequest,
            PointConfiguration,
        )
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        cfg = PointConfiguration(
            points=tuple(
                LabelledRationalPoint(label=label, coordinates=(_cr(x), _cr(y)))
                for label, x, y in [("a", 0, 0), ("b", 1, 0), ("c", 2, 0)]
            )
        )
        return compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(configuration=cfg, anchor=(_cr(0), _cr(1)))
        )

    def test_producer_emits_sorted_pair_ledgers(self):
        result = self._collinear_profile()
        assert len(result.lines) >= 1
        for entry in result.lines:
            assert entry.pairs == tuple(sorted(entry.pairs))

    def test_unsorted_pair_ledger_rejected(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import PinnedLineDistanceResult

        result = self._collinear_profile()
        payload = result.model_dump(mode="json")
        collinear_entry = max(
            range(len(payload["lines"])),
            key=lambda i: len(payload["lines"][i]["pairs"]),
        )
        assert len(payload["lines"][collinear_entry]["pairs"]) >= 3
        payload["lines"][collinear_entry]["pairs"] = [[1, 2], [0, 1], [0, 2]]
        with pytest.raises(ValidationError, match="must be sorted"):
            PinnedLineDistanceResult.model_validate(payload)

    def test_serialization_identity_preserved_for_authentic_results(self):
        from jacobian.math.geometry.exact._models import PinnedLineDistanceResult

        result = self._collinear_profile()
        assert (
            PinnedLineDistanceResult.model_validate(result.model_dump(mode="json"))
            == result
        )


class TestPinnedCoordinateCapIsSchemaEnforced:
    """The 256-digit coordinate cap must be published as standard,
    enforceable JSON Schema constraints (pattern/maxLength) on both the
    configuration coordinates and the anchor, without narrowing the shared
    point type (review thread: express the digit limit as a schema
    constraint)."""

    def _big_num(self, digits: int) -> str:
        return "9" * digits

    def _cfg(self, pts):
        from fractions import Fraction

        return PointConfiguration(
            points=tuple(
                LabelledRationalPoint(
                    label=label,
                    coordinates=(
                        CanonicalRational.from_fraction(Fraction(x)),
                        CanonicalRational.from_fraction(Fraction(y)),
                    ),
                )
                for label, x, y in pts
            ),
        )

    def _anchor(self, x, y):
        from fractions import Fraction

        return (
            CanonicalRational.from_fraction(Fraction(x)),
            CanonicalRational.from_fraction(Fraction(y)),
        )

    def test_schema_publishes_enforceable_component_bounds(self):

        schema = PinnedLineDistanceRequest.model_json_schema()
        defs = schema["$defs"]
        rational_def = defs["PinnedBoundedRational"]
        num = rational_def["properties"]["num"]
        den = rational_def["properties"]["den"]
        assert num["maxLength"] == COORDINATE_DIGITS + 1
        assert den["maxLength"] == COORDINATE_DIGITS
        assert f"{{0,{COORDINATE_DIGITS - 1}}}" in num["pattern"]
        # Both the configuration points and the anchor reference the bounded type.
        anchor_ref = schema["properties"]["anchor"]["items"]["$ref"]
        assert anchor_ref.endswith("/PinnedBoundedRational")
        point_ref = defs["PinnedLineConfiguration"]["properties"]["points"]["items"][
            "$ref"
        ]
        assert point_ref.endswith("/PinnedLinePoint")
        coord_items = defs["PinnedLinePoint"]["properties"]["coordinates"]["items"]
        assert coord_items["$ref"].endswith("/PinnedBoundedRational")
        # The result model publishes the same constraint.
        result_schema = PinnedLineDistanceResult.model_json_schema()
        result_anchor = result_schema["properties"]["anchor"]["items"]["$ref"]
        assert result_anchor.endswith("/PinnedBoundedRational")

    def test_over_cap_configuration_numerator_rejected_at_parse_time(self):
        import pytest
        from pydantic import ValidationError

        over = self._big_num(COORDINATE_DIGITS + 1)
        with pytest.raises(ValidationError):
            PinnedLineDistanceRequest(
                configuration={
                    "points": [
                        {"label": "a", "coordinates": [{"num": "0", "den": "1"}]},
                        {
                            "label": "b",
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": over, "den": "1"},
                            ],
                        },
                        {"label": "c", "coordinates": [{"num": "0", "den": "1"}]},
                    ]
                },
                anchor=({"num": "0", "den": "1"}, {"num": "0", "den": "1"}),
            )

    def test_over_cap_anchor_rejected_at_parse_time(self):
        import pytest
        from pydantic import ValidationError

        over = self._big_num(COORDINATE_DIGITS + 1)
        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 0, 1)])
        with pytest.raises(ValidationError):
            PinnedLineDistanceRequest(
                configuration=cfg,
                anchor=(
                    CanonicalRational(num="0", den="1"),
                    CanonicalRational(num=over, den="1"),
                ),
            )

    def test_boundary_cap_components_are_accepted(self):
        edge = CanonicalRational(num=self._big_num(COORDINATE_DIGITS), den="1")
        cfg = self._cfg(
            [("a", 0, 0), ("b", 1, 0), ("c", 0, 1)],
        )
        request = PinnedLineDistanceRequest(
            configuration=cfg, anchor=(edge, CanonicalRational(num="0", den="1"))
        )
        assert request.anchor[0].as_fraction().numerator == int(
            self._big_num(COORDINATE_DIGITS)
        )

    def test_shared_point_type_is_not_narrowed(self):
        """distance_profile/distance_graph keep the canonical component range;
        only the pinned-line view carries the 256-digit cap."""
        big = CanonicalRational(num=self._big_num(300), den="1")
        pts = (
            LabelledRationalPoint(label="a", coordinates=(big, big)),
            LabelledRationalPoint(label="b", coordinates=(big, big)),
        )
        configuration = PointConfiguration(points=pts)
        DistanceProfileRequest(configuration=configuration)

    def test_shared_configuration_value_composes_unchanged(self):
        """An existing shared PointConfiguration instance is accepted by the
        pinned request without translation."""
        cfg = self._cfg([("a", 0, 0), ("b", 1, 0), ("c", 0, 1)])
        request = PinnedLineDistanceRequest(
            configuration=cfg, anchor=self._anchor(0, 0)
        )
        assert request.point_count if hasattr(request, "point_count") else True
        assert len(request.configuration.points) == 3


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


class TestIncidenceSearchResultSourceBinding:
    def _point(self, label: str, *coords):
        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import LabelledRationalPoint

        return LabelledRationalPoint(
            label=label,
            coordinates=tuple(
                CanonicalRational.from_fraction(__import__("fractions").Fraction(c))
                for c in coords
            ),
        )

    def test_rejects_nonplanar_source_with_forged_collinear_witness(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            PointConfiguration,
        )

        configuration = PointConfiguration(
            points=(
                self._point("a", 0, 0, 0),
                self._point("b", 1, 0, 1),
                self._point("c", 2, 0, 0),
            )
        )
        with pytest.raises(ValidationError, match="dimension must match"):
            IncidenceSearchResult(
                configuration=configuration,
                dimension=2,
                point_count=3,
                holds=True,
                witnesses=((0, 1, 2),),
                kind="COLLINEAR_TRIPLE",
            )

    def test_rejects_one_dimensional_source_without_index_error(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            PointConfiguration,
        )

        configuration = PointConfiguration(
            points=(self._point("a", 0), self._point("b", 1), self._point("c", 2))
        )
        with pytest.raises(ValidationError, match="dimension must match"):
            IncidenceSearchResult(
                configuration=configuration,
                dimension=2,
                point_count=3,
                holds=False,
                witnesses=(),
                kind="COLLINEAR_TRIPLE",
            )

    def test_rejects_nonplanar_source_for_concyclic_kind(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            PointConfiguration,
        )

        configuration = PointConfiguration(
            points=(
                self._point("a", 0, 0, 0),
                self._point("b", 1, 0, 0),
                self._point("c", 0, 1, 0),
                self._point("d", 0, 0, 1),
            )
        )
        with pytest.raises(ValidationError, match="dimension must match"):
            IncidenceSearchResult(
                configuration=configuration,
                dimension=2,
                point_count=4,
                holds=True,
                witnesses=((0, 1, 2, 3),),
                kind="CONCYCLIC_QUADRUPLE",
            )

    def test_planar_result_round_trips(self):
        from jacobian.math.geometry.exact._models import (
            CollinearTriplesRequest,
            IncidenceSearchResult,
            PointConfiguration,
        )
        from jacobian.math.geometry.exact._operations import compute_collinear_triples

        configuration = PointConfiguration(
            points=(
                self._point("a", 0, 0),
                self._point("b", 1, 0),
                self._point("c", 2, 0),
            )
        )
        result = compute_collinear_triples(
            CollinearTriplesRequest(configuration=configuration)
        )
        replayed = IncidenceSearchResult.model_validate(result.model_dump())
        assert replayed == result
        assert replayed.holds is True
        assert replayed.witnesses == ((0, 1, 2),)


class TestIncidenceDistinctCoordinatesAndCap:
    def test_requests_reject_coordinate_coincident_points(self):
        """(1,0),(1,0),(0,1),(-1,0) all lie on the unit circle, but a
        repeated point makes the concyclicity guard skip the quadruple;
        coincident coordinates are therefore rejected at the boundary."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            CollinearTriplesRequest,
            ConcyclicQuadruplesRequest,
            PointConfiguration,
        )

        def _cr(value):
            return CanonicalRational(num=str(value), den="1")

        points = (
            LabelledRationalPoint(label="a", coordinates=(_cr(1), _cr(0))),
            LabelledRationalPoint(label="b", coordinates=(_cr(1), _cr(0))),
            LabelledRationalPoint(label="c", coordinates=(_cr(0), _cr(1))),
            LabelledRationalPoint(label="d", coordinates=(_cr(-1), _cr(0))),
        )
        with pytest.raises(ValidationError, match="distinct coordinates"):
            CollinearTriplesRequest(configuration=PointConfiguration(points=points))
        with pytest.raises(ValidationError, match="distinct coordinates"):
            ConcyclicQuadruplesRequest(configuration=PointConfiguration(points=points))

    def test_result_rejects_retained_configuration_over_coordinate_cap(self):
        """A deserialized result whose retained configuration carries
        beyond-cap rationals must be rejected before any replay work."""
        import pytest
        from pydantic import ValidationError

        from jacobian.canonical import format_canonical_integer
        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            PointConfiguration,
        )

        def _cr(value):
            return CanonicalRational(num=str(value), den="1")

        huge_num = format_canonical_integer(10**30000)
        huge = CanonicalRational(num=huge_num, den="1")
        points = (
            LabelledRationalPoint(label="a", coordinates=(huge, _cr(0))),
            LabelledRationalPoint(label="b", coordinates=(_cr(0), _cr(1))),
            LabelledRationalPoint(label="c", coordinates=(_cr(1), _cr(1))),
        )
        with pytest.raises(ValidationError, match="point 0 coordinate 0"):
            IncidenceSearchResult(
                configuration=PointConfiguration(points=points),
                dimension=2,
                point_count=3,
                holds=False,
                witnesses=(),
                kind="COLLINEAR_TRIPLE",
            )


class TestConcyclicWorkBound:
    @staticmethod
    def _height_config(n: int, digits: int) -> PointConfiguration:
        """n planar points whose coordinate height is exactly `digits`."""

        def cr(offset: int):
            return CanonicalRational(num=str(10 ** (digits - 1) + offset), den="1")

        return PointConfiguration(
            points=tuple(
                LabelledRationalPoint(
                    label=f"p{i}",
                    coordinates=(cr(i), cr(i + n)),
                )
                for i in range(n)
            )
        )

    def test_request_rejects_joint_budget_violation(self):
        """C(18,4)*22 = 67320 > 65536: within the point cap and the
        per-coordinate digit cap, but jointly too much enumeration work."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import ConcyclicQuadruplesRequest

        with pytest.raises(ValidationError, match="joint work budget"):
            ConcyclicQuadruplesRequest(configuration=self._height_config(18, 22))

    def test_request_rejects_former_24_point_bound(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_QUADRUPLE_SEARCH_POINTS,
            ConcyclicQuadruplesRequest,
        )

        assert MAX_QUADRUPLE_SEARCH_POINTS == 18
        with pytest.raises(ValidationError, match="enumeration bound"):
            ConcyclicQuadruplesRequest(configuration=self._height_config(19, 2))

    def test_request_admits_budget_boundary_configuration(self):
        from jacobian.math.geometry.exact._models import ConcyclicQuadruplesRequest

        request = ConcyclicQuadruplesRequest(configuration=self._height_config(18, 21))
        assert len(request.configuration.points) == 18

    def test_result_rejects_work_bound_before_replay(self):
        """A forged result must not pull budget-bypassing replay work; the
        joint bound fires before witness conversion or search replay."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            LabelledRationalPoint,
            PointConfiguration,
        )

        base = 10**63
        points = tuple(
            LabelledRationalPoint(
                label=f"p{i}",
                coordinates=(
                    CanonicalRational(num=str(base + i), den="1"),
                    CanonicalRational(num=str(base + i + 64), den="1"),
                ),
            )
            for i in range(18)
        )
        with pytest.raises(ValidationError, match="joint work budget"):
            IncidenceSearchResult(
                configuration=PointConfiguration(points=points),
                dimension=2,
                point_count=18,
                holds=False,
                witnesses=(),
                kind="CONCYCLIC_QUADRUPLE",
            )


class TestResultCardinalityMirrorsRequest:
    def _point(self, label: str, *coords):
        from jacobian._exact import CanonicalRational
        from jacobian.math.geometry.exact._models import LabelledRationalPoint

        return LabelledRationalPoint(
            label=label,
            coordinates=tuple(
                CanonicalRational.from_fraction(__import__("fractions").Fraction(c))
                for c in coords
            ),
        )

    def test_undersized_concyclic_result_rejected(self):
        """A CONCYCLIC_QUADRUPLE result retaining only three points can
        never come from the operation, whose request requires at least
        four; replay over three points enumerates nothing."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            IncidenceSearchResult,
            PointConfiguration,
        )

        configuration = PointConfiguration(
            points=(
                self._point("a", 0, 0),
                self._point("b", 1, 0),
                self._point("c", 0, 1),
            )
        )
        with pytest.raises(ValidationError, match="at least 4 retained points"):
            IncidenceSearchResult(
                configuration=configuration,
                dimension=2,
                point_count=3,
                holds=False,
                witnesses=(),
                kind="CONCYCLIC_QUADRUPLE",
            )

    def test_request_schemas_advertise_distinct_coordinates(self):
        """The pairwise-distinct coordinate rule is validator-enforced, so
        it must be visible in the advertised schema descriptions."""
        from jacobian.math.geometry.exact import _models

        for request_type in (
            _models.CollinearTriplesRequest,
            _models.ConcyclicQuadruplesRequest,
        ):
            schema = request_type.model_json_schema()
            model_description = (schema.get("description") or "").lower()
            field_description = (
                schema["properties"]["configuration"].get("description") or ""
            ).lower()
            assert "distinct" in model_description, request_type.__name__
            assert "distinct" in field_description, request_type.__name__


class TestIncidenceWitnessCanonicalOrder:
    def _collinear_result(self):
        from jacobian.math.geometry.exact._models import (
            CollinearTriplesRequest,
            PointConfiguration,
        )
        from jacobian.math.geometry.exact._operations import compute_collinear_triples

        def cr(value):
            return CanonicalRational.from_fraction(
                __import__("fractions").Fraction(value)
            )

        configuration = PointConfiguration(
            points=tuple(
                LabelledRationalPoint(label=label, coordinates=(cr(x), cr(y)))
                for label, x, y in (("a", 0, 0), ("b", 1, 0), ("c", 2, 0), ("d", 3, 0))
            )
        )
        return compute_collinear_triples(
            CollinearTriplesRequest(configuration=configuration)
        )

    def test_operation_emits_canonical_order(self):
        result = self._collinear_result()
        assert result.witnesses == tuple(sorted(result.witnesses))

    def test_reversed_complete_witness_list_rejected(self):
        """Reversing the four witnesses of four collinear points is the
        same exact result; the second serialization must not revalidate."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import IncidenceSearchResult

        result = self._collinear_result()
        with pytest.raises(ValidationError, match="canonically ordered"):
            IncidenceSearchResult(
                configuration=result.configuration,
                dimension=2,
                point_count=4,
                holds=True,
                witnesses=tuple(reversed(result.witnesses)),
                kind="COLLINEAR_TRIPLE",
            )

    def test_swapped_witness_permutation_rejected(self):
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import IncidenceSearchResult

        result = self._collinear_result()
        permuted = list(result.witnesses)
        permuted[0], permuted[1] = permuted[1], permuted[0]
        assert tuple(permuted) != tuple(sorted(permuted))
        with pytest.raises(ValidationError, match="canonically ordered"):
            IncidenceSearchResult(
                configuration=result.configuration,
                dimension=2,
                point_count=4,
                holds=True,
                witnesses=tuple(permuted),
                kind="COLLINEAR_TRIPLE",
            )
