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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
            PinnedLineEntry(
                line_coefficients=(_cr(0), _cr(0), _cr(0)),
                squared_distance=_cr(1),
                pairs=oversized,
            )
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
            PinnedLineDistanceResult(
                configuration=cfg,
                anchor=self._anchor(0, 0),
                dimension=2,
                point_count=2,
                lines=tuple([cheap_entry] * (MAX_PAIRS + 1)),
                distance_multiplicities=(),
            )
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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


class TestCanonicalPointValueComposition:
    """The exact-geometry domain composes through the one canonical
    point-configuration value: every surviving operation accepts a shared
    ``PointConfiguration`` unchanged and no operation-local model recreates
    the canonical point types (review: reuse the canonical type)."""

    @staticmethod
    def _configuration() -> PointConfiguration:
        return PointConfiguration(
            points=(
                LabelledRationalPoint(label="a", coordinates=(_cr(0), _cr(0))),
                LabelledRationalPoint(label="b", coordinates=(_cr(1), _cr(0))),
                LabelledRationalPoint(label="c", coordinates=(_cr(0), _cr(1))),
            )
        )

    def test_every_surviving_operation_accepts_canonical_configuration(self):
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        configuration = self._configuration()
        profile = compute_distance_profile(
            DistanceProfileRequest(configuration=configuration)
        )
        graph = compute_distance_graph(
            DistanceGraphRequest(
                configuration=configuration,
                target_squared_distance=CanonicalRational(num="1", den="1"),
            )
        )
        pinned = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(
                configuration=configuration, anchor=(_cr(0), _cr(0))
            )
        )
        assert profile.point_count == 3
        assert graph.vertex_count == 3
        assert pinned.point_count == 3

    def test_retained_configuration_feeds_sibling_operations_unchanged(self):
        """A producer's retained configuration is the canonical domain value,
        so consumers accept it without translation through a second type."""
        from jacobian.math.geometry.exact._operations import (
            compute_pinned_line_distance_profile,
        )

        configuration = self._configuration()
        result = compute_pinned_line_distance_profile(
            PinnedLineDistanceRequest(
                configuration=configuration, anchor=(_cr(0), _cr(1))
            )
        )
        # The retained value is the shared configuration (the pinned request's
        # schema view is a subclass, never a parallel recreation).
        assert isinstance(result.configuration, PointConfiguration)
        replayed = PinnedLineDistanceResult.model_validate(
            result.model_dump(mode="json")
        )
        assert replayed == result
        # A consumer accepts the producer's retained configuration directly
        # and computes exactly what it would from the original source value.
        profile = compute_distance_profile(
            DistanceProfileRequest(configuration=result.configuration)
        )
        assert profile == compute_distance_profile(
            DistanceProfileRequest(configuration=configuration)
        )
        graph = compute_distance_graph(
            DistanceGraphRequest(
                configuration=result.configuration,
                target_squared_distance=CanonicalRational(num="1", den="1"),
            )
        )
        assert graph == compute_distance_graph(
            DistanceGraphRequest(
                configuration=configuration,
                target_squared_distance=CanonicalRational(num="1", den="1"),
            )
        )

    def test_incidence_projections_no_longer_define_local_value_models(self):
        """The removed duplicate family took its recreated CanonicalRational /
        LabelledRationalPoint / PointConfiguration views with it."""
        import jacobian.math.geometry.exact._models as models

        for name in (
            "IncidenceBoundedRational",
            "IncidencePoint",
            "IncidencePointConfiguration",
            "CollinearTriplesRequest",
            "ConcyclicQuadruplesRequest",
            "IncidenceSearchResult",
        ):
            assert not hasattr(models, name), name


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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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
        with pytest.raises(ValidationError):
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


class TestAuthoredComponentBudget:
    def test_forged_rational_components_rejected_before_parsing(self):
        """Multi-megabyte authored coefficients are rejected pre-parse."""
        import pytest
        from pydantic import ValidationError

        # Just above the aggregate result budget: every valid entry needs at
        # least two characters per canonical rational, so this much raw
        # numerator text cannot belong to an admissible profile.
        from jacobian.math.geometry.exact._models import (
            MAX_PINNED_PROFILE_RESULT_BYTES as _MAX_BYTES,
        )
        from jacobian.math.geometry.exact._models import (
            PinnedLineDistanceResult,
        )

        huge = "1" * (_MAX_BYTES + 1)
        payload = {
            "dimension": 2,
            "point_count": 2,
            "lines": [
                {
                    "line_coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    "squared_distance": {"num": huge, "den": "1"},
                    "pairs": [[0, 1]],
                }
            ],
            "distance_multiplicities": [
                [{"num": "1", "den": "1"}, 1],
            ],
        }
        with pytest.raises(ValidationError):
            PinnedLineDistanceResult.model_validate(payload)


class TestAuthoredComponentCoverage:
    def test_oversized_line_coefficient_rejected_before_parsing(self):
        """Line coefficients count toward the pre-parse aggregate budget."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PINNED_PROFILE_RESULT_BYTES,
            PinnedLineDistanceResult,
        )

        huge = "1" * (MAX_PINNED_PROFILE_RESULT_BYTES + 1)
        payload = {
            "dimension": 2,
            "point_count": 2,
            "lines": [
                {
                    "line_coefficients": [
                        {"num": huge, "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    "squared_distance": {"num": "1", "den": "1"},
                    "pairs": [[0, 1]],
                }
            ],
            "distance_multiplicities": [[{"num": "1", "den": "1"}, 1]],
        }
        with pytest.raises(ValidationError):
            PinnedLineDistanceResult.model_validate(payload)

    def test_oversized_multiplicity_rational_rejected_before_parsing(self):
        """Distance-multiplicity rationals count toward the pre-parse bound."""
        import pytest
        from pydantic import ValidationError

        from jacobian.math.geometry.exact._models import (
            MAX_PINNED_PROFILE_RESULT_BYTES,
            PinnedLineDistanceResult,
        )

        huge = "1" * (MAX_PINNED_PROFILE_RESULT_BYTES + 1)
        payload = {
            "dimension": 2,
            "point_count": 2,
            "lines": [
                {
                    "line_coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    "squared_distance": {"num": "1", "den": "1"},
                    "pairs": [[0, 1]],
                }
            ],
            "distance_multiplicities": [[{"num": huge, "den": "1"}, 1]],
        }
        with pytest.raises(ValidationError):
            PinnedLineDistanceResult.model_validate(payload)
