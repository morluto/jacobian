"""Behavioral and exact-expression tests for Euclidean polygon triangulation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.math.geometry._euclidean_triangulation import (
    minimum_euclidean_weight_triangulation,
)
from jacobian.math.geometry._models import (
    MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS,
    MAX_EUCLIDEAN_TRIANGULATION_VERTICES,
    EuclideanConvexPolygonTriangulationRequest,
    EuclideanConvexPolygonTriangulationResult,
    _echoed_result_envelope_chars,
    _span_term_occurrences,
    _verify_euclidean_triangulation_claim,
)

_FLOOR_TERM_CHARS = 2 * (4 * 1 + 1) + 128


def _floor_estimate_chars(vertices: int) -> int:
    return (
        sum((vertices - span) * (span - 1) for span in range(1, vertices - 1))
        + 2 * (vertices - 3)
    ) * _FLOOR_TERM_CHARS


def _expected_vertex_ceiling() -> int:
    count = 4
    while _floor_estimate_chars(count + 1) <= 7_000_000:
        count += 1
    return count


def _point(x: int, y: int) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": str(x), "den": "1"},
        "y": {"num": str(y), "den": "1"},
    }


def _big_point(x: int, y: int) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": format_canonical_integer(x), "den": "1"},
        "y": {"num": format_canonical_integer(y), "den": "1"},
    }


def _fraction_point(x: Fraction, y: Fraction) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": str(x.numerator), "den": str(x.denominator)},
        "y": {"num": str(y.numerator), "den": str(y.denominator)},
    }


def _big_fraction_point(x: Fraction, y: Fraction) -> dict[str, dict[str, str]]:
    return {
        "x": {
            "num": format_canonical_integer(x.numerator),
            "den": format_canonical_integer(x.denominator),
        },
        "y": {
            "num": format_canonical_integer(y.numerator),
            "den": format_canonical_integer(y.denominator),
        },
    }


def _translated_parabola_ring(
    count: int, digits: int
) -> tuple[dict[str, dict[str, str]], ...]:
    # The review-thread shape: an anchored rational Q/P with huge components
    # translated along a parabola, so pairwise differences stay four digits
    # while every echoed coordinate carries ``digits``-digit components.
    multiple = 9001 * 8009
    denominator = ((10**digits + multiple - 1) // multiple) * multiple
    anchor = Fraction(denominator + 1, denominator)
    return tuple(
        _big_fraction_point(
            anchor + Fraction(139 * index, 9001),
            anchor + Fraction(2 * index * index, 8009),
        )
        for index in range(count)
    )


def _request(points: tuple[dict[str, dict[str, str]], ...]):
    return EuclideanConvexPolygonTriangulationRequest(polygon={"points": points})


class TestEuclideanTriangulation:
    def test_unit_square_returns_one_exact_diagonal_expression(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert result.comparison_precision_bits == 128
        assert tuple((edge.first, edge.second) for edge in result.diagonals) == (
            (1, 3),
        )
        assert result.optimum is not None
        assert tuple(term.as_fraction() for term in result.optimum.squared_lengths) == (
            Fraction(2),
        )
        assert len(result.triangles) == 2
        assert len(result.split_table) == 3

    def test_source_fixture_recovers_the_fan_without_decimal_costs(self) -> None:
        # The source coordinates are rational, not necessarily integral.
        positive = tuple(
            _fraction_point(Fraction(2**index - index, 5), Fraction(2**index + 1))
            for index in range(1, 14)
        )
        negative = tuple(
            _fraction_point(Fraction(2**index - index, 5), Fraction(-(2**index + 1)))
            for index in range(13, 0, -1)
        )
        # This source lists the upper chain clockwise.  Reverse the nonzero
        # vertices to satisfy the operation's explicit CCW convention.
        source_ring = tuple(reversed((*positive, _point(16000, 0), *negative)))
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), *source_ring))
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert tuple((edge.first, edge.second) for edge in result.diagonals) == tuple(
            (0, index) for index in range(2, 27)
        )
        assert result.optimum is not None
        assert len(result.optimum.squared_lengths) == 25
        assert all(term.as_fraction() > 0 for term in result.optimum.squared_lengths)

    def test_near_equal_nonidentical_diagonals_remain_unresolved(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _point(scale, 0),
                    _point(scale, 1),
                    _point(0, 2),
                )
            )
        )

        assert result.status == "COMPARISON_UNRESOLVED"
        assert result.optimum is None
        assert result.unresolved_comparison is not None
        assert result.unresolved_comparison.left_split == 2
        assert result.unresolved_comparison.right_split == 1

    def test_unresolved_result_round_trips_through_model_validate(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _point(scale, 0),
                    _point(scale, 1),
                    _point(0, 2),
                )
            )
        )
        assert result.unresolved_comparison is not None

        validated = EuclideanConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )

        assert validated.status == "COMPARISON_UNRESOLVED"
        assert validated.unresolved_comparison is not None
        assert validated.unresolved_comparison.start == 0
        assert validated.unresolved_comparison.end == 3
        assert validated.unresolved_comparison.left_split == 2
        assert validated.unresolved_comparison.right_split == 1

    def test_unresolved_claim_verifier_rejects_forged_expressions(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _point(scale, 0),
                    _point(scale, 1),
                    _point(0, 2),
                )
            )
        )
        payload = result.model_dump(mode="json")
        payload["unresolved_comparison"]["left"]["squared_lengths"] = [
            {"num": "5", "den": "1"}
        ]
        payload["unresolved_comparison"]["right"]["squared_lengths"] = [
            {"num": "7", "den": "1"}
        ]

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_unresolved_result_rejects_an_inverted_split_order(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _point(scale, 0),
                    _point(scale, 1),
                    _point(0, 2),
                )
            )
        )
        payload = result.model_dump(mode="json")
        payload["unresolved_comparison"]["left_split"] = 1
        payload["unresolved_comparison"]["right_split"] = 2

        with pytest.raises(ValidationError):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_unresolved_result_rejects_a_span_outside_the_polygon(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _point(scale, 0),
                    _point(scale, 1),
                    _point(0, 2),
                )
            )
        )
        payload = result.model_dump(mode="json")
        payload["unresolved_comparison"]["end"] = 4

        with pytest.raises(ValidationError):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_unresolved_claim_verifier_rejects_a_resolvable_recurrence(self) -> None:
        payload = {
            "status": "COMPARISON_UNRESOLVED",
            "polygon": {
                "points": (
                    _point(0, 0),
                    _point(1, 0),
                    _point(1, 1),
                    _point(0, 1),
                )
            },
            "vertex_count": 4,
            "comparison_precision_bits": 128,
            "unresolved_comparison": {
                "start": 0,
                "end": 3,
                "left_split": 2,
                "right_split": 1,
                "left": {"squared_lengths": [{"num": "2", "den": "1"}]},
                "right": {"squared_lengths": [{"num": "2", "den": "1"}]},
                "precision_bits": 128,
            },
        }

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_certified_claim_verifier_rejects_a_mutated_diagonal_length(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["diagonals"][0]["squared_length"] = {"num": "3", "den": "1"}
        payload["optimum"]["squared_lengths"] = [{"num": "3", "den": "1"}]

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_certified_claim_verifier_rejects_a_mutated_source_polygon(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["polygon"]["points"][3]["y"] = {"num": "2", "den": "1"}

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_unresolved_root_stops_before_a_cheaper_later_pivot(self) -> None:
        scale = 10**30
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(5 * scale, 5 * scale),
                    _point(5 * scale + 8, 8 * scale - 1),
                    _point(-3 * scale, 4 * scale),
                    _point(0, -scale),
                    _point(4 * scale, 0),
                )
            )
        )

        assert result.status == "COMPARISON_UNRESOLVED"
        assert result.unresolved_comparison is not None
        comparison = result.unresolved_comparison
        assert (comparison.start, comparison.end) == (0, 4)
        assert (comparison.left_split, comparison.right_split) == (2, 1)

    def test_certified_result_rejects_a_pivot_beyond_an_unresolved_comparison(
        self,
    ) -> None:
        # The first two root candidates differ by about 65/(2*sqrt(65))*10**-30
        # and stay unresolved at 128 bits, while the third pivot is much
        # cheaper; execution therefore never compares against it.
        scale = 10**30
        points = (
            _point(5 * scale, 5 * scale),
            _point(5 * scale + 8, 8 * scale - 1),
            _point(-3 * scale, 4 * scale),
            _point(0, -scale),
            _point(4 * scale, 0),
        )
        assert (
            minimum_euclidean_weight_triangulation(_request(points)).status
            == "COMPARISON_UNRESOLVED"
        )
        vertices = tuple(
            (
                Fraction(int(point["x"]["num"]), int(point["x"]["den"])),
                Fraction(int(point["y"]["num"]), int(point["y"]["den"])),
            )
            for point in points
        )

        def squared_length(first: int, second: int) -> Fraction:
            return (vertices[second][0] - vertices[first][0]) ** 2 + (
                vertices[second][1] - vertices[first][1]
            ) ** 2

        def expression(values: tuple[Fraction, ...]) -> dict[str, object]:
            return {
                "squared_lengths": [
                    {"num": str(value.numerator), "den": str(value.denominator)}
                    for value in values
                ]
            }

        def rational(value: Fraction) -> dict[str, str]:
            return {"num": str(value.numerator), "den": str(value.denominator)}

        # Replay the recurrence but force the root choice onto the cheaper
        # third pivot, producing every otherwise-consistent certificate field.
        def hull_edge(start: int, end: int) -> bool:
            return end == start + 1 or (start, end) == (0, 4)

        optimum: dict[tuple[int, int], tuple[Fraction, ...]] = {
            (index, index + 1): () for index in range(4)
        }
        splits: dict[tuple[int, int], int] = {}
        ledger = []
        for span in range(2, 5):
            for start in range(5 - span):
                end = start + span
                boundary = (
                    () if hull_edge(start, end) else (squared_length(start, end),)
                )
                candidates = {
                    pivot: tuple(
                        sorted(optimum[start, pivot] + optimum[pivot, end] + boundary)
                    )
                    for pivot in range(start + 1, end)
                }
                chosen, split = min(
                    (candidate, pivot) for pivot, candidate in candidates.items()
                )
                if (start, end) == (0, 4):
                    chosen, split = candidates[3], 3
                optimum[start, end] = chosen
                splits[start, end] = split
                ledger.append(
                    {
                        "start": start,
                        "end": end,
                        "split": split,
                        "optimum": expression(chosen),
                    }
                )
        triangles: list[list[int]] = []
        diagonals: list[tuple[int, int]] = []

        def reconstruct(start: int, end: int) -> None:
            if end == start + 1:
                return
            pivot = splits[start, end]
            triangles.append(sorted((start, pivot, end)))
            if not hull_edge(start, end):
                diagonals.append((start, end))
            reconstruct(start, pivot)
            reconstruct(pivot, end)

        reconstruct(0, 4)
        payload = {
            "status": "CERTIFIED_OPTIMUM",
            "polygon": {"points": points},
            "vertex_count": 5,
            "comparison_precision_bits": 128,
            "diagonals": [
                {
                    "first": first,
                    "second": second,
                    "squared_length": rational(squared_length(first, second)),
                }
                for first, second in sorted(diagonals)
            ],
            "triangles": [{"vertices": triangle} for triangle in sorted(triangles)],
            "split_table": ledger,
            "optimum": expression(optimum[0, 4]),
        }

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_certified_claim_verifier_rejects_a_later_equal_pivot(self) -> None:
        # Both root pivots cost exactly sqrt(2); execution canonically keeps
        # the earlier one, so a certificate claiming the later pivot fails.
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["split_table"][2]["split"] = 2
        payload["diagonals"] = [
            {"first": 0, "second": 2, "squared_length": {"num": "2", "den": "1"}}
        ]
        payload["triangles"] = [
            {"vertices": [0, 1, 2]},
            {"vertices": [0, 2, 3]},
        ]

        forged = EuclideanConvexPolygonTriangulationResult.model_validate(payload)
        with pytest.raises(ValidationError):
            _verify_euclidean_triangulation_claim(forged)

    def test_rejects_a_nonconvex_polygon_before_arb(self) -> None:
        with pytest.raises(ValidationError):
            _request((_point(0, 0), _point(2, 0), _point(1, 1), _point(2, 2)))

    def test_rejects_a_self_intersecting_ring_despite_positive_turns(self) -> None:
        with pytest.raises(ValidationError):
            _request(
                (
                    _point(0, 3),
                    _point(-2, -3),
                    _point(3, 1),
                    _point(-3, 1),
                    _point(2, -3),
                )
            )

    def test_request_rejects_a_triangle_below_the_admitted_vertex_floor(self) -> None:
        with pytest.raises(ValidationError):
            _request((_point(0, 0), _point(2, 0), _point(0, 2)))

    def test_request_admits_a_ring_past_the_former_fixed_vertex_ceiling(
        self,
    ) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request(tuple(_point(index, index * index) for index in range(29)))
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert result.vertex_count == 29
        assert len(result.split_table) == (29 - 1) * (29 - 2) // 2
        assert len(result.diagonals) == result.vertex_count - 3
        assert len(result.triangles) == result.vertex_count - 2

    def test_ring_past_the_former_ceiling_round_trips_through_model_validate(
        self,
    ) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request(tuple(_point(index, index * index) for index in range(29)))
        )

        validated = EuclideanConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )

        assert validated.vertex_count == 29
        assert validated.split_table == result.split_table
        assert validated.optimum == result.optimum

    def test_request_rejects_a_ring_beyond_the_derived_vertex_ceiling(self) -> None:
        with pytest.raises(
            ValidationError,
        ):
            _request(
                tuple(
                    _point(index, index * index)
                    for index in range(MAX_EUCLIDEAN_TRIANGULATION_VERTICES + 1)
                )
            )

    def test_request_admits_a_far_translated_unit_square(self) -> None:
        scale = 10**32 + 7
        plain = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        shifted = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(scale, 0),
                    _point(scale + 1, 0),
                    _point(scale + 1, 1),
                    _point(scale, 1),
                )
            )
        )

        assert shifted.status == "CERTIFIED_OPTIMUM"
        assert shifted.optimum is not None
        assert plain.optimum is not None
        assert shifted.split_table == plain.split_table
        assert tuple((edge.first, edge.second) for edge in shifted.diagonals) == (
            (1, 3),
        )
        assert tuple(
            edge.squared_length.as_fraction() for edge in shifted.diagonals
        ) == (Fraction(2),)
        assert shifted.optimum.squared_lengths == plain.optimum.squared_lengths
        assert tuple(
            term.as_fraction() for term in shifted.optimum.squared_lengths
        ) == (Fraction(2),)

    def test_request_admits_a_square_scaled_past_the_integer_string_limit(
        self,
    ) -> None:
        scale = 10**5000
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _big_point(scale, 0),
                    _big_point(scale, 1),
                    _point(0, 1),
                )
            )
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert result.optimum is not None
        assert tuple((edge.first, edge.second) for edge in result.diagonals) == (
            (1, 3),
        )
        assert tuple(term.as_fraction() for term in result.optimum.squared_lengths) == (
            Fraction(scale * scale + 1),
        )

    def test_request_rejects_unrepresentable_squared_lengths_at_admission(
        self,
    ) -> None:
        scale = 10**20000
        with pytest.raises(ValidationError):
            _request(
                (
                    _point(0, 0),
                    _big_point(scale, 0),
                    _big_point(scale, 1),
                    _point(0, 1),
                )
            )

    def test_request_admits_squared_lengths_inside_the_canonical_rational_cap(
        self,
    ) -> None:
        scale = 10**9000
        result = minimum_euclidean_weight_triangulation(
            _request(
                (
                    _point(0, 0),
                    _big_point(scale, 0),
                    _big_point(scale, 1),
                    _point(0, 1),
                )
            )
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert result.optimum is not None
        assert tuple(term.as_fraction() for term in result.optimum.squared_lengths) == (
            Fraction(scale * scale + 1),
        )

    def test_request_rejects_extent_beyond_the_derived_output_bound(self) -> None:
        spread = 10**90
        with pytest.raises(ValidationError):
            _request(
                tuple(
                    _point(index * spread, index * index * spread)
                    for index in range(38)
                )
            )

    def test_request_rejects_a_root_optimum_expression_beyond_the_output_bound(
        self,
    ) -> None:
        # Regression: counting only non-root table spans estimated this
        # 10-vertex ring at 6,588,512 characters; the omitted root entry and
        # its duplicated optimum raise the worst case to 7,412,076, over the
        # published bound.
        scale = 10**7335
        assert (2 * (4 * 7337 + 1) + 128) * (
            _span_term_occurrences(10) - 2 * (10 - 3)
        ) == 6_588_512
        assert (2 * (4 * 7337 + 1) + 128) * _span_term_occurrences(10) == 7_412_076
        with pytest.raises(ValidationError):
            _request(
                tuple(
                    {
                        "x": {"num": str(index), "den": "1"},
                        "y": {
                            "num": format_canonical_integer(index * index * scale),
                            "den": "1",
                        },
                    }
                    for index in range(10)
                )
            )

    def test_request_admits_a_ring_sized_by_span_specific_term_counts(self) -> None:
        # A strict convex (i, i^2) ring of 49 vertices carries four-digit
        # pairwise differences: charging every DP state the root's 46 terms
        # rejected it, while the span-specific sum keeps its serialized
        # split table inside the output budget.
        result = minimum_euclidean_weight_triangulation(
            _request(tuple(_point(index, index * index) for index in range(49)))
        )

        assert result.status == "CERTIFIED_OPTIMUM"
        assert result.vertex_count == 49
        assert len(result.split_table) == (49 - 1) * (49 - 2) // 2
        assert len(result.diagonals) == result.vertex_count - 3

    def test_request_rejects_a_translation_ring_whose_echo_exceeds_the_output_bound(
        self,
    ) -> None:
        # Regression: the estimate charged only the split table, whose
        # serialized size is invariant under translation. The review-thread
        # ring keeps every pairwise difference at four digits - base
        # estimate 6,759,288 characters - so a ~32768-digit anchored source
        # passed admission, ran the kernel, and only then exceeded the
        # canonical output limit on its echoed polygon. Admission now
        # measures the echoed source and result metadata directly.
        assert _span_term_occurrences(64) * (2 * (4 * 4 + 1) + 128) == 6_759_288
        with pytest.raises(ValidationError):
            _request(_translated_parabola_ring(64, 1200))

    def test_request_admits_a_translation_ring_on_the_refined_envelope_boundary(
        self,
    ) -> None:
        # The same shape at an 801-digit anchor: the split-table share alone
        # stays at 6,759,288 characters, and the measured echo keeps the
        # whole deterministic envelope just inside the published bound.
        request = _request(_translated_parabola_ring(64, 800))

        assert len(request.polygon.points) == 64

    def test_request_rejects_a_translation_ring_one_step_past_the_refined_boundary(
        self,
    ) -> None:
        # A 901-digit anchor adds about twenty-six thousand echo characters
        # and crosses the published bound by roughly six thousand, so the
        # boundary is tight rather than a coarse fallback.
        with pytest.raises(ValidationError):
            _request(_translated_parabola_ring(64, 900))

    def test_translated_source_completes_inside_the_published_result_bound(
        self,
    ) -> None:
        # Translation still composes end to end when the complete envelope
        # fits: an 8001-digit translation of the (i, i**2) ring is admitted,
        # produces the untranslated split table, and its canonical output
        # stays within both the admission estimate and every transport limit.
        plain = minimum_euclidean_weight_triangulation(
            _request(tuple(_point(index, index * index) for index in range(29)))
        )
        scale = 10**8000
        shifted = minimum_euclidean_weight_triangulation(
            _request(
                tuple(
                    _big_point(scale + index, scale + index * index)
                    for index in range(29)
                )
            )
        )

        assert shifted.status == "CERTIFIED_OPTIMUM"
        assert shifted.split_table == plain.split_table
        encoded = canonicalize_json(shifted.model_dump(mode="json"))
        term_chars = 2 * (4 * 3 + 1) + 128
        estimate = (
            _span_term_occurrences(29) * term_chars
            + _echoed_result_envelope_chars(shifted.polygon)
            + (29 - 3) * term_chars
            + (29 - 2) * 32
            + 2 * term_chars
        )
        assert len(encoded) <= estimate
        assert estimate <= MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS
        assert len(encoded) <= 10 * 1024 * 1024

    def test_schema_publishes_the_admitted_envelope_and_preconditions(self) -> None:
        schema = EuclideanConvexPolygonTriangulationRequest.model_json_schema()
        points = schema["$defs"]["EuclideanTriangulationPolygonRequest"]["properties"][
            "points"
        ]
        assert points["minItems"] == 4
        assert (
            points["maxItems"]
            == MAX_EUCLIDEAN_TRIANGULATION_VERTICES
            == _expected_vertex_ceiling()
        )
        assert points["maxItems"] >= 29
        assert points["maximum_serialized_result_characters"] == 7_000_000
        assert "strictly convex" in points["description"]
        assert "simple" in points["description"]
        assert "echoed source ring" in points["description"]
        description = schema.get("description", "")
        assert f"4 to {MAX_EUCLIDEAN_TRIANGULATION_VERTICES} vertices" in description
        assert "convexity and ring simplicity are enforced" in description

    def test_certified_result_round_trips_through_model_validate(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(3, 0), _point(2, 2), _point(0, 1)))
        )
        assert result.status == "CERTIFIED_OPTIMUM"

        validated = EuclideanConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )

        assert validated.status == "CERTIFIED_OPTIMUM"
        assert validated.optimum is not None
