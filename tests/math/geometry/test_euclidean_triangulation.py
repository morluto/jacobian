"""Behavioral and exact-expression tests for Euclidean polygon triangulation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._euclidean_triangulation import (
    minimum_euclidean_weight_triangulation,
)
from jacobian.math.geometry._models import (
    MAX_EUCLIDEAN_TRIANGULATION_VERTICES,
    EuclideanConvexPolygonTriangulationRequest,
    EuclideanConvexPolygonTriangulationResult,
)


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


def _request(
    points: tuple[dict[str, dict[str, str]], ...],
) -> EuclideanConvexPolygonTriangulationRequest:
    return EuclideanConvexPolygonTriangulationRequest.model_validate(
        {"polygon": {"points": points}}
    )


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

    def test_rejects_a_nonconvex_polygon_before_arb(self) -> None:
        request = _request((_point(0, 0), _point(2, 0), _point(1, 1), _point(2, 2)))
        with pytest.raises(OperationDomainValidationError):
            minimum_euclidean_weight_triangulation(request)

    def test_rejects_a_self_intersecting_ring_despite_positive_turns(self) -> None:
        request = _request(
            (
                _point(0, 3),
                _point(-2, -3),
                _point(3, 1),
                _point(-3, 1),
                _point(2, -3),
            )
        )
        with pytest.raises(OperationDomainValidationError):
            minimum_euclidean_weight_triangulation(request)

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
        with pytest.raises(ValidationError):
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

    @pytest.mark.scale
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

    @pytest.mark.scale
    def test_request_rejects_unrepresentable_squared_lengths_at_admission(
        self,
    ) -> None:
        scale = 10**20000
        request = _request(
            (
                _point(0, 0),
                _big_point(scale, 0),
                _big_point(scale, 1),
                _point(0, 1),
            )
        )
        with pytest.raises(OperationDomainValidationError):
            minimum_euclidean_weight_triangulation(request)

    @pytest.mark.scale
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

    @pytest.mark.scale
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

    @pytest.mark.scale
    def test_request_admits_a_translation_ring_on_the_refined_envelope_boundary(
        self,
    ) -> None:
        # The same shape at an 801-digit anchor: the split-table share alone
        # stays at 6,759,288 characters, and the measured echo keeps the
        # whole deterministic envelope just inside the published bound.
        request = _request(_translated_parabola_ring(64, 800))

        assert len(request.polygon.points) == 64

    @pytest.mark.scale
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

    def test_schema_publishes_the_admitted_envelope_and_preconditions(self) -> None:
        schema = EuclideanConvexPolygonTriangulationRequest.model_json_schema()
        points = schema["$defs"]["EuclideanTriangulationPolygonRequest"]["properties"][
            "points"
        ]
        assert points["minItems"] == 4
        assert points["maxItems"] == MAX_EUCLIDEAN_TRIANGULATION_VERTICES == 68
        assert points["maxItems"] >= 29
        assert "strictly convex" in points["description"]
        assert "simple" in points["description"]
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
