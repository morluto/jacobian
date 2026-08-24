"""Behavioral and exact-expression tests for Euclidean polygon triangulation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry._euclidean_triangulation import (
    minimum_euclidean_weight_triangulation,
)
from jacobian.math.geometry._models import (
    MAX_EUCLIDEAN_TRIANGULATION_COORDINATE_DIGITS,
    MAX_EUCLIDEAN_TRIANGULATION_VERTICES,
    EuclideanConvexPolygonTriangulationRequest,
    EuclideanConvexPolygonTriangulationResult,
)


def _point(x: int, y: int) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": str(x), "den": "1"},
        "y": {"num": str(y), "den": "1"},
    }


def _fraction_point(x: Fraction, y: Fraction) -> dict[str, dict[str, str]]:
    return {
        "x": {"num": str(x.numerator), "den": str(x.denominator)},
        "y": {"num": str(y.numerator), "den": str(y.denominator)},
    }


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

    def test_unresolved_result_rejects_forged_expressions(self) -> None:
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

        with pytest.raises(ValidationError, match="equal their DP candidates"):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

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

        with pytest.raises(ValidationError, match="strictly inside its span"):
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

        with pytest.raises(ValidationError, match="subproblem span"):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_unresolved_claim_on_a_resolvable_recurrence_is_rejected(self) -> None:
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

        with pytest.raises(ValidationError, match="replayed recurrence"):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_certified_result_rejects_a_mutated_diagonal_length(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["diagonals"][0]["squared_length"] = {"num": "3", "den": "1"}

        with pytest.raises(ValidationError, match="optimum expression"):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_certified_result_rejects_a_mutated_source_polygon(self) -> None:
        result = minimum_euclidean_weight_triangulation(
            _request((_point(0, 0), _point(1, 0), _point(1, 1), _point(0, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["polygon"]["points"][3]["y"] = {"num": "2", "den": "1"}

        with pytest.raises(ValidationError, match="selected recurrence"):
            EuclideanConvexPolygonTriangulationResult.model_validate(payload)

    def test_rejects_a_nonconvex_polygon_before_arb(self) -> None:
        with pytest.raises(ValidationError, match="strict CCW convexity"):
            _request((_point(0, 0), _point(2, 0), _point(1, 1), _point(2, 2)))

    def test_rejects_a_self_intersecting_ring_despite_positive_turns(self) -> None:
        with pytest.raises(ValidationError, match="simple ring"):
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
        with pytest.raises(ValidationError, match="at least 4 items"):
            _request((_point(0, 0), _point(2, 0), _point(0, 2)))

    def test_request_rejects_a_ring_above_the_admitted_vertex_ceiling(self) -> None:
        with pytest.raises(ValidationError, match="at most 28 items"):
            _request(tuple(_point(index, index * index) for index in range(29)))

    def test_request_rejects_coordinates_beyond_the_digit_bound(self) -> None:
        scale = 10**MAX_EUCLIDEAN_TRIANGULATION_COORDINATE_DIGITS
        with pytest.raises(ValidationError, match="exceed the 32-digit bound"):
            _request(
                (
                    _point(0, 0),
                    {"x": {"num": str(scale), "den": "1"}, "y": {"num": "0", "den": "1"}},
                    {
                        "x": {"num": str(scale + 1), "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
                )
            )

    def test_schema_publishes_the_admitted_envelope_and_preconditions(self) -> None:
        schema = EuclideanConvexPolygonTriangulationRequest.model_json_schema()
        points = schema["$defs"]["EuclideanTriangulationPolygonRequest"][
            "properties"
        ]["points"]
        assert points["minItems"] == 4
        assert points["maxItems"] == MAX_EUCLIDEAN_TRIANGULATION_VERTICES == 28
        assert (
            points["coordinate_digit_bound"]
            == MAX_EUCLIDEAN_TRIANGULATION_COORDINATE_DIGITS
            == 32
        )
        assert "strictly convex" in points["description"]
        assert "simple" in points["description"]
        assert "32 decimal digits" in points["description"]
        description = schema.get("description", "")
        assert "4 to 28 vertices" in description
        assert "convexity and ring simplicity are enforced" in description

    def test_catalog_example_returns_a_replayable_public_result(self) -> None:
        catalog = Catalog.open()
        operation_id = "geometry.polygon.triangulation.minimum_euclidean_weight.compute"
        operation = catalog.operation(operation_id)
        assert operation is not None

        public_result = invoke_operation(
            operation_id, operation.examples[0].input, catalog
        )

        assert public_result.operation_id == operation_id
        validated = EuclideanConvexPolygonTriangulationResult.model_validate(
            public_result.output
        )
        assert validated.status == "CERTIFIED_OPTIMUM"
