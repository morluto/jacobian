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
