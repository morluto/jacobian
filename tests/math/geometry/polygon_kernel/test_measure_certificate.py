"""Exact measure certificates over replayed visibility kernels (#974).

Covers the frozen Nakano pentagon oracle (areas, perimeters, ratios, gap),
degenerate kernels, irrational-edge absence, caller comparisons, verifier
replay, and admission boundaries.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
)
from jacobian.math.geometry.polygon_kernel import (
    measure_certificate,
    verify_measure_certificate,
    visibility_kernel,
)
from jacobian.math.geometry.polygon_kernel._models import (
    KernelPolygon,
    MeasureCertificateRequest,
    MeasureCertificateResult,
    MeasureComparison,
    PolygonKernelResult,
)
from jacobian.math.geometry.polygon_kernel._tools import TOOLS


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _point(x: int | Fraction, y: int | Fraction) -> dict[str, object]:
    x_value, y_value = Fraction(x), Fraction(y)
    return {
        "x": {"num": x_value.numerator, "den": x_value.denominator},
        "y": {"num": y_value.numerator, "den": y_value.denominator},
    }


def _polygon(
    points: tuple[tuple[int | Fraction, int | Fraction], ...],
) -> KernelPolygon:
    return KernelPolygon.model_validate({"points": [_point(x, y) for x, y in points]})


NAKANO_PENTAGON = ((0, 4620), (0, -4620), (23100, -385), (22176, 0), (23100, 385))

NAKANO_KERNEL_VERTICES = (
    (0, -4620),
    (19800, -990),
    (22176, 0),
    (19800, 990),
    (0, 4620),
)


def _nakano_kernel() -> PolygonKernelResult:
    return visibility_kernel(_polygon(NAKANO_PENTAGON))


class TestFrozenOracle:
    def test_nakano_areas_reflex_and_kernel_vertices(self) -> None:
        kernel = _nakano_kernel()

        assert kernel.kernel_dimension == "POLYGON"
        assert kernel.reflex_vertex_indices == (3,)
        assert kernel.polygon_area.as_fraction() == 115259760
        assert kernel.kernel_area.as_fraction() == 113430240
        assert kernel.convex_hull_area.as_fraction() == 115615500
        assert kernel.kernel_to_polygon_area_ratio.as_fraction() == Fraction(62, 63)
        assert [
            (row.point.x.as_fraction(), row.point.y.as_fraction())
            for row in kernel.kernel_boundary
        ] == [(Fraction(x), Fraction(y)) for x, y in NAKANO_KERNEL_VERTICES]

    def test_nakano_perimeters_and_edge_lengths(self) -> None:
        result = measure_certificate(_nakano_kernel(), ())

        assert result.polygon_measures.perimeter is not None
        assert result.polygon_measures.perimeter.as_fraction() == 58212
        assert result.hull_measures.perimeter is not None
        assert result.hull_measures.perimeter.as_fraction() == 56980
        assert result.kernel_measures.perimeter is not None
        assert all(
            entry.exact_length is not None for entry in result.polygon_measures.edges
        )
        assert verify_measure_certificate(result)

    def test_nakano_ratio_gap(self) -> None:
        kernel = _nakano_kernel()
        result = measure_certificate(
            kernel,
            (
                MeasureComparison(
                    left="KERNEL_TO_POLYGON_AREA_RATIO",
                    right="POLYGON_TO_HULL_AREA_RATIO",
                    operator="LT",
                    expected_difference=CanonicalRational.from_fraction(
                        Fraction(-262, 20475)
                    ),
                ),
            ),
        )
        (outcome,) = result.comparisons

        assert outcome.difference.as_fraction() == Fraction(-262, 20475)
        assert outcome.holds
        # The published gap: G = |K|/|F| = 62/63, P = Per(C)/Per(F) = 185/189.
        ratio = kernel.kernel_to_polygon_area_ratio.as_fraction()
        assert result.hull_measures.perimeter is not None
        assert result.polygon_measures.perimeter is not None
        perimeter_ratio = (
            result.hull_measures.perimeter.as_fraction()
            / result.polygon_measures.perimeter.as_fraction()
        )
        assert ratio == Fraction(62, 63)
        assert perimeter_ratio == Fraction(185, 189)
        assert ratio - perimeter_ratio == Fraction(1, 189) > 0

    def test_kernel_strictly_inside_polygon(self) -> None:
        result = measure_certificate(
            _nakano_kernel(),
            (
                MeasureComparison(
                    left="KERNEL_AREA", right="POLYGON_AREA", operator="LT"
                ),
            ),
        )
        (outcome,) = result.comparisons

        assert outcome.holds
        assert outcome.difference.as_fraction() < 0


class TestConvexAndDegenerate:
    def test_convex_polygon_kernel_is_itself(self) -> None:
        square = ((0, 0), (2, 0), (2, 2), (0, 2))
        result = measure_certificate(visibility_kernel(_polygon(square)), ())

        assert result.kernel_measures.perimeter is not None
        assert result.kernel_measures.perimeter.as_fraction() == 8
        assert result.polygon_measures.perimeter is not None
        assert result.polygon_measures.perimeter.as_fraction() == 8
        assert result.kernel.polygon_area.as_fraction() == 4
        assert result.kernel.kernel_to_polygon_area_ratio.as_fraction() == 1

    def test_irrational_edges_stay_absent(self) -> None:
        # The hypotenuse sqrt(2) is exact as a square but has no rational root:
        # its length and the triangle perimeter stay absent, never promoted.
        triangle = ((0, 0), (1, 0), (0, 1))
        result = measure_certificate(visibility_kernel(_polygon(triangle)), ())

        assert [
            entry.squared_length.as_fraction()
            for entry in result.polygon_measures.edges
        ] == [
            Fraction(1),
            Fraction(2),
            Fraction(1),
        ]
        assert [entry.exact_length for entry in result.polygon_measures.edges] == [
            result.polygon_measures.edges[0].exact_length,
            None,
            result.polygon_measures.edges[2].exact_length,
        ]
        assert result.polygon_measures.perimeter is None
        assert verify_measure_certificate(result)

    def test_comparison_over_missing_measure_is_rejected(self) -> None:
        triangle = ((0, 0), (1, 0), (0, 1))
        kernel = visibility_kernel(_polygon(triangle))

        with pytest.raises(OperationDomainValidationError) as exc_info:
            measure_certificate(
                kernel,
                (
                    MeasureComparison(
                        left="POLYGON_PERIMETER",
                        right="HULL_PERIMETER",
                        operator="LT",
                    ),
                ),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "geometry.measure_comparison_unavailable"
        )

    def test_empty_kernel_has_no_ring(self) -> None:
        dart = ((0, 0), (4, 0), (1, 1), (4, 4), (0, 4), (1, 2))
        kernel = visibility_kernel(_polygon(dart))

        assert kernel.kernel_dimension == "EMPTY"
        result = measure_certificate(kernel, ())

        assert result.kernel_measures.vertices == ()
        assert result.kernel_measures.edges == ()
        assert result.kernel_measures.perimeter is None
        assert result.kernel_segment_length is None
        assert result.kernel.kernel_area.as_fraction() == 0
        assert verify_measure_certificate(result)

    def test_expected_difference_mismatch_reports_not_holds(self) -> None:
        # A wrong caller expectation is a valid False outcome, not an error:
        # one polygon must not promote a universal inequality.
        (outcome,) = measure_certificate(
            _nakano_kernel(),
            (
                MeasureComparison(
                    left="KERNEL_AREA",
                    right="POLYGON_AREA",
                    operator="LT",
                    expected_difference=CanonicalRational.from_fraction(Fraction(0)),
                ),
            ),
        ).comparisons

        assert outcome.difference.as_fraction() < 0
        assert not outcome.holds


class TestReplayAndForgery:
    def test_native_and_catalog_paths_agree(self) -> None:
        kernel = _nakano_kernel()
        tool = _tool("geometry.polygon.measure_certificate.compute")

        assert tool.run(
            MeasureCertificateRequest(
                kernel=kernel,
                comparisons=(
                    MeasureComparison(
                        left="KERNEL_AREA", right="POLYGON_AREA", operator="LT"
                    ),
                ),
            )
        ) == measure_certificate(
            kernel,
            (
                MeasureComparison(
                    left="KERNEL_AREA", right="POLYGON_AREA", operator="LT"
                ),
            ),
        )

    def test_round_trip_and_verify(self) -> None:
        result = measure_certificate(
            _nakano_kernel(),
            (
                MeasureComparison(
                    left="KERNEL_AREA", right="POLYGON_AREA", operator="LT"
                ),
            ),
        )
        restored = MeasureCertificateResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_measure_certificate(restored)

    def test_tampered_difference_fails_verify(self) -> None:
        # Tamper a squared length and its root together: the claim stays
        # wire-consistent but false, so only replay catches it.
        result = measure_certificate(_nakano_kernel(), ())
        forged = json.loads(result.model_dump_json())
        forged["polygon_measures"]["edges"][0]["squared_length"] = {
            "num": "4",
            "den": "1",
        }
        forged["polygon_measures"]["edges"][0]["exact_length"] = {
            "num": "2",
            "den": "1",
        }
        forged_claim = MeasureCertificateResult.model_validate_json(json.dumps(forged))
        assert not verify_measure_certificate(forged_claim)

    def test_tampered_perimeter_fails_verify(self) -> None:
        # A zero perimeter with rational edges passes wire shape (the
        # None-pattern agrees) but fails replay: values are checked by
        # recomputation, never by re-derivation in validators.
        result = measure_certificate(_nakano_kernel(), ())
        forged = json.loads(result.model_dump_json())
        forged["polygon_measures"]["perimeter"] = {"num": "0", "den": "1"}
        forged_claim = MeasureCertificateResult.model_validate_json(json.dumps(forged))
        assert not verify_measure_certificate(forged_claim)

    def test_tampered_area_fails_replay(self) -> None:
        result = measure_certificate(_nakano_kernel(), ())
        forged = json.loads(result.model_dump_json())
        forged["kernel"]["polygon_area"] = {"num": "1", "den": "1"}
        forged_claim = MeasureCertificateResult.model_validate_json(json.dumps(forged))
        assert not verify_measure_certificate(forged_claim)

    def test_clockwise_claim_is_rejected(self) -> None:
        kernel = _nakano_kernel()
        forged = json.loads(kernel.model_dump_json())
        forged["polygon"]["points"] = list(reversed(forged["polygon"]["points"]))
        clockwise = PolygonKernelResult.model_validate_json(json.dumps(forged))

        with pytest.raises(OperationDomainValidationError):
            measure_certificate(clockwise, ())

    def test_non_simple_claim_is_rejected(self) -> None:
        kernel = _nakano_kernel()
        forged = json.loads(kernel.model_dump_json())

        def wire_point(x: int, y: int) -> dict[str, object]:
            return {"x": {"num": str(x), "den": "1"}, "y": {"num": str(y), "den": "1"}}

        forged["polygon"]["points"] = [
            wire_point(0, 0),
            wire_point(2, 2),
            wire_point(2, 0),
            wire_point(0, 2),
        ]
        forged["half_planes"] = forged["half_planes"][:4]
        for position, row in enumerate(forged["half_planes"]):
            row["edge_index"] = position
        forged["vertex_turns"] = forged["vertex_turns"][:4]
        for position, row in enumerate(forged["vertex_turns"]):
            row["vertex_index"] = position
        forged["reflex_vertex_indices"] = []
        bowtie = PolygonKernelResult.model_validate_json(json.dumps(forged))

        with pytest.raises(OperationDomainValidationError) as exc_info:
            measure_certificate(bowtie, ())
        assert "simple" in str(exc_info.value).lower() or (
            exc_info.value.errors()[0]["type"]
            == "geometry.measure_certificate_kernel_mismatch"
        )


class TestModelBranches:
    def _nakano_result(self) -> MeasureCertificateResult:
        return measure_certificate(_nakano_kernel(), ())

    def test_segment_branch_shape(self) -> None:
        result = self._nakano_result()
        forged = json.loads(result.model_dump_json())
        forged["kernel"]["kernel_dimension"] = "SEGMENT"
        boundary = forged["kernel"]["kernel_boundary"][:2]
        forged["kernel"]["kernel_boundary"] = boundary
        forged_claim = MeasureCertificateResult.model_validate_json(
            json.dumps(
                {
                    **forged,
                    "kernel_measures": {
                        "vertices": [row["point"] for row in boundary],
                        "edges": [],
                        "perimeter": None,
                    },
                    "kernel_segment_length": {"num": "5", "den": "1"},
                }
            )
        )
        assert forged_claim.kernel_segment_length is not None

    def test_point_branch_shape(self) -> None:
        result = self._nakano_result()
        forged = json.loads(result.model_dump_json())
        forged["kernel"]["kernel_dimension"] = "POINT"
        boundary = forged["kernel"]["kernel_boundary"][:1]
        forged["kernel"]["kernel_boundary"] = boundary
        forged_claim = MeasureCertificateResult.model_validate_json(
            json.dumps(
                {
                    **forged,
                    "kernel_measures": {
                        "vertices": [boundary[0]["point"]],
                        "edges": [],
                        "perimeter": {"num": "0", "den": "1"},
                    },
                    "kernel_segment_length": None,
                }
            )
        )
        assert forged_claim.kernel_measures.perimeter is not None

    def test_empty_branch_shape(self) -> None:
        result = self._nakano_result()
        forged = json.loads(result.model_dump_json())
        forged["kernel"]["kernel_dimension"] = "EMPTY"
        forged["kernel"]["kernel_boundary"] = []
        forged_claim = MeasureCertificateResult.model_validate_json(
            json.dumps(
                {
                    **forged,
                    "kernel_measures": {
                        "vertices": [],
                        "edges": [],
                        "perimeter": None,
                    },
                    "kernel_segment_length": None,
                }
            )
        )
        assert forged_claim.kernel_measures.vertices == ()

    def test_too_many_comparisons_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasureCertificateRequest(
                kernel=_nakano_kernel(),
                comparisons=tuple(
                    MeasureComparison(
                        left="KERNEL_AREA", right="POLYGON_AREA", operator="LT"
                    )
                    for _ in range(17)
                ),
            )
