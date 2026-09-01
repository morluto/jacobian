"""Typed declarations for the triangle area profile operation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.exact.triangle_area_profile._models import (
    TriangleAreaProfileRequest,
    TriangleAreaProfileResult,
)
from jacobian.math.geometry.exact.triangle_area_profile.operations import (
    compute_triangle_area_profile,
)


def _compute(request: TriangleAreaProfileRequest) -> TriangleAreaProfileResult:
    return compute_triangle_area_profile(request.configuration)


TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.points.triangle_area_profile.compute",
        title="Compute complete triangle-area profiles of rational configurations",
        description=(
            "Given a bounded distinct rational planar point configuration, "
            "return every unordered source triple grouped by its exact unsigned "
            "triangle area, with collinear triples retained in the zero-area class."
        ),
        request_type=TriangleAreaProfileRequest,
        result_type=TriangleAreaProfileResult,
        run=_compute,
        tags=("geometry", "triangle", "area", "exact"),
        examples=(
            OperationExample(
                name="unit_square",
                description="Triangle areas of the unit square corners.",
                input={
                    "configuration": {
                        "points": [
                            {
                                "label": "a",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "label": "b",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "label": "c",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "label": "d",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
