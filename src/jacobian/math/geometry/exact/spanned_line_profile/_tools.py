"""Typed declarations for the spanned-line profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.exact.spanned_line_profile._models import (
    SpannedLineProfileRequest,
    SpannedLineProfileResult,
)
from jacobian.math.geometry.exact.spanned_line_profile.operations import (
    compute_spanned_line_profile,
)


def _compute(request: SpannedLineProfileRequest) -> SpannedLineProfileResult:
    return compute_spanned_line_profile(request.configuration)


TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.points.spanned_line_profile.compute",
        title="Profile pair-spanned affine lines of a rational point configuration",
        description=(
            "For one bounded-dimensional PointConfiguration of pairwise distinct "
            "labelled rational points, return every distinct affine line spanned "
            "by an unordered source pair."
        ),
        request_type=SpannedLineProfileRequest,
        result_type=SpannedLineProfileResult,
        run=_compute,
        tags=("geometry", "line", "affine", "exact"),
        examples=(
            example(
                "three_collinear",
                "Three collinear points span one line.",
                {
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
                                    {"num": "2", "den": "1"},
                                    {"num": "0", "den": "1"},
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
