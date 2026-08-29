"""Public declarations for exact rational box unions."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry.boxes._models import (
    BoxUnionVolumeRequest,
    BoxUnionVolumeResult,
)
from jacobian.math.geometry.boxes.operations import compute_box_union_volume


def _box_union_volume(request: BoxUnionVolumeRequest) -> BoxUnionVolumeResult:
    return compute_box_union_volume(request.boxes)


TOOLS: MathTools = (
    geometry_operation(
        "geometry.box_union.volume.compute",
        "Compute exact volume of a finite rational box union",
        "Compute exact Lebesgue volume for closed rational axis-aligned boxes. "
        "Return every nonempty indexed subset intersection and its volume, plus "
        "the complete source-bound inclusion-exclusion sum. Inputs contain one "
        "or more same-dimensional boxes in dimensions 1..64 whose echoed source "
        "and complete ledger fit the published byte budget, at most 16 nonempty "
        "boxes, and endpoint components within the canonical 32,768-digit "
        "rational limit; per-axis growth and result-byte budgets determine "
        "admission. intervals=null is the "
        "canonical empty box; equal endpoints are valid. Preflight bounds "
        "replay work, rational growth, and worst-case ledger bytes.",
        BoxUnionVolumeRequest,
        BoxUnionVolumeResult,
        _box_union_volume,
        "geometry",
        "measure",
        "box-union",
        "inclusion-exclusion",
        examples=(
            example(
                "three_overlapping_boxes",
                "Compute the exact volume 9/2 of three overlapping rational "
                "boxes and return all seven nonempty intersections; every box "
                "uses the same three standard coordinate axes and satisfies "
                "the complete-ledger budgets.",
                {
                    "boxes": [
                        {
                            "dimension": 3,
                            "intervals": [
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "2", "den": "1"},
                                },
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                },
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                },
                            ],
                        },
                        {
                            "dimension": 3,
                            "intervals": [
                                {
                                    "lower": {"num": "1", "den": "1"},
                                    "upper": {"num": "3", "den": "1"},
                                },
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                },
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                },
                            ],
                        },
                        {
                            "dimension": 3,
                            "intervals": [
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "3", "den": "1"},
                                },
                                {
                                    "lower": {"num": "0", "den": "1"},
                                    "upper": {"num": "1", "den": "1"},
                                },
                                {
                                    "lower": {"num": "1", "den": "2"},
                                    "upper": {"num": "3", "den": "2"},
                                },
                            ],
                        },
                    ]
                },
            ),
            example(
                "empty_and_degenerate_boxes",
                "Compute a zero-volume union containing the canonical empty "
                "box and a singleton interval; boxes share dimension one and "
                "the singleton remains a nonempty zero-volume ledger entry.",
                {
                    "boxes": [
                        {"dimension": 1, "intervals": None},
                        {
                            "dimension": 1,
                            "intervals": [
                                {
                                    "lower": {"num": "2", "den": "1"},
                                    "upper": {"num": "2", "den": "1"},
                                }
                            ],
                        },
                    ]
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
