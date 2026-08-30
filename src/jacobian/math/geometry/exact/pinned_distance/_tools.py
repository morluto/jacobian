"""Pinned distance support profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.exact.pinned_distance._models import (
    PinnedDistanceSupportProfileRequest,
    PinnedDistanceSupportProfileResult,
)
from jacobian.math.geometry.exact.pinned_distance.operations import (
    compute_pinned_distance_support_profile,
)


def compute_pinned_distance_support_profile_op(
    request: PinnedDistanceSupportProfileRequest,
) -> PinnedDistanceSupportProfileResult:
    return compute_pinned_distance_support_profile(request.configuration)


def pdp_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    pdp_operation(
        "geometry.points.pinned_distance_support_profile.compute",
        "Compute the pinned distance support profile of a point configuration",
        (
            "For each source point in a finite labelled rational point "
            "configuration, return the complete sorted partition of all "
            "other source labels by exact squared Euclidean distance."
        ),
        PinnedDistanceSupportProfileRequest,
        PinnedDistanceSupportProfileResult,
        compute_pinned_distance_support_profile_op,
        "geometry",
        "exact",
        examples=(
            example(
                "unit_square",
                "Unit square with 4 points.",
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
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "label": "d",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
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
