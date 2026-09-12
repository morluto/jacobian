"""Public declaration for the exact root--critical distance profile."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.root_critical._models import (
    RootCriticalDistanceProfile,
    RootCriticalDistanceProfileRequest,
)
from jacobian.math.polynomials.root_critical.operations import (
    root_critical_distance_profile,
)

_CUBIC_EXAMPLE = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [3]},
                {"coefficient": {"num": "-1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
}


def compute_root_critical_distance_profile(
    request: RootCriticalDistanceProfileRequest,
) -> RootCriticalDistanceProfile:
    return root_critical_distance_profile(
        request.polynomial,
        max_pair_rows=request.max_pair_rows,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.root_critical_distance_profile.compute",
        title="Compute an exact root-critical distance profile",
        description=(
            "Return every distinct root of a bounded nonconstant univariate QQ "
            "polynomial, every distinct root of its derivative, their source "
            "multiplicities, and the complete Cartesian profile of exact squared "
            "complex distances. Root rectangles and real isolating intervals are "
            "certified rational enclosures; backend failures are operational "
            "errors and never imply an empty family."
        ),
        request_type=RootCriticalDistanceProfileRequest,
        result_type=RootCriticalDistanceProfile,
        run=compute_root_critical_distance_profile,
        tags=("polynomial", "roots", "critical-points", "distance", "exact"),
        examples=(
            OperationExample(
                name="cubic_roots_and_critical_point",
                description=(
                    "The three roots of z^3-1 are compared with the repeated "
                    "critical point 0; every squared distance is exactly 1."
                ),
                input=_CUBIC_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
