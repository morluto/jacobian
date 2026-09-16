"""Public declaration for exact convex-polytope local-motion kernels."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.convex._models import (
    DirectionLocalMotionRequest,
    DirectionLocalMotionResult,
)
from jacobian.math.geometry.convex.operations import direction_local_motion


def _run_direction_local_motion(
    request: DirectionLocalMotionRequest,
) -> DirectionLocalMotionResult:
    return direction_local_motion(request.polytope, request.point, request.direction)


_UNIT_SQUARE_INEQUALITIES = [
    {
        "inequality_id": "bottom",
        "normal": [{"num": "0", "den": "1"}, {"num": "-1", "den": "1"}],
        "bound": {"num": "0", "den": "1"},
    },
    {
        "inequality_id": "left",
        "normal": [{"num": "-1", "den": "1"}, {"num": "0", "den": "1"}],
        "bound": {"num": "0", "den": "1"},
    },
    {
        "inequality_id": "right",
        "normal": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
        "bound": {"num": "1", "den": "1"},
    },
    {
        "inequality_id": "top",
        "normal": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
        "bound": {"num": "1", "den": "1"},
    },
]

TOOLS = (
    MathTool(
        operation_id="convex_geometry.polytope.direction_local_motion.compute",
        title="Decide exact strict local motion at a polytope boundary point",
        description=(
            "For a bounded labelled H-polytope, a boundary point, and a nonzero "
            "rational direction on one identical coordinate space, return the "
            "exact slack ledger and, for every active inequality, the exact "
            "pairing <a_i, u> with kind STRICTLY_INWARD, TANGENT, or OUTWARD. "
            "The aggregate is ILLUMINATES_STRICTLY iff every active inequality "
            "is strictly inward, MOVES_OUTSIDE iff any is outward, else "
            "TANGENT_OR_PARTIAL. Outside points, interior points, and zero "
            "directions are rejected; positive direction rescaling preserves "
            "the result."
        ),
        request_type=DirectionLocalMotionRequest,
        result_type=DirectionLocalMotionResult,
        run=_run_direction_local_motion,
        tags=("convex-geometry", "polytope", "illumination", "exact"),
        discovery_terms=(
            "strict illumination direction",
            "polytope boundary local motion",
            "active facet normal pairing",
            "tangent cone direction test",
        ),
        examples=(
            OperationExample(
                name="square_corner_strict_entry",
                description=(
                    "Decide strict entry of the bottom-left square corner along "
                    "(1,1); the point must lie on the polytope boundary and "
                    "the direction must be nonzero."
                ),
                input={
                    "polytope": {
                        "space": {"axes": ["x", "y"]},
                        "inequalities": _UNIT_SQUARE_INEQUALITIES,
                    },
                    "point": {
                        "space": {"axes": ["x", "y"]},
                        "coordinates": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "direction": {
                        "space": {"axes": ["x", "y"]},
                        "components": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
