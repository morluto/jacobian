"""Configuration-level geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    GeneralPositionRequest,
    GeneralPositionResult,
)
from jacobian.math.geometry._tools import (
    circumradius_profile,
    general_position_search,
)

CONFIGURATION_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.points.general_position.search",
        title="Search for collinear triples and concyclic quadruples",
        description="Given a bounded rational planar point configuration (3..32 points, each "
        "coordinate at most 256 digits, C(n,4)*max_digits^2<=1000000 to bound the "
        "exhaustive determinant work) exhaustively find all collinear triples and "
        "all concyclic quadruples, or establish that none exist. Returns "
        "source-labelled witnesses with sorted indices.",
        request_type=GeneralPositionRequest,
        result_type=GeneralPositionResult,
        run=general_position_search,
        tags=("geometry", "incidence", "configuration"),
        examples=(
            OperationExample(
                name="square_general_position",
                description="Search a unit square for collinear triples and concyclic "
                "quadruples; the four vertices of a square are concyclic (points "
                "are bounded: 3..32 points, each coordinate <=256 digits, "
                "C(n,4)*max_digits^2<=1000000).",
                input={
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.circumradius_profile.compute",
        title="Compute circumradius data for all triples",
        description="Given a bounded rational planar point configuration (3..32 points, each "
        "coordinate at most 256 digits, worst-case profile size "
        "C(n,3)*(80*max_digits+80) characters within the 8,000,000-character "
        "output budget) return the complete circumradius squared for every unordered "
        "triple, with explicit degenerate (collinear) disposition. Each entry includes "
        "the source-labelled triple indices and the exact rational squared "
        "circumradius.",
        request_type=CircumradiusProfileRequest,
        result_type=CircumradiusProfileResult,
        run=circumradius_profile,
        tags=("geometry", "circumradius", "configuration"),
        examples=(
            OperationExample(
                name="unit_triangle",
                description="Compute circumradius profile for a triangle; the three "
                "vertices must be unique points and coordinates are bounded to "
                "256 digits with the worst-case profile size within budget.",
                input={
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
)
