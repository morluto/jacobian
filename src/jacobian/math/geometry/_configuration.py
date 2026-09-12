"""Configuration-level geometry operations."""

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    MAX_SPANNED_CIRCLE_WORK,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    GeneralPositionRequest,
    GeneralPositionResult,
    SpannedCircleProfileRequest,
    SpannedCircleProfileResult,
)
from jacobian.math.geometry._tools import (
    circumradius_profile,
    general_position_search,
    spanned_circle_profile,
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
        operation_id="geometry.points.spanned_circle_profile.compute",
        title="Compute the exact spanned-circle incidence profile",
        description=(
            "Given a bounded labelled PointConfiguration (3..32 planar points, "
            "each coordinate at most 256 digits after translating by a "
            "minimum-height source origin, collinearity work C(n,3)*max_digits^2 plus circumcircle "
            "construction per non-collinear triple plus incidence work "
            "n*(distinct circles)*max_digits^2 at most 2000000, restored circle "
            f"components at most {MAX_CANONICAL_INTEGER_DIGITS} digits and "
            f"aggregate at most {MAX_SPANNED_CIRCLE_WORK} digits, at most C(n,3) "
            "circle rows and globally 4960), enumerate every distinct circle "
            "determined by a non-collinear source triple and return its canonical "
            "circle value with the complete source-point incidence set. Collinear "
            "triples are omitted; exhaustive triple generation and "
            "point-membership work are bounded before incidence expansion."
        ),
        request_type=SpannedCircleProfileRequest,
        result_type=SpannedCircleProfileResult,
        run=spanned_circle_profile,
        tags=("geometry", "circle", "incidence", "configuration"),
        examples=(
            OperationExample(
                name="unit_square_one_circle",
                description=(
                    "Compute the one circle through all four unit-square vertices; "
                    "the source points must be distinct rational planar points."
                ),
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
                        ]
                    }
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
