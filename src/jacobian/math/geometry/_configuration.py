"""Configuration-level geometry operations."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    GeneralPositionRequest,
    GeneralPositionResult,
)
from jacobian.math.geometry._operations import (
    circumradius_profile,
    general_position_search,
)
from jacobian.math.geometry._support import geometry_operation

CONFIGURATION_OPERATIONS = (
    geometry_operation(
        "geometry.points.general_position.search",
        "Search for collinear triples and concyclic quadruples",
        "Given a bounded rational planar point configuration (3..32 points, each "
        "coordinate at most 256 digits, C(n,4)*max_digits^2<=1000000 to bound the "
        "exhaustive determinant work) exhaustively find all collinear triples and "
        "all concyclic quadruples, or establish that none exist. Returns "
        "source-labelled witnesses with sorted indices.",
        GeneralPositionRequest,
        GeneralPositionResult,
        general_position_search,
        "geometry",
        "incidence",
        "configuration",
        examples=(
            example(
                "square_general_position",
                "Search a unit square for collinear triples and concyclic "
                "quadruples; the four vertices of a square are concyclic (points "
                "are bounded: 3..32 points, each coordinate <=256 digits, "
                "C(n,4)*max_digits^2<=1000000).",
                {
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
    geometry_operation(
        "geometry.points.circumradius_profile.compute",
        "Compute circumradius data for all triples",
        "Given a bounded rational planar point configuration (3..32 points, each "
        "coordinate at most 256 digits, worst-case profile size "
        "C(n,3)*(80*max_digits+80) characters within the 8,000,000-character "
        "output budget) return the complete circumradius squared for every unordered "
        "triple, with explicit degenerate (collinear) disposition. Each entry includes "
        "the source-labelled triple indices and the exact rational squared "
        "circumradius.",
        CircumradiusProfileRequest,
        CircumradiusProfileResult,
        circumradius_profile,
        "geometry",
        "circumradius",
        "configuration",
        examples=(
            example(
                "unit_triangle",
                "Compute circumradius profile for a triangle; the three "
                "vertices must be unique points and coordinates are bounded to "
                "256 digits with the worst-case profile size within budget.",
                {
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
