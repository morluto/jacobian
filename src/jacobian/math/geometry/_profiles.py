"""Circumradius and forbidden-pattern operation declarations."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
)
from jacobian.math.geometry._operations import (
    circumradius_profile,
    forbidden_patterns,
)
from jacobian.math.geometry._support import geometry_operation

PROFILE_OPERATIONS = (
    geometry_operation(
        "geometry.circumradius.profile.compute",
        "Compute complete circumradius data for point configurations",
        "For one bounded labelled rational planar point configuration, return "
        "the complete exact circumradius data of all unordered triples: each "
        "nondegenerate triangle has an exact squared circumradius and each "
        "collinear triple is flagged as degenerate.",
        CircumradiusProfileRequest,
        CircumradiusProfileResult,
        circumradius_profile,
        "geometry",
        "circumradius",
        examples=(
            example(
                "unit_square_triples",
                "Compute circumradius data for four points of a unit square.",
                {
                    "points": [
                        {"label": "A", "point": {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}}},
                        {"label": "B", "point": {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}}},
                        {"label": "C", "point": {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}}},
                        {"label": "D", "point": {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}}},
                    ],
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.configuration.forbidden_patterns.check",
        "Find collinear triples and concyclic quadruples",
        "Given a bounded labelled rational planar point configuration, find a "
        "witness to either a collinear triple or a concyclic quadruple, or "
        "establish after complete bounded enumeration that neither exists.",
        ForbiddenPatternsRequest,
        ForbiddenPatternsResult,
        forbidden_patterns,
        "geometry",
        "incidence",
        examples=(
            example(
                "no_forbidden_patterns",
                "A unit square has no collinear triples or concyclic quadruples "
                "when checked as four points.",
                {
                    "configuration": {
                        "points": [
                            {"label": "A", "point": {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}}},
                            {"label": "B", "point": {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}}},
                            {"label": "C", "point": {"x": {"num": "2", "den": "1"}, "y": {"num": "0", "den": "1"}}},
                            {"label": "D", "point": {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}}},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["PROFILE_OPERATIONS"]
