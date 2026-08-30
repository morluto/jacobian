"""Exact planar framework operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.geometry.framework._models import (
    PlanarRigidityProfile,
    PlanarRigidityProfileRequest,
)
from jacobian.math.geometry.framework.operations import planar_rigidity_profile


def _run_planar_rigidity_profile(
    request: PlanarRigidityProfileRequest,
) -> PlanarRigidityProfile:
    return planar_rigidity_profile(
        request.configuration, request.graph, enforce_transport_limit=True
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="geometry.framework.planar_rigidity_profile.compute",
        title="Compute an exact planar framework rigidity profile",
        description=(
            "Given a labelled rational planar point configuration and a simple "
            "undirected graph on exactly the same labels, construct the exact "
            "coordinate-sparse rigidity matrix on the configuration's vertex "
            "axis and the lexicographically sorted graph-edge axis, then return "
            "its exact rational rank and pivot columns. Rank 2|V|-3 establishes "
            "infinitesimal rigidity of the supplied realization; a smaller rank "
            "does not decide local or global rigidity."
        ),
        request_type=PlanarRigidityProfileRequest,
        result_type=PlanarRigidityProfile,
        run=_run_planar_rigidity_profile,
        tags=("geometry", "framework", "rigidity-matrix", "exact-rational"),
        examples=(
            example(
                "rational_triangle",
                "Compute the exact rigidity matrix and rank of a non-collinear "
                "rational triangle; the graph labels must exactly match the "
                "planar configuration labels.",
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
                        ]
                    },
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["b", "c"], ["a", "c"], ["a", "b"]],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
