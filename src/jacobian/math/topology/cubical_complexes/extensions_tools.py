# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cubical_complexes.extensions import *


def _boundary(r: Any) -> Any:
    return boundary(r.cells)


def _relative(r: Any) -> Any:
    return relative_homology(r)


def _triangulate(r: Any) -> Any:
    return triangulate(r)


_SQUARE = {"cells": [{"intervals": [[0, 1], [0, 1]]}]}
TOOLS = (
    MathTool(
        operation_id="cubical.boundary.compute",
        title="Compute the signed cubical boundary",
        description="Compute the codimension-one oriented boundary terms of canonical maximal unit cubes, retaining signed reconstruction and cancellation.",
        request_type=CubicalBoundaryRequest,
        result_type=CubicalBoundaryResult,
        run=_boundary,
        tags=("topology", "cubical", "boundary", "exact"),
        examples=(
            OperationExample(
                name="square_boundary",
                description="Compute the signed boundary of one unit square; intervals are lattice intervals of length zero or one.",
                input=_SQUARE,
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.relative_homology.compute",
        title="Compute finite cubical relative homology",
        description="Compute exact finite-field Betti numbers of a face-closed cubical pair (X,A), retaining both ambient axes and rejecting a non-subcomplex A.",
        request_type=RelativeCubicalHomologyRequest,
        result_type=RelativeCubicalHomologyResult,
        run=_relative,
        tags=("topology", "cubical", "relative-homology", "exact"),
        examples=(
            OperationExample(
                name="square_mod_boundary",
                description="Compute relative homology of a square modulo its full boundary; the supplied subcomplex must be contained in the ambient face closure.",
                input={
                    "cells": [{"intervals": [[0, 1], [0, 1]]}],
                    "subcomplex_cells": [
                        {"intervals": [[0, 0], [0, 1]]},
                        {"intervals": [[1, 1], [0, 1]]},
                        {"intervals": [[0, 1], [0, 0]]},
                        {"intervals": [[0, 1], [1, 1]]},
                    ],
                    "prime": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.triangulation.compute",
        title="Triangulate finite cubical cells",
        description="Return the deterministic staircase triangulation of each source cube and its ambient lattice-point axis.",
        request_type=CubicalTriangulationRequest,
        result_type=CubicalTriangulationResult,
        run=_triangulate,
        tags=("topology", "cubical", "triangulation", "exact"),
        examples=(
            OperationExample(
                name="square_staircase",
                description="Triangulate one unit square by its two staircase triangles; source intervals must be unit lattice intervals.",
                input=_SQUARE,
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
