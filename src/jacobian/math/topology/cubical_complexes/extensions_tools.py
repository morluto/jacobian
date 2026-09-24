# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cubical_complexes._models import (
    MAX_CELLS,
    MAX_CUBICAL_BITMAP_RESULT_BYTES,
    MAX_CUBICAL_BITMAP_SIDE,
)
from jacobian.math.topology.cubical_complexes.extensions import *


def _bitmap_to_complex(r: Any) -> Any:
    return bitmap_to_complex(r)


def _boundary(r: Any) -> Any:
    return boundary(r.cells)


def _relative(r: Any) -> Any:
    return relative_homology(r)


def _triangulate(r: Any) -> Any:
    return triangulate(r)


_SQUARE = {"cells": [{"intervals": [[0, 1], [0, 1]]}]}
TOOLS = (
    MathTool(
        operation_id="topology.cubical_complex.from_binary_bitmap_2d.compute",
        title="Construct a cubical complex from a 2D binary bitmap",
        description=(
            "Treat each true bitmap pixel as its closed unit square ([c,c+1] "
            "in the column axis, [r,r+1] in the downward-increasing row axis), "
            "then return the complete cubical face closure. Rows and columns "
            f"are each bounded to {MAX_CUBICAL_BITMAP_SIDE}, foreground pixels "
            f"to {MAX_CELLS}, and the conservative result encoding to "
            f"{MAX_CUBICAL_BITMAP_RESULT_BYTES} bytes. An all-background bitmap "
            "returns the empty complex on the ordered (column, row) axes."
        ),
        request_type=CubicalBitmapRequest,
        result_type=CubicalBitmapResult,
        run=_bitmap_to_complex,
        tags=("topology", "cubical", "bitmap", "exact"),
        examples=(
            OperationExample(
                name="one_foreground_pixel",
                description=(
                    "Convert a 1-by-1 bitmap whose sole true pixel is the closed "
                    "unit square on the (column, row) lattice axes."
                ),
                input={"pixels": [[True]]},
            ),
        ),
    ),
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
