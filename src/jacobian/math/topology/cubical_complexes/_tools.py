"""Cubical complex operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalChainComplexRequest,
    CubicalChainComplexResult,
    CubicalComplexRequest,
    FaceClosureRequest,
    FaceClosureResult,
    FVectorResult,
)
from jacobian.math.topology.cubical_complexes.operations import (
    chain_complex,
    f_vector,
    face_closure,
)


def _f_vector(request: CubicalComplexRequest) -> FVectorResult:
    return f_vector(request.cells)


def _face_closure(request: FaceClosureRequest) -> FaceClosureResult:
    return face_closure(request.cells)


def _chain_complex(request: CubicalChainComplexRequest) -> CubicalChainComplexResult:
    return chain_complex(request.cells, request.coefficient_ring, request.prime)


# A single 2D square: [(0,1),(0,1)] + [(0,1),(1,2)] + [(1,2),(0,1)] + [(1,2),(1,2)]
_CELLS = {
    "cells": [
        {"intervals": [[0, 1], [0, 1]]},
        {"intervals": [[0, 1], [1, 2]]},
        {"intervals": [[1, 2], [0, 1]]},
        {"intervals": [[1, 2], [1, 2]]},
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="cubical.f_vector.compute",
        title="Compute the f-vector of a cubical complex",
        description="Compute the f-vector (cell counts by dimension) and Euler "
        "characteristic of a finite cubical complex composed of "
        "elementary unit lattice cubes.",
        request_type=CubicalComplexRequest,
        result_type=FVectorResult,
        run=_f_vector,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="four_squares",
                description="Compute the f-vector of four unit squares forming a 2x2 grid; "
                "each interval must be unit length (b = a + 1).",
                input=_CELLS,
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.face_closure.compute",
        title="Compute the face closure of a cubical complex",
        description="Compute the full face closure (all proper faces) of a set "
        "of elementary cubes, returning total cell count and "
        "cells by dimension.",
        request_type=FaceClosureRequest,
        result_type=FaceClosureResult,
        run=_face_closure,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="single_square_closure",
                description="Compute the face closure of a single unit square; "
                "each interval must be unit length (b = a + 1).",
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.cubical_complex.chain_complex.compute",
        title="Compute the oriented cubical chain complex",
        description="Close a finite family of elementary integer-lattice cubes "
        "under every cubical face, then construct the exact based chain complex "
        "with oriented cubical boundary "
        "dQ = sum_j (-1)^(j-1) (Q_j^+ - Q_j^-) over ZZ or GF(p). Returns the "
        "canonical per-dimension cell bases, the chain complex value, and a "
        "replayed d^2 = 0 ledger; each chain group is bounded and the shared "
        "chain-complex kernel verifies the square-zero identity.",
        request_type=CubicalChainComplexRequest,
        result_type=CubicalChainComplexResult,
        run=_chain_complex,
        tags=(
            "topology",
            "cubical-complex",
            "chain-complex",
            "boundary-matrix",
            "exact",
        ),
        discovery_terms=(
            "cubical chain complex",
            "cubical boundary",
        ),
        examples=(
            OperationExample(
                name="unit_square_integer_chain_complex",
                description="Build the integer cubical chain complex of one unit "
                "square (4 vertices, 4 edges, 1 square); faces are closed "
                "automatically and intervals must be unit length (b = a + 1).",
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
            OperationExample(
                name="unit_square_mod_two_chain_complex",
                description="Build the GF(2) cubical chain complex of the same "
                "unit square with boundary coefficients reduced modulo two.",
                input={
                    "cells": [{"intervals": [[0, 1], [0, 1]]}],
                    "coefficient_ring": "GF_p",
                    "prime": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
