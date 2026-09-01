"""Typed declarations for Lie algebra homology operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cohomology.lie_algebra._models import (
    ChevalleyEilenbergComplexRequest,
    ChevalleyEilenbergComplexResult,
    LieHomologyRequest,
    LieHomologyResult,
)
from jacobian.math.topology.cohomology.lie_algebra.operations import (
    chevalley_eilenberg_complex,
    lie_homology,
)


def _run_chevalley_eilenberg_complex(
    request: ChevalleyEilenbergComplexRequest,
) -> ChevalleyEilenbergComplexResult:
    return chevalley_eilenberg_complex(request.lie_algebra)


def _run_lie_homology(request: LieHomologyRequest) -> LieHomologyResult:
    return lie_homology(request.lie_algebra)


_CE_EXAMPLE: dict[str, Any] = {
    "lie_algebra": {
        "prime": 2,
        "dimension": 2,
        "structure_constants": [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
    },
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lie_algebra.chevalley_eilenberg.complex.compute",
        title="Compute the Chevalley-Eilenberg chain complex for a Lie algebra",
        description="Given a finite-dimensional Lie algebra over GF(p) with structure "
        "constants, compute the Chevalley-Eilenberg chain complex with "
        "trivial coefficients. The chain groups are C_p = Lambda^p(g) and "
        "the differential is built from the Lie bracket structure constants. "
        "Structure constants are validated as a genuine Lie bracket "
        "(alternating, antisymmetric, Jacobi) at the request boundary.",
        request_type=ChevalleyEilenbergComplexRequest,
        result_type=ChevalleyEilenbergComplexResult,
        run=_run_chevalley_eilenberg_complex,
        tags=("lie-algebra", "chevalley-eilenberg", "homology", "exact"),
        examples=(
            OperationExample(
                name="abelian_2d",
                description="Compute the CE chain complex of a 2D abelian Lie algebra over "
                "GF(2); structure_constants must be the dimension x dimension x "
                "dimension tensor of canonical GF(prime) residues defining an "
                "alternating bracket satisfying antisymmetry and Jacobi.",
                input=_CE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.homology.compute",
        title="Compute Lie algebra homology with trivial coefficients",
        description="Given a finite-dimensional Lie algebra over GF(p) with structure "
        "constants, compute the exact homology groups H_p(g, K) using the "
        "Chevalley-Eilenberg chain complex and Gaussian elimination over "
        "GF(p). Structure constants are validated as a genuine Lie bracket "
        "(alternating, antisymmetric, Jacobi) at the request boundary.",
        request_type=LieHomologyRequest,
        result_type=LieHomologyResult,
        run=_run_lie_homology,
        tags=("lie-algebra", "homology", "exact"),
        examples=(
            OperationExample(
                name="abelian_2d_homology",
                description="Compute homology of a 2D abelian Lie algebra over GF(2); "
                "structure_constants must be the dimension x dimension x "
                "dimension tensor of canonical GF(prime) residues defining an "
                "alternating bracket satisfying antisymmetry and Jacobi.",
                input=_CE_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
