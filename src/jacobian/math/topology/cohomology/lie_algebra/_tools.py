"""Typed declarations for Lie algebra homology operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def lie_homology_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    lie_homology_operation(
        "lie_algebra.chevalley_eilenberg.complex.compute",
        "Compute the Chevalley-Eilenberg chain complex for a Lie algebra",
        "Given a finite-dimensional Lie algebra over GF(p) with structure "
        "constants, compute the Chevalley-Eilenberg chain complex with "
        "trivial coefficients. The chain groups are C_p = Lambda^p(g) and "
        "the differential is built from the Lie bracket structure constants. "
        "Structure constants are validated as a genuine Lie bracket "
        "(alternating, antisymmetric, Jacobi) at the request boundary.",
        ChevalleyEilenbergComplexRequest,
        ChevalleyEilenbergComplexResult,
        _run_chevalley_eilenberg_complex,
        "lie-algebra",
        "chevalley-eilenberg",
        "homology",
        "exact",
        examples=(
            example(
                "abelian_2d",
                "Compute the CE chain complex of a 2D abelian Lie algebra over "
                "GF(2); structure_constants must be the dimension x dimension x "
                "dimension tensor of canonical GF(prime) residues defining an "
                "alternating bracket satisfying antisymmetry and Jacobi.",
                _CE_EXAMPLE,
            ),
        ),
    ),
    lie_homology_operation(
        "lie_algebra.homology.compute",
        "Compute Lie algebra homology with trivial coefficients",
        "Given a finite-dimensional Lie algebra over GF(p) with structure "
        "constants, compute the exact homology groups H_p(g, K) using the "
        "Chevalley-Eilenberg chain complex and Gaussian elimination over "
        "GF(p). Structure constants are validated as a genuine Lie bracket "
        "(alternating, antisymmetric, Jacobi) at the request boundary.",
        LieHomologyRequest,
        LieHomologyResult,
        _run_lie_homology,
        "lie-algebra",
        "homology",
        "exact",
        examples=(
            example(
                "abelian_2d_homology",
                "Compute homology of a 2D abelian Lie algebra over GF(2); "
                "structure_constants must be the dimension x dimension x "
                "dimension tensor of canonical GF(prime) residues defining an "
                "alternating bracket satisfying antisymmetry and Jacobi.",
                _CE_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
