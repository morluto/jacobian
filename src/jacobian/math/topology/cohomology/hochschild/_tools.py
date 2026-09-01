"""Typed declarations for Hochschild complex operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cohomology.hochschild._models import (
    HochschildChainComplexRequest,
    HochschildChainComplexResult,
    HochschildHomologyRequest,
    HochschildHomologyResult,
)
from jacobian.math.topology.cohomology.hochschild.operations import (
    hochschild_chain_complex,
    hochschild_homology,
)


def _chain_complex(
    request: HochschildChainComplexRequest,
) -> HochschildChainComplexResult:
    return hochschild_chain_complex(request.algebra, request.max_degree)


def _homology(request: HochschildHomologyRequest) -> HochschildHomologyResult:
    return hochschild_homology(request.algebra, request.max_degree)


_COMPLEX_EXAMPLE: dict[str, Any] = {
    "algebra": {
        "prime": 5,
        "dimension": 1,
        "structure_constants": [[[1]]],
        "augmentation": [1],
    },
    "max_degree": 2,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="hochschild.chain_complex.compute",
        title="Compute the Hochschild chain complex with trivial bimodule",
        description="Given a finite-dimensional algebra over GF(p) with structure "
        "constants and an augmentation epsilon (an algebra homomorphism to "
        "GF(p)), compute the Hochschild chain complex C_n = A^⊗n with "
        "coefficients in the trivial module K on which A acts through "
        "epsilon. The differential is the full Hochschild boundary: adjacent "
        "multiplications plus both augmentation-dependent endpoint faces.",
        request_type=HochschildChainComplexRequest,
        result_type=HochschildChainComplexResult,
        run=_chain_complex,
        tags=("hochschild", "chain-complex", "exact"),
        examples=(
            OperationExample(
                name="one_dim_algebra",
                description="Compute the Hochschild complex of a 1D algebra over GF(5).",
                input=_COMPLEX_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="hochschild.homology.compute",
        title="Compute Hochschild homology with trivial bimodule",
        description="Given a finite-dimensional algebra over GF(p) with structure "
        "constants and an augmentation epsilon, compute the exact Hochschild "
        "homology groups HH_n(A, K) with coefficients in the trivial module "
        "defined by epsilon, using Gaussian elimination over GF(p).",
        request_type=HochschildHomologyRequest,
        result_type=HochschildHomologyResult,
        run=_homology,
        tags=("hochschild", "homology", "exact"),
        examples=(
            OperationExample(
                name="one_dim_homology",
                description="Compute Hochschild homology of a 1D algebra over GF(5).",
                input=_COMPLEX_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
