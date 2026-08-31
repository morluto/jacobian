"""Typed declarations for the cyclic sumset representation profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.additive.cyclic_sumset_profile._models import (
    CyclicSumsetRequest,
    CyclicSumsetResult,
)
from jacobian.math.combinatorics.additive.cyclic_sumset_profile.operations import (
    compute_cyclic_sumset_profile,
)


def _compute(request: CyclicSumsetRequest) -> CyclicSumsetResult:
    return compute_cyclic_sumset_profile(request.modulus, request.left, request.right)


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.cyclic_sumset.representation_profile.compute",
        title="Compute exact cyclic sumset representation profiles",
        description=(
            "Given two finite subsets A, B of Z/mZ, return the complete exact "
            "cyclic representation function r_{A+B}(c) = |{(a,b) : a+b=c mod m}| "
            "on its occupied support; omitted residues have representation count zero."
        ),
        request_type=CyclicSumsetRequest,
        result_type=CyclicSumsetResult,
        run=_compute,
        tags=("additive", "combinatorics", "cyclic", "exact"),
        examples=(
            example(
                "z5",
                "Cyclic sumset of {0,1} + {0,2} in Z/5Z.",
                {"modulus": 5, "left": [0, 1], "right": [0, 2]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
