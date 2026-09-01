"""Cyclic prefix-sum residue profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    CyclicPrefixSumResidueProfileRequest,
    CyclicPrefixSumResidueProfileResult,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum.operations import (
    compute_cyclic_prefix_sum_residue_profile,
)


def compute_cyclic_prefix_sum_residue_profile_op(
    request: CyclicPrefixSumResidueProfileRequest,
) -> CyclicPrefixSumResidueProfileResult:
    return compute_cyclic_prefix_sum_residue_profile(request.sequence, request.modulus)


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.cyclic_prefix_sum.residue_profile.compute",
        title="Compute the cyclic prefix-sum residue profile of a sequence",
        description=(
            "Given a bounded ordered integer sequence and a positive modulus, "
            "return the complete partition of its nonempty prefix positions by "
            "their prefix sum residue modulo that modulus."
        ),
        request_type=CyclicPrefixSumResidueProfileRequest,
        result_type=CyclicPrefixSumResidueProfileResult,
        run=compute_cyclic_prefix_sum_residue_profile_op,
        tags=("additive-combinatorics", "exact"),
        examples=(
            OperationExample(
                name="z5_sequence_113",
                description="In Z/5Z, the sequence (1,1,3) has prefix residues 1,2,0.",
                input={
                    "sequence": {"items": ["1", "1", "3"]},
                    "modulus": "5",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
