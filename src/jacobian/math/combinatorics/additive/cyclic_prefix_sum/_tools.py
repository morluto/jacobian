"""Cyclic prefix-sum residue profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def cps_operation[
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


TOOLS: MathTools = (
    cps_operation(
        "additive.cyclic_prefix_sum.residue_profile.compute",
        "Compute the cyclic prefix-sum residue profile of a sequence",
        (
            "Given a bounded ordered integer sequence and a positive modulus, "
            "return the complete partition of its nonempty prefix positions by "
            "their prefix sum residue modulo that modulus."
        ),
        CyclicPrefixSumResidueProfileRequest,
        CyclicPrefixSumResidueProfileResult,
        compute_cyclic_prefix_sum_residue_profile_op,
        "additive-combinatorics",
        "exact",
        examples=(
            example(
                "z5_sequence_113",
                "In Z/5Z, the sequence (1,1,3) has prefix residues 1,2,0.",
                {
                    "sequence": {"items": ["1", "1", "3"]},
                    "modulus": "5",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
