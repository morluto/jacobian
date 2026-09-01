"""Declarations for the Sidon extension-profile operation."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics._sidon_extension_kernel import (
    compute_sidon_extension_profile as _compute_sidon_extension_profile,
)
from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionProfileRequest,
    SidonExtensionProfileResult,
)


def compute_sidon_extension_profile(
    request: SidonExtensionProfileRequest,
) -> SidonExtensionProfileResult:
    return _compute_sidon_extension_profile(
        request.source_elements,
        request.candidate_elements,
    )


SIDON_EXTENSION_OPERATION = (
    MathTool(
        operation_id="combinatorics.integer_set.sidon.extension_profile.compute",
        title="Compute Sidon extension profile",
        description=(
            "Given a source integer set A and a candidate set C disjoint from "
            "A, partition C into candidates x for which A plus x is Sidon and "
            "candidates for which it is not, each with a replayable "
            "repeated-difference obstruction."
        ),
        request_type=SidonExtensionProfileRequest,
        result_type=SidonExtensionProfileResult,
        run=compute_sidon_extension_profile,
        tags=("combinatorics", "additive-combinatorics", "sidon"),
        examples=(
            OperationExample(
                name="sidon_extension_basic",
                description=(
                    "With source Sidon set {1, 2} and candidates {3, 4}, "
                    "candidate 3 fails because difference 1 repeats, while "
                    "candidate 4 succeeds. Source and candidate elements must "
                    "be unique and disjoint."
                ),
                input={
                    "source_elements": ["1", "2"],
                    "candidate_elements": ["3", "4"],
                },
            ),
        ),
    ),
)

__all__ = ["SIDON_EXTENSION_OPERATION"]
