"""Cyclic prefix-sum residue profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    MAX_SEQUENCING_PERMUTATION_NODES,
    MAX_SEQUENCING_SOURCE_ITEMS,
    CyclicPrefixSumResidueProfileRequest,
    CyclicPrefixSumResidueProfileResult,
    ForbiddenPrefixSequencingRequest,
    ForbiddenPrefixSequencingResult,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum.operations import (
    compute_cyclic_prefix_sum_residue_profile,
    search_forbidden_prefix_cyclic_ordering,
)


def compute_cyclic_prefix_sum_residue_profile_op(
    request: CyclicPrefixSumResidueProfileRequest,
) -> CyclicPrefixSumResidueProfileResult:
    return compute_cyclic_prefix_sum_residue_profile(request.sequence, request.modulus)


def search_forbidden_prefix_cyclic_ordering_op(
    request: ForbiddenPrefixSequencingRequest,
) -> ForbiddenPrefixSequencingResult:
    return search_forbidden_prefix_cyclic_ordering(
        request.source,
        request.first_element,
        request.forbidden_values,
        search_node_limit=request.search_node_limit,
    )


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
    MathTool(
        operation_id="additive.cyclic_prefix_sum.forbidden_prefix_sequencing.find",
        title="Find a cyclic ordering avoiding forbidden proper prefix sums",
        description=(
            "Given a distinct subset of an explicitly presented finite Abelian "
            "group, an optional prescribed first element, and a finite forbidden "
            "set, return the first deterministic ordering whose proper prefix "
            "sums are pairwise distinct, nonzero, and omit every forbidden value, "
            "or establish exact nonexistence after exhausting the admitted "
            "source-index permutation search. The result preserves the "
            "standard cyclic convention when a zero-sum source returns to zero. The source is "
            f"bounded to {MAX_SEQUENCING_SOURCE_ITEMS} elements, its ambient "
            "group axes and retained coordinates to the operation's coordinate envelope, and the complete search "
            f"to {MAX_SEQUENCING_PERMUTATION_NODES:,} states; a node budget stop "
            "would be UNKNOWN, not nonexistence."
        ),
        request_type=ForbiddenPrefixSequencingRequest,
        result_type=ForbiddenPrefixSequencingResult,
        run=search_forbidden_prefix_cyclic_ordering_op,
        tags=(
            "additive-combinatorics",
            "finite-abelian-group",
            "cyclic-sequencing",
            "partial-sums",
            "bounded-search",
            "exact",
        ),
        examples=(
            OperationExample(
                name="fixed_start_z7",
                description=(
                    "In Z/7Z, search the four-element source {1,2,5,6} with "
                    "first element 2 while avoiding proper prefix sum 1; the "
                    "input group must be a finite cyclic product and reduced "
                    "source elements must be distinct and sorted."
                ),
                input={
                    "source": {
                        "group": {"moduli": [7]},
                        "elements": [[1], [2], [5], [6]],
                    },
                    "first_element": [2],
                    "forbidden_values": [[1]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
