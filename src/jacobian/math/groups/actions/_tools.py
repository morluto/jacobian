"""Finite group-action operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.actions._models import (
    MAX_GROUP_ORDER,
    BurnsideCountRequest,
    BurnsideCountResult,
    CycleIndexRequest,
    CycleIndexResult,
    ElementCyclesRequest,
    ElementCyclesResult,
    PolyaInventoryRequest,
    PolyaInventoryResult,
    SubsetCanonicalizationRequest,
    SubsetCanonicalizationResult,
)
from jacobian.math.groups.actions.operations import (
    burnside_count,
    cycle_index,
    element_cycles,
    polya_inventory,
    subset_canonicalization,
)


def _element_cycles(request: ElementCyclesRequest) -> ElementCyclesResult:
    return element_cycles(request.action, request.element)


def _subset(request: SubsetCanonicalizationRequest) -> SubsetCanonicalizationResult:
    return subset_canonicalization(request.subset)


def _cycle_index(request: CycleIndexRequest) -> CycleIndexResult:
    return cycle_index(request.action)


def _burnside(request: BurnsideCountRequest) -> BurnsideCountResult:
    return burnside_count(request.action)


def _polya(request: PolyaInventoryRequest) -> PolyaInventoryResult:
    return polya_inventory(request.action, request.colors)


# The cyclic group C_3 acting on three labelled points by rotation.
# generator: 0 -> 1 -> 2 -> 0, i.e. permutation[i] = (i+1) mod 3.
_ACTION = {
    "domain": ["a", "b", "c"],
    "generators": [[1, 2, 0]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="group_action.element_cycles.compute",
        title="Compute the cycle decomposition of one group element",
        description="Compute the complete cycle decomposition of one exact group element "
        "in a finite permutation action, including cycle partition, cycle "
        "lengths, cycle type as an integer partition of |X|, fixed-point set "
        "and count, and support set.",
        request_type=ElementCyclesRequest,
        result_type=ElementCyclesResult,
        run=_element_cycles,
        tags=("algebra", "group", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_c3_identity_cycles",
                description="Compute the cycle decomposition of the identity element of "
                "the cyclic group C_3 acting on three points.",
                input={
                    "action": _ACTION,
                    "element": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group_action.subset.canonicalize",
        title="Canonicalize a subset under a finite permutation action",
        description="Return the lexicographically least image of a subset under the "
        "generated group as an action-bound value whose increasing domain "
        "positions carry their permutation action, together with the "
        "lexicographically least transporter to that image and exact "
        "subset-orbit and setwise-stabilizer sizes.",
        request_type=SubsetCanonicalizationRequest,
        result_type=SubsetCanonicalizationResult,
        run=_subset,
        tags=(
            "algebra",
            "group",
            "permutation",
            "subset",
            "canonicalization",
            "transporter",
            "orbit-stabilizer",
            "exact",
        ),
        examples=(
            OperationExample(
                name="cyclic_c3_singleton_canonicalization",
                description="Canonicalize the singleton at position 2 under C_3; subset "
                "positions are bound to the declared action and must be "
                "distinct indices into its domain, and the generated group "
                f"must have order at most {MAX_GROUP_ORDER}.",
                input={
                    "subset": {
                        "action": _ACTION,
                        "positions": [2],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group_action.cycle_index.compute",
        title="Compute the cycle-index polynomial of a permutation action",
        description="Compute the cycle-index polynomial Z(G) = (1/|G|) sum over g in G "
        "of product of x_i^{c_i(g)} as coefficient data, returned as the "
        "exact cycle-type multiplicity table.",
        request_type=CycleIndexRequest,
        result_type=CycleIndexResult,
        run=_cycle_index,
        tags=("algebra", "group", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_c3_cycle_index",
                description="Compute the cycle index of the cyclic group C_3 acting on "
                "three points.",
                input={
                    "action": _ACTION,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group_action.burnside_count.compute",
        title="Compute the number of orbits via Burnside's lemma",
        description="Compute the number of orbits under the action using Burnside's "
        "lemma: |G\\X| = (1/|G|) sum_{g in G} |Fix(g)|, with the exact "
        "per-element fixed-point contribution table.",
        request_type=BurnsideCountRequest,
        result_type=BurnsideCountResult,
        run=_burnside,
        tags=("algebra", "group", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_c3_burnside",
                description="Compute the number of orbits of the cyclic group C_3 acting "
                "on three points via Burnside's lemma.",
                input={
                    "action": _ACTION,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group_action.polya_inventory.compute",
        title="Compute the Pólya enumeration inventory polynomial",
        description="Compute the Pólya enumeration inventory polynomial for colouring X "
        "with k colours, returned as sparse coefficient data mapping each "
        "colour-multiplicity monomial to its orbit count.",
        request_type=PolyaInventoryRequest,
        result_type=PolyaInventoryResult,
        run=_polya,
        tags=("algebra", "group", "permutation", "exact"),
        examples=(
            OperationExample(
                name="cyclic_c3_polya_2_colors",
                description="Compute the Pólya inventory for C_3 acting on three points "
                "with 2 colours.",
                input={
                    "action": _ACTION,
                    "colors": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
