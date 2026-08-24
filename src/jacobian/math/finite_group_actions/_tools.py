"""Finite group-action operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_group_actions._models import (
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
from jacobian.math.finite_group_actions._operations import (
    compute_burnside_count,
    compute_cycle_index,
    compute_element_cycles,
    compute_polya_inventory,
    compute_subset_canonicalization,
)


def _op[
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
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# The cyclic group C_3 acting on three labelled points by rotation.
# generator: 0 -> 1 -> 2 -> 0, i.e. permutation[i] = (i+1) mod 3.
_ACTION = {
    "domain": ["a", "b", "c"],
    "generators": [[1, 2, 0]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "group_action.element_cycles.compute",
        "Compute the cycle decomposition of one group element",
        "Compute the complete cycle decomposition of one exact group element "
        "in a finite permutation action, including cycle partition, cycle "
        "lengths, cycle type as an integer partition of |X|, fixed-point set "
        "and count, and support set.",
        ElementCyclesRequest,
        ElementCyclesResult,
        compute_element_cycles,
        "algebra",
        "group",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_c3_identity_cycles",
                "Compute the cycle decomposition of the identity element of "
                "the cyclic group C_3 acting on three points.",
                {
                    "action": _ACTION,
                    "element": 0,
                },
            ),
        ),
    ),
    _op(
        "group_action.subset.canonicalize",
        "Canonicalize a subset under a finite permutation action",
        "Return the lexicographically least image of a subset under the "
        "generated group as an action-bound value whose increasing domain "
        "positions carry their permutation action, together with the "
        "lexicographically least transporter to that image and exact "
        "subset-orbit and setwise-stabilizer sizes.",
        SubsetCanonicalizationRequest,
        SubsetCanonicalizationResult,
        compute_subset_canonicalization,
        "algebra",
        "group",
        "permutation",
        "subset",
        "canonicalization",
        "transporter",
        "orbit-stabilizer",
        "exact",
        examples=(
            example(
                "cyclic_c3_singleton_canonicalization",
                "Canonicalize the singleton at position 2 under C_3; subset "
                "positions are bound to the declared action and must be "
                "distinct indices into its domain, and the generated group "
                f"must have order at most {MAX_GROUP_ORDER}.",
                {
                    "subset": {
                        "action": _ACTION,
                        "positions": [2],
                    },
                },
            ),
        ),
    ),
    _op(
        "group_action.cycle_index.compute",
        "Compute the cycle-index polynomial of a permutation action",
        "Compute the cycle-index polynomial Z(G) = (1/|G|) sum over g in G "
        "of product of x_i^{c_i(g)} as coefficient data, returned as the "
        "exact cycle-type multiplicity table.",
        CycleIndexRequest,
        CycleIndexResult,
        compute_cycle_index,
        "algebra",
        "group",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_c3_cycle_index",
                "Compute the cycle index of the cyclic group C_3 acting on "
                "three points.",
                {
                    "action": _ACTION,
                },
            ),
        ),
    ),
    _op(
        "group_action.burnside_count.compute",
        "Compute the number of orbits via Burnside's lemma",
        "Compute the number of orbits under the action using Burnside's "
        "lemma: |G\\X| = (1/|G|) sum_{g in G} |Fix(g)|, with the exact "
        "per-element fixed-point contribution table.",
        BurnsideCountRequest,
        BurnsideCountResult,
        compute_burnside_count,
        "algebra",
        "group",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_c3_burnside",
                "Compute the number of orbits of the cyclic group C_3 acting "
                "on three points via Burnside's lemma.",
                {
                    "action": _ACTION,
                },
            ),
        ),
    ),
    _op(
        "group_action.polya_inventory.compute",
        "Compute the Pólya enumeration inventory polynomial",
        "Compute the Pólya enumeration inventory polynomial for colouring X "
        "with k colours, returned as sparse coefficient data mapping each "
        "colour-multiplicity monomial to its orbit count.",
        PolyaInventoryRequest,
        PolyaInventoryResult,
        compute_polya_inventory,
        "algebra",
        "group",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_c3_polya_2_colors",
                "Compute the Pólya inventory for C_3 acting on three points "
                "with 2 colours.",
                {
                    "action": _ACTION,
                    "colors": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
