"""Public declaration for the finite-Abelian zero-sum atom hypergraph."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.zero_sum_atoms._models import (
    ZeroSumAtomHypergraphRequest,
    ZeroSumAtomHypergraphResult,
)
from jacobian.math.combinatorics.additive.zero_sum_atoms.operations import (
    construct_zero_sum_atom_hypergraph,
)


def _construct_zero_sum_atom_hypergraph(
    request: ZeroSumAtomHypergraphRequest,
) -> ZeroSumAtomHypergraphResult:
    return construct_zero_sum_atom_hypergraph(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.zero_sum.atom_hypergraph.construct",
        title="Construct the minimal zero-sum subset hypergraph",
        description=(
            "Given a distinct subset of an explicitly presented finite Abelian "
            "group, return the complete finite hypergraph whose hyperedges are "
            "the inclusion-minimal nonempty zero-sum subsets. Vertices are "
            "stable decimal source indices, and the retained source supplies "
            "their group elements and parent moduli. The complete nonempty-subset "
            "search admits at most 24 source elements and rejects a request "
            "before execution when its complete enumeration or exact atom "
            "family would exceed the published bounds."
        ),
        request_type=ZeroSumAtomHypergraphRequest,
        result_type=ZeroSumAtomHypergraphResult,
        run=_construct_zero_sum_atom_hypergraph,
        tags=(
            "additive-combinatorics",
            "finite-abelian-group",
            "hypergraph",
            "zero-sum",
            "inclusion-minimal",
            "exact",
        ),
        examples=(
            OperationExample(
                name="z7_atoms",
                description=(
                    "In Z/7Z, construct the atom hypergraph of the source "
                    "{1,2,3,4,5,6}; reduced source rows must be distinct and "
                    "sorted, and the complete zero-sum subset search must fit "
                    "the published bounds."
                ),
                input={
                    "source": {
                        "group": {"moduli": [7]},
                        "elements": [
                            [1],
                            [2],
                            [3],
                            [4],
                            [5],
                            [6],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
