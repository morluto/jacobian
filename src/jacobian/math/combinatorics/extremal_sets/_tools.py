"""Binary-union relation hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationRequest,
    BinaryUnionRelationResult,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_binary_union_relation,
)


def compute_binary_union_relation(
    request: BinaryUnionRelationRequest,
) -> BinaryUnionRelationResult:
    return construct_binary_union_relation(request.family)


def es_operation[
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
    es_operation(
        "set_system.binary_union_relation_hypergraph.compute",
        "Compute the binary-union relation hypergraph of a set family",
        (
            "Given a finite indexed family of distinct finite sets, return the "
            "complete 3-uniform relation hypergraph whose edges record triples "
            "(i,j,k) where S_i union S_j = S_k, with i < j and k distinct from both."
        ),
        BinaryUnionRelationRequest,
        BinaryUnionRelationResult,
        compute_binary_union_relation,
        "combinatorics",
        "extremal-set-theory",
        "exact",
        examples=(
            example(
                "boolean_lattice_2",
                "Family {empty, {a}, {b}, {a,b}} has one union relation.",
                {
                    "family": [
                        [],
                        [0],
                        [1],
                        [0, 1],
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
