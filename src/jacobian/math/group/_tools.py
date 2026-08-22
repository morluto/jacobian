"""Exact finite group operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.group._models import (
    GroupConjugacyClassesRequest,
    GroupConjugacyClassesResult,
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    PermutationGroupRequest,
)
from jacobian.math.group._operations import (
    compute_element_order,
    compute_group_conjugacy_classes,
    compute_group_orbit,
    compute_group_order,
)


def group_operation[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


S3_GENERATORS = {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]}

GROUP_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    group_operation(
        "group.order.compute",
        "Compute the exact order of a finite permutation group",
        "Compute the exact order of a permutation group given by generators via SymPy's Schreier-Sims algorithm.",
        PermutationGroupRequest,
        GroupOrderResult,
        compute_group_order,
        "group",
        "order",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_group_order_4",
                "Compute C4's order; each generator must be a bijection of 0..degree-1.",
                {
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
    group_operation(
        "group.element_order.compute",
        "Compute the exact order of one permutation",
        "Compute the order of one permutation element via SymPy's Permutation.order().",
        GroupElementOrderRequest,
        GroupElementOrderResult,
        compute_element_order,
        "group",
        "element-order",
        "permutation",
        "exact",
        examples=(
            example(
                "four_cycle_order",
                "Compute the 4-cycle's order; its generator must be a bijection of 0..degree-1.",
                {
                    "degree": 4,
                    "generator": [1, 2, 3, 0],
                },
            ),
        ),
    ),
    group_operation(
        "group.orbit.compute",
        "Compute the orbit of a point under a permutation group",
        "Compute the orbit of a point under a permutation group given by generators via SymPy's PermutationGroup.orbit().",
        GroupOrbitRequest,
        GroupOrbitResult,
        compute_group_orbit,
        "group",
        "orbit",
        "permutation",
        "exact",
        examples=(
            example(
                "cyclic_orbit",
                "Compute point 0's orbit; generators must be bijections and points lie in 0..degree-1.",
                {
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                    "point": 0,
                },
            ),
        ),
    ),
    group_operation(
        "group.conjugacy_classes.compute",
        "Compute conjugacy classes of a permutation group",
        "Given a permutation group by generators, return its conjugacy classes "
        "(the partition into conjugacy classes) as permutation array forms, "
        "using SymPy. Each class lists the elements conjugate to a "
        "representative; class sizes are orbit sizes under conjugation. "
        "Classes are canonically ordered (members sorted, classes sorted by "
        "smallest member), so the same group always yields an identical "
        "result. The generated group must have order at most 5000; larger "
        "groups are rejected before enumeration.",
        GroupConjugacyClassesRequest,
        GroupConjugacyClassesResult,
        compute_group_conjugacy_classes,
        "group",
        "permutation",
        "conjugacy",
        "exact",
        examples=(
            example(
                "s3_conjugacy_classes",
                (
                    "Conjugacy classes of S3 (generators (1,2,0) and (1,0,2)); "
                    "S3 has three classes of sizes 1, 2, 3 (identity, 3-cycles, "
                    "transpositions). Generators must be permutations of 0..n-1."
                ),
                S3_GENERATORS,
            ),
        ),
    ),
)

TOOLS = GROUP_OPERATIONS

__all__ = ["TOOLS"]
