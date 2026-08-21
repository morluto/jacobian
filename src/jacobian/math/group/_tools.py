"""Exact finite group operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.group._models import (
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    GroupStabilizerRequest,
    GroupStabilizerResult,
    PermutationGroupRequest,
)
from jacobian.math.group._operations import (
    compute_element_order,
    compute_group_orbit,
    compute_group_order,
    compute_group_stabilizer,
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


S3_STABILIZER_POINT_0 = {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]], "point": 0}

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
        "group.stabilizer.compute",
        "Compute the stabilizer of a point in a permutation group",
        "Given a permutation group by generators and a point, return generators "
        "of the point stabilizer subgroup (elements fixing the point) using "
        "SymPy's stabilizer computation. By the orbit-stabilizer theorem, "
        "|G| = |orbit(point)| * |stabilizer(point)|, composable with "
        "group.order.compute and group.orbit.compute.",
        GroupStabilizerRequest,
        GroupStabilizerResult,
        compute_group_stabilizer,
        "group",
        "permutation",
        "stabilizer",
        "orbit-stabilizer",
        "exact",
        examples=(
            example(
                "s3_stabilizer_of_0",
                (
                    "Stabilizer of point 0 in S3 (generators (1,2,0) and "
                    "(1,0,2)); the stabilizer has order 2 and orbit-stabilizer "
                    "gives 6 = 3 * 2. Generators must be permutations of 0..n-1."
                ),
                S3_STABILIZER_POINT_0,
            ),
        ),
    ),
)

TOOLS = GROUP_OPERATIONS

__all__ = ["TOOLS"]
