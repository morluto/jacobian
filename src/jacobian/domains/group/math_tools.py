"""Exact finite group operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.group import (
    GroupElementOrderRequest,
    GroupElementOrderResult,
    GroupOrbitRequest,
    GroupOrbitResult,
    GroupOrderResult,
    PermutationGroupRequest,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.group.operations import (
    compute_element_order,
    compute_group_orbit,
    compute_group_order,
)
from jacobian.math_tools import MathTool


def group_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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
)
