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
    GroupSubgroupLatticeRequest,
    GroupSubgroupLatticeResult,
    PermutationGroupRequest,
)
from jacobian.math.group._operations import (
    compute_conjugacy_classes,
    compute_element_order,
    compute_group_orbit,
    compute_group_order,
    compute_subgroup_lattice,
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
        "Compute the exact conjugacy classes and class sizes of a bounded "
        "permutation group via SymPy. Each class is returned with its "
        "representative elements and size. The generated group must have "
        "order at most 5000.",
        GroupConjugacyClassesRequest,
        GroupConjugacyClassesResult,
        compute_conjugacy_classes,
        "group",
        "conjugacy",
        "permutation",
        "exact",
        examples=(
            example(
                "s3_conjugacy_classes",
                "Compute conjugacy classes of S3; generators must be bijections.",
                {
                    "degree": 3,
                    "generators": [[1, 0, 2], [0, 2, 1]],
                },
            ),
        ),
    ),
    group_operation(
        "group.subgroup_lattice.compute",
        "Enumerate all subgroups of a bounded permutation group",
        "Enumerate all subgroups of a bounded permutation group via SymPy. "
        "Each subgroup is returned with its generators and order. Bounded "
        "to groups of order at most 64.",
        GroupSubgroupLatticeRequest,
        GroupSubgroupLatticeResult,
        compute_subgroup_lattice,
        "group",
        "subgroup",
        "permutation",
        "exact",
        examples=(
            example(
                "c4_subgroups",
                "Enumerate all subgroups of C4; generators must be bijections.",
                {
                    "degree": 4,
                    "generators": [[1, 2, 3, 0]],
                },
            ),
        ),
    ),
)

TOOLS = GROUP_OPERATIONS

__all__ = ["TOOLS"]
