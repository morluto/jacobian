"""Permutation group operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.contracts.permutation_group import (
    PermutationGroupOrbitRequest,
    PermutationGroupOrbitResult,
    PermutationGroupOrderResult,
    PermutationGroupRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.permutation_group.operations import (
    compute_pg_orbit,
    compute_pg_order,
)
from jacobian.math_tools import MathTool


def pg_operation[RequestT: ContractModel, ResultT: ContractModel](
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


PERMUTATION_GROUP_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    pg_operation(
        "permutation_group.order.compute",
        "Compute the order of a permutation group",
        "Compute the exact order of a permutation group via SymPy's Schreier-Sims algorithm.",
        PermutationGroupRequest,
        PermutationGroupOrderResult,
        compute_pg_order,
        "permutation-group",
        "order",
        "exact",
        examples=(
            example(
                "cyclic_pg_order",
                "Order of C4.",
                {"degree": 4, "generators": [[1, 2, 3, 0]]},
            ),
        ),
    ),
    pg_operation(
        "permutation_group.orbit.compute",
        "Compute the orbit of a point under a permutation group",
        "Compute the orbit of a point under a permutation group using SymPy.",
        PermutationGroupOrbitRequest,
        PermutationGroupOrbitResult,
        compute_pg_orbit,
        "permutation-group",
        "orbit",
        "exact",
        examples=(
            example(
                "cyclic_pg_orbit",
                "Orbit of 0 under C4.",
                {"degree": 4, "generators": [[1, 2, 3, 0]], "point": 0},
            ),
        ),
    ),
)
