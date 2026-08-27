"""Inverse multiplicative function operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPowerSumResult,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageCountResult,
    EulerPhiPreimageRequest,
    EulerPhiPreimageResult,
)
from jacobian.math.inverse_multiplicative._operations import (
    compute_euler_phi_power_sum,
    compute_euler_phi_preimage,
    compute_euler_phi_preimage_count,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    discovery_terms: tuple[str, ...] = (),
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
        discovery_terms=discovery_terms,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "number_theory.euler_phi.preimages.compute",
        "Compute the preimage of the Euler totient function",
        "Find all n such that phi(n) = target, where phi is Euler's totient "
        "function. Builds the complete preimage exactly via the recursive prime-factor construction.",
        EulerPhiPreimageRequest,
        EulerPhiPreimageResult,
        compute_euler_phi_preimage,
        "number-theory",
        "euler-phi",
        "exact",
        discovery_terms=(
            "inverse totient",
            "inverse phi",
            "totient inverse image",
            "solve phi",
        ),
        examples=(
            example(
                "phi_preimage_1",
                "Find all n with phi(n) = 1.",
                {"target": 1},
            ),
        ),
    ),
    _op(
        "number_theory.euler_phi.preimage_count.compute",
        "Count the preimage of the Euler totient function",
        "Count the number of n such that phi(n) = target.",
        EulerPhiPreimageCountRequest,
        EulerPhiPreimageCountResult,
        compute_euler_phi_preimage_count,
        "number-theory",
        "euler-phi",
        "exact",
        examples=(
            example(
                "phi_preimage_count_1",
                "Count n with phi(n) = 1.",
                {"target": 1},
            ),
        ),
    ),
    _op(
        "number_theory.euler_phi.preimage_power_sums.compute",
        "Compute the sum of k-th powers of the phi preimage",
        "Compute sum of n^k for all n with phi(n) = target.",
        EulerPhiPowerSumRequest,
        EulerPhiPowerSumResult,
        compute_euler_phi_power_sum,
        "number-theory",
        "euler-phi",
        "exact",
        examples=(
            example(
                "phi_power_sum_1_2",
                "Compute sum of squares of phi preimage of 1.",
                {"target": 1, "exponent": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
