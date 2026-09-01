"""Inverse multiplicative function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPowerSumResult,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageCountResult,
    EulerPhiPreimageRequest,
    EulerPhiPreimageResult,
)
from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative.operations import (
    euler_phi_preimage_count,
    euler_phi_preimage_power_profile,
    euler_phi_preimages,
)


def compute_euler_phi_preimage(
    request: EulerPhiPreimageRequest,
) -> EulerPhiPreimageResult:
    preimage = euler_phi_preimages(request.target)
    return EulerPhiPreimageResult(preimage=preimage, count=len(preimage))


def compute_euler_phi_preimage_count(
    request: EulerPhiPreimageCountRequest,
) -> EulerPhiPreimageCountResult:
    return EulerPhiPreimageCountResult(count=euler_phi_preimage_count(request.target))


def compute_euler_phi_power_sum(
    request: EulerPhiPowerSumRequest,
) -> EulerPhiPowerSumResult:
    power_sum, count = euler_phi_preimage_power_profile(
        request.target, request.exponent
    )
    return EulerPhiPowerSumResult(
        power_sum=power_sum,
        count=count,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_theory.euler_phi.preimages.compute",
        title="Compute the preimage of the Euler totient function",
        description="Find all n such that phi(n) = target, where phi is Euler's totient "
        "function. Builds the complete preimage exactly via the recursive prime-factor construction.",
        request_type=EulerPhiPreimageRequest,
        result_type=EulerPhiPreimageResult,
        run=compute_euler_phi_preimage,
        tags=("number-theory", "euler-phi", "exact"),
        discovery_terms=(
            "inverse totient",
            "inverse phi",
            "totient inverse image",
            "solve phi",
        ),
        examples=(
            OperationExample(
                name="phi_preimage_1",
                description="Find all n with phi(n) = 1.",
                input={"target": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.euler_phi.preimage_count.compute",
        title="Count the preimage of the Euler totient function",
        description="Count the number of n such that phi(n) = target.",
        request_type=EulerPhiPreimageCountRequest,
        result_type=EulerPhiPreimageCountResult,
        run=compute_euler_phi_preimage_count,
        tags=("number-theory", "euler-phi", "exact"),
        examples=(
            OperationExample(
                name="phi_preimage_count_1",
                description="Count n with phi(n) = 1.",
                input={"target": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.euler_phi.preimage_power_sums.compute",
        title="Compute the sum of k-th powers of the phi preimage",
        description="Compute sum of n^k for all n with phi(n) = target.",
        request_type=EulerPhiPowerSumRequest,
        result_type=EulerPhiPowerSumResult,
        run=compute_euler_phi_power_sum,
        tags=("number-theory", "euler-phi", "exact"),
        examples=(
            OperationExample(
                name="phi_power_sum_1_2",
                description="Compute sum of squares of phi preimage of 1.",
                input={"target": 1, "exponent": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
