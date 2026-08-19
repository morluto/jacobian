"""Domain functions for inverse multiplicative function operations."""

from __future__ import annotations

from jacobian.math.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPowerSumResult,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageCountResult,
    EulerPhiPreimageRequest,
    EulerPhiPreimageResult,
)


def _euler_phi(n: int) -> int:
    """Compute Euler's totient function phi(n)."""
    if n <= 0:
        return 0
    result = n
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def compute_euler_phi_preimage(
    request: EulerPhiPreimageRequest,
) -> EulerPhiPreimageResult:
    """Compute all n such that phi(n) = target."""
    target = request.target
    if target == 1:
        return EulerPhiPreimageResult(preimage=(1, 2), count=2)
    preimage: list[int] = []
    upper = 4 * target if target > 1 else 2
    for n in range(1, upper + 1):
        if _euler_phi(n) == target:
            preimage.append(n)
    return EulerPhiPreimageResult(
        preimage=tuple(preimage),
        count=len(preimage),
    )


def compute_euler_phi_preimage_count(
    request: EulerPhiPreimageCountRequest,
) -> EulerPhiPreimageCountResult:
    """Count the number of n such that phi(n) = target."""
    target = request.target
    if target == 1:
        return EulerPhiPreimageCountResult(count=2)
    count = 0
    upper = 4 * target if target > 1 else 2
    for n in range(1, upper + 1):
        if _euler_phi(n) == target:
            count += 1
    return EulerPhiPreimageCountResult(count=count)


def compute_euler_phi_power_sum(
    request: EulerPhiPowerSumRequest,
) -> EulerPhiPowerSumResult:
    """Compute the sum of k-th powers of the preimage of phi."""
    target = request.target
    exponent = request.exponent
    if target == 1:
        return EulerPhiPowerSumResult(power_sum=1**exponent + 2**exponent, count=2)
    total = 0
    count = 0
    upper = 4 * target if target > 1 else 2
    for n in range(1, upper + 1):
        if _euler_phi(n) == target:
            total += n**exponent
            count += 1
    return EulerPhiPowerSumResult(power_sum=total, count=count)
