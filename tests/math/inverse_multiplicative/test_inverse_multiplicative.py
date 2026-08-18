"""Tests for inverse multiplicative function operations."""

from jacobian.math.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageRequest,
)
from jacobian.math.inverse_multiplicative._operations import (
    compute_euler_phi_power_sum,
    compute_euler_phi_preimage,
    compute_euler_phi_preimage_count,
)
from jacobian.math.inverse_multiplicative._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "number_theory.euler_phi.preimages.compute",
        "number_theory.euler_phi.preimage_count.compute",
        "number_theory.euler_phi.preimage_power_sums.compute",
    }


def test_preimage_of_1() -> None:
    request = EulerPhiPreimageRequest(target=1)
    result = compute_euler_phi_preimage(request)
    assert result.preimage == (1, 2)
    assert result.count == 2


def test_preimage_count_of_1() -> None:
    request = EulerPhiPreimageCountRequest(target=1)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 2


def test_power_sum_of_1_squared() -> None:
    request = EulerPhiPowerSumRequest(target=1, exponent=2)
    result = compute_euler_phi_power_sum(request)
    assert result.power_sum == 5  # 1^2 + 2^2 = 5
    assert result.count == 2


def test_preimage_of_4() -> None:
    request = EulerPhiPreimageRequest(target=4)
    result = compute_euler_phi_preimage(request)
    assert result.count > 0
    assert 5 in result.preimage  # phi(5) = 4
    assert 8 in result.preimage  # phi(8) = 4
    assert 10 in result.preimage  # phi(10) = 4
    assert 12 in result.preimage  # phi(12) = 4
