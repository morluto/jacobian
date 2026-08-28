"""Tests for inverse multiplicative function operations."""

from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative import (
    euler_phi_preimage_count,
    euler_phi_preimage_power_sum,
    euler_phi_preimages,
)
from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageRequest,
)
from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative._tools import (
    TOOLS,
    compute_euler_phi_power_sum,
    compute_euler_phi_preimage,
    compute_euler_phi_preimage_count,
)


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
    assert euler_phi_preimages(1) == result.preimage


def test_preimage_count_of_1() -> None:
    request = EulerPhiPreimageCountRequest(target=1)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 2
    assert euler_phi_preimage_count(1) == result.count


def test_power_sum_of_1_squared() -> None:
    request = EulerPhiPowerSumRequest(target=1, exponent=2)
    result = compute_euler_phi_power_sum(request)
    assert result.power_sum == 5  # 1^2 + 2^2 = 5
    assert result.count == 2
    assert euler_phi_preimage_power_sum(1, 2) == result.power_sum


def test_preimage_of_4() -> None:
    request = EulerPhiPreimageRequest(target=4)
    result = compute_euler_phi_preimage(request)
    assert result.count > 0
    assert 5 in result.preimage  # phi(5) = 4
    assert 8 in result.preimage  # phi(8) = 4
    assert 10 in result.preimage  # phi(10) = 4
    assert 12 in result.preimage  # phi(12) = 4


def test_preimage_of_4_is_complete() -> None:
    """phi^{-1}(4) = {5, 8, 10, 12} — the old 4*target bound found all of these."""
    request = EulerPhiPreimageRequest(target=4)
    result = compute_euler_phi_preimage(request)
    assert result.preimage == (5, 8, 10, 12)
    assert result.count == 4


def test_preimage_of_48_includes_210() -> None:
    """phi(210) = 48 but 210 > 4*48 = 192, so the old bound dropped it."""
    request = EulerPhiPreimageRequest(target=48)
    result = compute_euler_phi_preimage(request)
    assert 210 in result.preimage
    assert result.count == 11
    assert result.count == len(result.preimage)


def test_preimage_count_of_48() -> None:
    request = EulerPhiPreimageCountRequest(target=48)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 11


def test_preimage_of_80_includes_330() -> None:
    """phi(330) = 80 but 330 > 4*80 = 320, so the old bound dropped it."""
    request = EulerPhiPreimageRequest(target=80)
    result = compute_euler_phi_preimage(request)
    assert 330 in result.preimage
    assert result.count == 10


def test_preimage_count_of_80() -> None:
    request = EulerPhiPreimageCountRequest(target=80)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 10


def test_preimage_of_96_includes_390_and_420() -> None:
    """phi(390) = phi(420) = 96 but both exceed 4*96 = 384, so the old bound dropped them."""
    request = EulerPhiPreimageRequest(target=96)
    result = compute_euler_phi_preimage(request)
    assert 390 in result.preimage
    assert 420 in result.preimage
    assert result.count == 17


def test_preimage_count_of_96() -> None:
    request = EulerPhiPreimageCountRequest(target=96)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 17


def test_preimage_of_480_includes_2310() -> None:
    """phi(2310) = 480 but 2310 > 4*480 = 1920, so the old bound dropped it."""
    request = EulerPhiPreimageRequest(target=480)
    result = compute_euler_phi_preimage(request)
    assert 2310 in result.preimage
    assert result.count == 37


def test_preimage_count_of_480() -> None:
    request = EulerPhiPreimageCountRequest(target=480)
    result = compute_euler_phi_preimage_count(request)
    assert result.count == 37


def test_power_sum_of_48() -> None:
    """Power sum of the complete preimage of 48, including the previously-missed 210."""
    request = EulerPhiPowerSumRequest(target=48, exponent=1)
    result = compute_euler_phi_power_sum(request)
    expected = 65 + 104 + 105 + 112 + 130 + 140 + 144 + 156 + 168 + 180 + 210
    assert result.power_sum == expected
    assert result.count == 11


def test_preimage_odd_target_above_one_has_no_solutions() -> None:
    """phi(n) is even for n > 2, so odd targets > 1 have no preimage."""
    for target in (3, 5, 7, 9, 99):
        request = EulerPhiPreimageRequest(target=target)
        result = compute_euler_phi_preimage(request)
        assert result.preimage == ()
        assert result.count == 0
