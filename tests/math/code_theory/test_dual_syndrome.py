"""Tests for dual code and syndrome operations."""

from jacobian.math.code_theory._dual_operations import (
    compute_dual_code,
    compute_syndrome,
)
from jacobian.math.code_theory._models import DualCodeRequest, SyndromeRequest


def test_dual_hamming_7_4() -> None:
    result = compute_dual_code(
        DualCodeRequest(
            field_order=2,
            generator_matrix=(
                (1, 0, 0, 0, 1, 1, 0),
                (0, 1, 0, 0, 1, 0, 1),
                (0, 0, 1, 0, 0, 1, 1),
                (0, 0, 0, 1, 1, 1, 1),
            ),
        )
    )
    assert result.code_dimension == 4
    assert result.code_length == 7
    assert result.dual_dimension == 3
    assert len(result.parity_check_matrix) == 3
    assert len(result.parity_check_matrix[0]) == 7


def test_dual_identity_matrix() -> None:
    result = compute_dual_code(
        DualCodeRequest(
            field_order=2,
            generator_matrix=((1, 0), (0, 1)),
        )
    )
    assert result.code_dimension == 2
    assert result.code_length == 2
    assert result.dual_dimension == 0


def test_syndrome_zero() -> None:
    result = compute_syndrome(
        SyndromeRequest(
            field_order=2,
            parity_check_matrix=((1, 1, 0), (0, 1, 1)),
            received_word=(0, 0, 0),
        )
    )
    assert result.syndrome == (0, 0)


def test_syndrome_nonzero() -> None:
    result = compute_syndrome(
        SyndromeRequest(
            field_order=2,
            parity_check_matrix=((1, 1, 0), (0, 1, 1)),
            received_word=(1, 0, 1),
        )
    )
    assert result.syndrome == (1, 1)


def test_syndrome_mod_3() -> None:
    result = compute_syndrome(
        SyndromeRequest(
            field_order=3,
            parity_check_matrix=((1, 1), (0, 1)),
            received_word=(2, 2),
        )
    )
    # s = (1*2+1*2) mod 3 = 4 mod 3 = 1, (0*2+1*2) mod 3 = 2
    assert result.syndrome == (1, 2)
