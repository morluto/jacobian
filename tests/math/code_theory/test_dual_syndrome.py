"""Regression tests for the retained dual-code and syndrome operations."""

from jacobian.math.code_theory._dual_operations import (
    compute_dual_code,
    compute_syndrome,
)
from jacobian.math.code_theory._models import DualCodeRequest, SyndromeRequest
from jacobian.math.code_theory._tools import TOOLS


def test_dual_and_syndrome_operation_ids_remain_published() -> None:
    operation_ids = {tool.operation_id for tool in TOOLS}
    assert {"code.dual_code.compute", "code.syndrome.compute"} <= operation_ids


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


def test_dual_code_is_orthogonal_over_the_prime_field() -> None:
    request = DualCodeRequest(field_order=3, generator_matrix=((2, 1),))
    result = compute_dual_code(request)
    assert result.code_dimension == result.dual_dimension == 1
    assert all(
        sum(
            generator * dual
            for generator, dual in zip(request.generator_matrix[0], row, strict=True)
        )
        % request.field_order
        == 0
        for row in result.parity_check_matrix
    )


def test_syndrome_is_computed_modulo_the_field_order() -> None:
    result = compute_syndrome(
        SyndromeRequest(
            field_order=3,
            parity_check_matrix=((1, 1), (0, 1)),
            received_word=(2, 2),
        )
    )
    assert result.syndrome == (1, 2)
