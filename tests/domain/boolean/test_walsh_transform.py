"""Domain tests for the exact Boolean Walsh-Hadamard transform."""

from __future__ import annotations

import pytest

from jacobian.contracts.boolean import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.domains.boolean.operations import compute_walsh_hadamard_transform


def _request(truth_table: list[int]) -> BooleanTruthTableRequest:
    return BooleanTruthTableRequest(truth_table=tuple(truth_table))


def test_walsh_transform_of_delta_is_all_ones() -> None:
    # f(0)=1, f(1)=0 (majority of one bit inverted) -> WHT = [1, 1]
    result = compute_walsh_hadamard_transform(_request([1, 0]))
    assert isinstance(result, BooleanWalshTransformResult)
    assert result.spectrum == ("1", "1")
    assert result.variable_count == 1


def test_walsh_transform_of_constant_zero_is_all_zeros() -> None:
    result = compute_walsh_hadamard_transform(_request([0, 0, 0, 0]))
    assert result.spectrum == ("0", "0", "0", "0")
    assert result.variable_count == 2


def test_walsh_transform_of_constant_one_is_all_twos() -> None:
    result = compute_walsh_hadamard_transform(_request([1, 1, 1, 1]))
    assert result.spectrum == ("4", "0", "0", "0")
    assert result.variable_count == 2


def test_walsh_transform_of_not_function() -> None:
    # f(x) = NOT x on one variable: [1, 0]
    result = compute_walsh_hadamard_transform(_request([1, 0]))
    assert result.spectrum == ("1", "1")


def test_walsh_transform_round_trip_is_involutive() -> None:
    # WHT(WHT(f)) = n * f, where n is the length of the truth table
    from sympy.discrete.transforms import fwht

    truth = [0, 1, 1, 0, 1, 0, 0, 1]
    spectrum = fwht(truth)
    # WHT is its own inverse up to a factor of n
    doubled = fwht(spectrum)
    n = len(truth)
    assert [value // n for value in doubled] == truth


def test_walsh_transform_rejects_non_power_of_two_length() -> None:
    with pytest.raises(ValueError, match="power of two"):
        BooleanTruthTableRequest(truth_table=(0, 1, 1))


def test_walsh_transform_rejects_empty_truth_table() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        BooleanTruthTableRequest.model_validate({"truth_table": []})


def test_walsh_transform_rejects_non_boolean_entries() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        BooleanTruthTableRequest.model_validate({"truth_table": [0, 1, 1, 2]})


def test_walsh_transform_kernel_rejects_non_binary_values() -> None:
    from jacobian.math.boolean import walsh_hadamard_transform

    with pytest.raises(ValueError, match="0 or 1"):
        walsh_hadamard_transform([0, 1, 1, 2])
