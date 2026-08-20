"""Domain tests for the exact Boolean Walsh-Hadamard transform."""

from __future__ import annotations

import pytest

from jacobian.math.boolean._models import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.math.boolean._operations import compute_walsh_hadamard_transform


def _request(truth_table: list[int]) -> BooleanTruthTableRequest:
    return BooleanTruthTableRequest(truth_table=tuple(truth_table))


def test_walsh_transform_of_constant_zero_on_one_bit() -> None:
    # f=[0,0] -> sign=[1,1] -> spectrum=[2,0]
    result = compute_walsh_hadamard_transform(_request([0, 0]))
    assert isinstance(result, BooleanWalshTransformResult)
    assert result.spectrum == ("2", "0")
    assert result.variable_count == 1


def test_walsh_transform_of_identity_on_one_bit() -> None:
    # f=[0,1] -> sign=[1,-1] -> spectrum=[0,2]
    result = compute_walsh_hadamard_transform(_request([0, 1]))
    assert isinstance(result, BooleanWalshTransformResult)
    assert result.spectrum == ("0", "2")
    assert result.variable_count == 1


def test_walsh_transform_of_constant_zero_first_nonzero() -> None:
    result = compute_walsh_hadamard_transform(_request([0, 0, 0, 0]))
    assert result.spectrum == ("4", "0", "0", "0")
    assert result.variable_count == 2


def test_walsh_transform_of_constant_one_is_all_zeros_except_first() -> None:
    # f=[1,1,1,1] -> sign=[-1,-1,-1,-1] -> spectrum=[-4,0,0,0]
    result = compute_walsh_hadamard_transform(_request([1, 1, 1, 1]))
    assert result.spectrum == ("-4", "0", "0", "0")
    assert result.variable_count == 2


def test_walsh_transform_of_not_function() -> None:
    # f(x) = NOT x on one variable: [1, 0] -> sign=[-1, 1] -> spectrum=[0,-2]
    result = compute_walsh_hadamard_transform(_request([1, 0]))
    assert result.spectrum == ("0", "-2")


def test_walsh_parseval_identity() -> None:
    """Parseval: sum of W_f(u)^2 = 2^(2n) for n variables."""
    for n in range(1, 6):
        truth = [0] * (1 << n)
        truth[0] = 1  # Any function; here delta at 0
        truth[1] = 1
        result = compute_walsh_hadamard_transform(_request(truth))
        parseval = sum(int(v) ** 2 for v in result.spectrum)
        assert parseval == 1 << (2 * n), f"Parseval failed for n={n}"


def test_walsh_complement_identity() -> None:
    """W_{1-f} = -W_f (complement identity)."""
    for truth in (
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ):
        r1 = compute_walsh_hadamard_transform(_request(truth))
        complement = [1 - b for b in truth]
        r2 = compute_walsh_hadamard_transform(_request(complement))
        for v1, v2 in zip(r1.spectrum, r2.spectrum, strict=False):
            assert int(v1) == -int(v2), f"Complement identity failed for {truth}"


def test_walsh_constant_zero_spectrum() -> None:
    """Constant-zero has spectrum [2^n, 0, ..., 0]."""
    for n in range(1, 5):
        result = compute_walsh_hadamard_transform(_request([0] * (1 << n)))
        assert int(result.spectrum[0]) == 1 << n
        assert all(int(v) == 0 for v in result.spectrum[1:])


def test_walsh_constant_one_spectrum() -> None:
    """Constant-one has spectrum [-2^n, 0, ..., 0]."""
    for n in range(1, 5):
        result = compute_walsh_hadamard_transform(_request([1] * (1 << n)))
        assert int(result.spectrum[0]) == -(1 << n)
        assert all(int(v) == 0 for v in result.spectrum[1:])


def test_walsh_affine_has_one_nonzero() -> None:
    """Affine functions have exactly one nonzero spectral coefficient of magnitude 2^n."""
    # f(x) = x_0 (linear) on 2 variables: truth table [0,0,1,1] (index 0->0, 1->0, 2->1, 3->1)
    # Actually for f(x) = x_0: f(0)=0, f(1)=0, f(2)=1, f(3)=1 in natural order where x=(x0,x1), index=x0+2*x1
    # Wait, let's use f(x)=x0 on 1 variable: [0,1] -> sign=[1,-1] -> [0,2]
    result = compute_walsh_hadamard_transform(_request([0, 1]))
    nonzero = [int(v) for v in result.spectrum if int(v) != 0]
    assert len(nonzero) == 1
    assert abs(nonzero[0]) == 2


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
