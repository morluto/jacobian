"""Domain tests for the exact Boolean Walsh-Hadamard transform."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.boolean import verify_walsh_transform
from jacobian.math.analysis.boolean._models import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.math.analysis.boolean._tools import _walsh_hadamard_transform


def _request(truth_table: list[int]) -> BooleanTruthTableRequest:
    return BooleanTruthTableRequest.model_validate({"truth_table": truth_table})


def _character_sum_spectrum(truth_table: list[int]) -> tuple[int, ...]:
    """Compute the Walsh spectrum directly from its defining character sum."""

    return tuple(
        sum(
            (-1) ** (truth_table[x] + ((u & x).bit_count() & 1))
            for x in range(len(truth_table))
        )
        for u in range(len(truth_table))
    )


def test_walsh_transform_of_constant_zero_on_one_bit() -> None:
    # f=[0,0] -> sign=[1,1] -> spectrum=[2,0]
    result = _walsh_hadamard_transform(_request([0, 0]))
    assert isinstance(result, BooleanWalshTransformResult)
    assert result.spectrum.values == (2, 0)
    assert result.variable_count == 1


def test_walsh_transform_of_identity_on_one_bit() -> None:
    # f=[0,1] -> sign=[1,-1] -> spectrum=[0,2]
    result = _walsh_hadamard_transform(_request([0, 1]))
    assert isinstance(result, BooleanWalshTransformResult)
    assert result.spectrum.values == (0, 2)
    assert result.variable_count == 1


def test_walsh_transform_of_constant_zero_first_nonzero() -> None:
    result = _walsh_hadamard_transform(_request([0, 0, 0, 0]))
    assert result.spectrum.values == (4, 0, 0, 0)
    assert result.variable_count == 2


def test_walsh_transform_of_constant_one_is_all_zeros_except_first() -> None:
    # f=[1,1,1,1] -> sign=[-1,-1,-1,-1] -> spectrum=[-4,0,0,0]
    result = _walsh_hadamard_transform(_request([1, 1, 1, 1]))
    assert result.spectrum.values == (-4, 0, 0, 0)
    assert result.variable_count == 2


def test_walsh_transform_of_not_function() -> None:
    # f(x) = NOT x on one variable: [1, 0] -> sign=[-1, 1] -> spectrum=[0,-2]
    result = _walsh_hadamard_transform(_request([1, 0]))
    assert result.spectrum.values == (0, -2)


def test_serialized_walsh_claim_retains_source_and_rejects_forgery() -> None:
    result = _walsh_hadamard_transform(_request([0, 1]))
    decoded = type(result).model_validate_json(result.model_dump_json())

    assert decoded.source.values[0].as_fraction() == 0
    assert decoded.source.values[1].as_fraction() == 1
    assert decoded.spectrum.values == (0, 2)
    assert all(type(value) is int for value in decoded.spectrum.values)
    assert decoded.spectrum.model_dump(mode="json")["values"] == ["0", "2"]
    assert verify_walsh_transform(decoded)

    payload = result.model_dump(mode="json")
    payload["spectrum"]["values"][1] = "99"
    forged = type(result).model_validate(payload)
    assert not verify_walsh_transform(forged)


def test_walsh_parseval_identity() -> None:
    """Parseval: sum of W_f(u)^2 = 2^(2n) for n variables."""
    for n in range(1, 6):
        truth = [0] * (1 << n)
        truth[0] = 1  # Any function; here delta at 0
        truth[1] = 1
        result = _walsh_hadamard_transform(_request(truth))
        parseval = sum(value**2 for value in result.spectrum.values)
        assert parseval == 1 << (2 * n), f"Parseval failed for n={n}"


def test_walsh_complement_identity() -> None:
    """W_{1-f} = -W_f (complement identity)."""
    for truth in (
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ):
        r1 = _walsh_hadamard_transform(_request(truth))
        complement = [1 - b for b in truth]
        r2 = _walsh_hadamard_transform(_request(complement))
        for v1, v2 in zip(r1.spectrum.values, r2.spectrum.values, strict=True):
            assert v1 == -v2, f"Complement identity failed for {truth}"


def test_walsh_constant_zero_spectrum() -> None:
    """Constant-zero has spectrum [2^n, 0, ..., 0]."""
    for n in range(1, 5):
        result = _walsh_hadamard_transform(_request([0] * (1 << n)))
        assert result.spectrum.values[0] == 1 << n
        assert all(value == 0 for value in result.spectrum.values[1:])


def test_walsh_constant_one_spectrum() -> None:
    """Constant-one has spectrum [-2^n, 0, ..., 0]."""
    for n in range(1, 5):
        result = _walsh_hadamard_transform(_request([1] * (1 << n)))
        assert result.spectrum.values[0] == -(1 << n)
        assert all(value == 0 for value in result.spectrum.values[1:])


def test_walsh_affine_has_one_nonzero() -> None:
    """Affine functions have exactly one nonzero spectral coefficient of magnitude 2^n."""
    # Use f(x)=x_0 on one variable: [0,1] -> sign=[1,-1] -> [0,2].
    result = _walsh_hadamard_transform(_request([0, 1]))
    nonzero = [value for value in result.spectrum.values if value != 0]
    assert len(nonzero) == 1
    assert abs(nonzero[0]) == 2


@pytest.mark.parametrize(
    "truth_table",
    (
        [0],
        [0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 1],
    ),
)
def test_walsh_transform_agrees_with_direct_character_sum(
    truth_table: list[int],
) -> None:
    result = _walsh_hadamard_transform(_request(truth_table))
    assert result.spectrum.values == _character_sum_spectrum(truth_table)


def test_walsh_transform_rejects_non_power_of_two_length() -> None:
    request = BooleanTruthTableRequest(truth_table=(0, 1, 1))
    with pytest.raises(OperationDomainValidationError, match="power of two"):
        _walsh_hadamard_transform(request)


def test_walsh_transform_rejects_empty_truth_table() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        BooleanTruthTableRequest.model_validate({"truth_table": []})


def test_walsh_transform_rejects_non_boolean_entries() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        BooleanTruthTableRequest.model_validate({"truth_table": [0, 1, 1, 2]})


def test_walsh_transform_kernel_rejects_non_binary_values() -> None:
    from jacobian.math.analysis.boolean import walsh_hadamard_transform

    with pytest.raises(OperationDomainValidationError, match="0 or 1"):
        walsh_hadamard_transform([0, 1, 1, 2])
