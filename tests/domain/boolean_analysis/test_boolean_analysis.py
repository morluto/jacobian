"""Tests for Boolean analysis operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.boolean_analysis import (
    BooleanErasureNoiseRequest,
    BooleanFourierRequest,
    BooleanInfluenceRequest,
    BooleanMultilinearExtensionRequest,
    BooleanTruthTable,
)
from jacobian.domains.boolean_analysis.operations import (
    compute_boolean_fourier,
    compute_boolean_influence,
    compute_erasure_noise,
    compute_multilinear_extension,
)


def test_fourier_of_and():
    """AND(a,b) = [-1,-1,-1,1] has Fourier coefficients: f_hat(00)=-1, f_hat(01)=-1, f_hat(10)=-1, f_hat(11)=1."""
    result = compute_boolean_fourier(
        BooleanFourierRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, -1, -1, 1],
            },
        })
    )
    coeffs = {c.subset_mask: c.coefficient for c in result.coefficients}
    # AND has f_hat(0) = -1, f_hat({a}) = -1, f_hat({b}) = -1, f_hat({a,b}) = 1
    # But these are numerators with denominator 2^2=4
    # f(x) = sum_S f_hat(S) chi_S(x), so f_hat(S) = (1/2^n) sum_x f(x) chi_S(x)
    assert coeffs[0] == -2  # f_hat(empty) * 4
    assert coeffs[3] == 2   # f_hat({a,b}) * 4


def test_fourier_of_dictator():
    """Dictator f(a,b) = a has Fourier coefficient only for {a}."""
    result = compute_boolean_fourier(
        BooleanFourierRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, 1, -1, 1],
            },
        })
    )
    coeffs = {c.subset_mask: c.coefficient for c in result.coefficients}
    assert coeffs[1] == 4  # f_hat({a}) * 4 = 4
    assert coeffs[0] == 0
    assert coeffs[2] == 0
    assert coeffs[3] == 0


def test_influence_of_and():
    """AND(a,b) has influence 2 for each variable."""
    result = compute_boolean_influence(
        BooleanInfluenceRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, -1, -1, 1],
            },
        })
    )
    assert result.influences == (2, 2)
    assert result.total_influence == 4


def test_influence_of_dictator():
    """Dictator f(a,b) = a has influence 4 for a, 0 for b."""
    result = compute_boolean_influence(
        BooleanInfluenceRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, 1, -1, 1],
            },
        })
    )
    assert result.influences == (4, 0)


def test_erasure_noise_no_erasure():
    """With 0 erasures, E|f(z)| = 1."""
    result = compute_erasure_noise(
        BooleanErasureNoiseRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, -1, -1, 1],
            },
            "erasure_count": 0,
        })
    )
    assert result.expected_absolute_value_numerator == 1
    assert result.expected_absolute_value_denominator == 1


def test_erasure_noise_one_erasure():
    """With 1 erasure on AND, E|f(z)| should be computable."""
    result = compute_erasure_noise(
        BooleanErasureNoiseRequest.model_validate({
            "truth_table": {
                "variable_names": ["a", "b"],
                "values": [-1, -1, -1, 1],
            },
            "erasure_count": 1,
        })
    )
    assert result.expected_absolute_value_denominator > 0
    assert result.expected_absolute_value_numerator > 0


def test_truth_table_validation():
    """Invalid truth table length should fail."""
    with pytest.raises(ValidationError, match="truth table must have"):
        BooleanTruthTable.model_validate({
            "variable_names": ["a", "b"],
            "values": [-1, 1, -1],
        })


def test_operations_discoverable():
    """All four operations should be discoverable."""
    from jacobian.domains.boolean_analysis import boolean_analysis_operations

    ops = boolean_analysis_operations()
    op_ids = [op.operation_id for op in ops]
    assert "boolean.fourier.compute" in op_ids
    assert "boolean.multilinear_extension.evaluate" in op_ids
    assert "boolean.influence.compute" in op_ids
    assert "boolean.erasure_noise.compute" in op_ids
