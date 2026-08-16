"""Tests for arithmetic function operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.arithmetic_functions import (
    DirichletConvolutionRequest,
    DirichletInverseRequest,
    MobiusTransformRequest,
    SummatoryFunctionRequest,
)
from jacobian.domains.number_theory.arithmetic_function_operations import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)


def test_dirichlet_convolution_identity():
    """(id * id)(n) = sum_{d|n} d * (n/d) = n * tau(n) where tau is divisor count."""
    # id = [1, 2, 3, 4, 5, 6]
    # (id * id)(1) = 1*1 = 1
    # (id * id)(2) = 1*2 + 2*1 = 4
    # (id * id)(3) = 1*3 + 3*1 = 6
    # (id * id)(4) = 1*4 + 2*2 + 4*1 = 12
    result = compute_dirichlet_convolution(
        DirichletConvolutionRequest.model_validate({
            "left": [1, 2, 3, 4],
            "right": [1, 2, 3, 4],
        })
    )
    assert result.values[0] == 1
    assert result.values[1] == 4
    assert result.values[2] == 6
    assert result.values[3] == 12


def test_dirichlet_convolution_euler_totient():
    """(phi * 1)(n) = n. phi = [1,1,2,2,4,2,6,4]."""
    # phi(n) for n=1..8: [1,1,2,2,4,2,6,4]
    # ones = [1,1,1,1,1,1,1,1]
    # (phi * 1)(n) should give [1, 2, 3, 4, 5, 6, 7, 8]
    result = compute_dirichlet_convolution(
        DirichletConvolutionRequest.model_validate({
            "left": [1, 1, 2, 2, 4, 2, 6, 4],
            "right": [1, 1, 1, 1, 1, 1, 1, 1],
        })
    )
    assert result.values == (1, 2, 3, 4, 5, 6, 7, 8)


def test_divisor_transform():
    """Divisor sum of [1,1,1,1] is [1, 3, 1, 7]."""
    result = compute_mobius_transform(
        MobiusTransformRequest.model_validate({"values": [1, 1, 1, 1]})
    )
    # g(1) = f(1) = 1
    # g(2) = f(1) + f(2) = 2
    # g(3) = f(1) + f(3) = 2
    # g(4) = f(1) + f(2) + f(4) = 3
    assert result.values[0] == 1
    assert result.values[1] == 2
    assert result.values[2] == 2
    assert result.values[3] == 3


def test_dirichlet_inverse():
    """Dirichlet inverse of the constant function 1 is the Mobius function."""
    # 1 = [1,1,1,1,1,1,1,1]
    # inverse should give mu(n) = [1,-1,-1,0,-1,1,-1,0]
    result = compute_dirichlet_inverse(
        DirichletInverseRequest.model_validate({
            "values": [1, 1, 1, 1, 1, 1, 1, 1],
        })
    )
    assert result.values[0] == 1  # mu(1) = 1
    assert result.values[1] == -1  # mu(2) = -1
    assert result.values[2] == -1  # mu(3) = -1
    assert result.values[3] == 0  # mu(4) = 0
    assert result.values[4] == -1  # mu(5) = -1
    assert result.values[5] == 1  # mu(6) = 1
    assert result.values[6] == -1  # mu(7) = -1
    assert result.values[7] == 0  # mu(8) = 0


def test_summatory_function():
    """Summatory function of [1,2,3,4] is [1,3,6,10]."""
    result = compute_summatory_function(
        SummatoryFunctionRequest.model_validate({"values": [1, 2, 3, 4]})
    )
    assert result.values == (1, 3, 6, 10)


def test_dirichlet_inverse_requires_f1_one():
    """Dirichlet inverse requires f(1) = 1."""
    with pytest.raises(ValidationError, match="Dirichlet inverse requires"):
        DirichletInverseRequest.model_validate({"values": [2, 1, 1]})


def test_convolution_length_mismatch():
    """Mismatched lengths should fail."""
    with pytest.raises(ValidationError, match="same length"):
        DirichletConvolutionRequest.model_validate({
            "left": [1, 2, 3],
            "right": [1, 2],
        })


def test_operations_discoverable():
    """All four operations should be discoverable."""
    from jacobian.domains.number_theory import number_theory_operations

    ops = number_theory_operations()
    op_ids = [op.operation_id for op in ops]
    assert "number_theory.arithmetic_function.dirichlet_convolution.compute" in op_ids
    assert "number_theory.arithmetic_function.divisor_transform.compute" in op_ids
    assert "number_theory.arithmetic_function.dirichlet_inverse.compute" in op_ids
    assert "number_theory.arithmetic_function.summatory.compute" in op_ids
