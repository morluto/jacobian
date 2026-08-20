"""Tests for polynomial interpolation operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.polynomial_interpolation_ops._models import (
    DividedDifferencesRequest,
    NewtonEvaluateRequest,
    NewtonFormRequest,
)
from jacobian.math.polynomial_interpolation_ops._operations import (
    compute_divided_differences,
    compute_newton_evaluate,
    compute_newton_form,
)
from jacobian.math.polynomial_interpolation_ops._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.interpolation.divided_differences.compute",
        "polynomial.interpolation.newton_form.compute",
        "polynomial.interpolation.newton_evaluate.compute",
    }


def test_divided_differences_basic() -> None:
    request = DividedDifferencesRequest(nodes=("0", "1", "2"), values=("1", "2", "5"))
    result = compute_divided_differences(request)
    assert result.coefficients == ("1", "1", "1")


def test_newton_form_basic() -> None:
    request = NewtonFormRequest(nodes=("0", "1", "2"), values=("1", "2", "5"))
    result = compute_newton_form(request)
    assert result.coefficients == ("1", "1", "1")
    assert result.nodes == ("0", "1", "2")


def test_newton_evaluate_at_3() -> None:
    request = NewtonEvaluateRequest(
        nodes=("0", "1", "2"),
        values=("1", "2", "5"),
        evaluation_point="3",
    )
    result = compute_newton_evaluate(request)
    assert result.result == "10"


def test_newton_evaluate_at_node() -> None:
    request = NewtonEvaluateRequest(
        nodes=("0", "1", "2"),
        values=("1", "2", "5"),
        evaluation_point="1",
    )
    result = compute_newton_evaluate(request)
    assert result.result == "2"


# --- Issue 1: require pairwise-distinct nodes ---


def test_divided_differences_rejects_repeated_nodes() -> None:
    with pytest.raises(ValidationError):
        DividedDifferencesRequest(nodes=("0", "1", "1"), values=("1", "2", "5"))


def test_newton_form_rejects_repeated_nodes() -> None:
    with pytest.raises(ValidationError):
        NewtonFormRequest(nodes=("0", "1", "1"), values=("1", "2", "5"))


def test_newton_evaluate_rejects_repeated_nodes() -> None:
    with pytest.raises(ValidationError):
        NewtonEvaluateRequest(
            nodes=("0", "1", "1"), values=("1", "2", "5"), evaluation_point="3"
        )


# --- Issue 2: require len(nodes) == len(values) ---


def test_newton_form_requires_equal_length() -> None:
    with pytest.raises(ValidationError):
        NewtonFormRequest(nodes=("0", "1", "2"), values=("1", "2"))


def test_newton_evaluate_requires_equal_length() -> None:
    with pytest.raises(ValidationError):
        NewtonEvaluateRequest(
            nodes=("0", "1", "2"), values=("1", "2"), evaluation_point="3"
        )


# --- Issue 3: consistent scalar domain (all operations use exact rationals) ---


def test_divided_differences_with_rationals() -> None:
    request = DividedDifferencesRequest(
        nodes=("0", "1/2", "1"), values=("1", "3/2", "2")
    )
    result = compute_divided_differences(request)
    assert result.coefficients == ("1", "1", "0")


def test_newton_form_with_rationals() -> None:
    request = NewtonFormRequest(nodes=("0", "1/2", "1"), values=("1", "3/2", "2"))
    result = compute_newton_form(request)
    assert result.coefficients == ("1", "1", "0")


def test_newton_evaluate_with_rationals() -> None:
    request = NewtonEvaluateRequest(
        nodes=("0", "1/2", "1"),
        values=("1", "3/2", "2"),
        evaluation_point="3/4",
    )
    result = compute_newton_evaluate(request)
    assert result.result == "7/4"


# --- Interpolation correctness ---


def test_interpolation_passes_through_sample_points() -> None:
    """Newton evaluation at every node must return the corresponding value."""
    nodes = ("0", "1", "2", "3")
    values = ("1", "2", "5", "10")
    for i, (n, v) in enumerate(zip(nodes, values, strict=True)):
        request = NewtonEvaluateRequest(nodes=nodes, values=values, evaluation_point=n)
        result = compute_newton_evaluate(request)
        assert result.result == v, (
            f"interpolation mismatch at node {i}: expected {v}, got {result.result}"
        )
