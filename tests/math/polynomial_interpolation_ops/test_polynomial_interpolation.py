"""Tests for polynomial interpolation operations."""

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
    request = DividedDifferencesRequest(
        nodes=("0", "1", "2"), values=("1", "2", "5")
    )
    result = compute_divided_differences(request)
    assert result.coefficients == ("1", "1", "1")


def test_newton_form_basic() -> None:
    request = NewtonFormRequest(
        nodes=("0", "1", "2"), values=("1", "2", "5")
    )
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
