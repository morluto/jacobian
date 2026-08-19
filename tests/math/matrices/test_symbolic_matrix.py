"""Domain tests for exact symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

import pytest

from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_characteristic_polynomial,
    compute_symbolic_determinant,
    compute_symbolic_eigenvalues,
    compute_symbolic_rank,
)


def _request(
    entries: list[list[str]], variables: list[str] | None = None
) -> SymbolicMatrixRequest:
    if variables is None:
        variables = []
    return SymbolicMatrixRequest.model_validate(
        {"matrix": {"variables": variables, "entries": entries}}
    )


def test_symbolic_determinant_of_two_by_two() -> None:
    request = _request([["a", "c"], ["b", "d"]], ["a", "b", "c", "d"])
    result = compute_symbolic_determinant(request)
    assert isinstance(result, SymbolicDeterminantResult)
    assert result.determinant == "a*d - b*c"


def test_symbolic_determinant_of_constant_matrix() -> None:
    request = _request([["1", "2"], ["3", "4"]], [])
    result = compute_symbolic_determinant(request)
    # SymPy may render integer results as "10" or "-2" etc.
    assert result.determinant in ("-2", "(-2)")


def test_symbolic_rank_of_full_matrix() -> None:
    request = _request([["a", "c"], ["b", "d"]], ["a", "b", "c", "d"])
    result = compute_symbolic_rank(request)
    assert isinstance(result, SymbolicRankResult)
    assert result.rank == 2
    assert result.pivot_columns == (0, 1)


def test_symbolic_rank_of_singular_matrix() -> None:
    # Rows are multiples over a rational function field
    request = _request([["a", "a"], ["a", "a"]], ["a"])
    result = compute_symbolic_rank(request)
    assert result.rank == 1


def test_symbolic_characteristic_polynomial_of_constant_matrix() -> None:
    request = _request([["1", "2"], ["3", "4"]], [])
    result = compute_symbolic_characteristic_polynomial(request)
    assert isinstance(result, SymbolicCharacteristicPolynomialResult)
    assert result.degree == 2
    assert result.coefficients_descending == ("1", "-5", "-2")


def test_symbolic_characteristic_polynomial_of_zero_matrix() -> None:
    request = _request([["0", "0"], ["0", "0"]], [])
    result = compute_symbolic_characteristic_polynomial(request)
    assert result.coefficients_descending == ("1", "0", "0")


def test_symbolic_eigenvalues_of_constant_matrix() -> None:
    request = _request([["1", "2"], ["3", "4"]], [])
    result = compute_symbolic_eigenvalues(request)
    assert isinstance(result, SymbolicEigenvaluesResult)
    # eigenvalues are (5 - sqrt(33))/2 and (5 + sqrt(33)/2, each with multiplicity 1
    assert len(result.eigenvalues) == 2
    assert result.multiplicities == (1, 1)


def test_symbolic_eigenvalues_of_quadratic_algebraic_entries() -> None:
    # A = [[1/4, sqrt(3)/4], [sqrt(3)/4, 3/4]] -> eigenvalues 0 and 1
    request = _request([["1/4", "sqrt(3)/4"], ["sqrt(3)/4", "3/4"]], [])
    result = compute_symbolic_eigenvalues(request)
    assert sorted(result.eigenvalues) == ["0", "1"]
    assert result.multiplicities == (1, 1)


def test_symbolic_determinant_requires_square_matrix() -> None:
    request = _request([["a", "b", "c"]], ["a", "b", "c"])
    with pytest.raises(ValueError, match="square"):
        compute_symbolic_determinant(request)


def test_symbolic_eigenvalues_requires_square_matrix() -> None:
    request = _request([["a", "b", "c"]], ["a", "b", "c"])
    with pytest.raises(ValueError, match="square"):
        compute_symbolic_eigenvalues(request)


def test_symbolic_matrix_rejects_non_rectangular() -> None:
    with pytest.raises(ValueError, match="same length"):
        SymbolicMatrixRequest.model_validate(
            {"matrix": {"variables": ["a"], "entries": [["a", "b"], ["a"]]}}
        )


def test_symbolic_matrix_rejects_oversized() -> None:
    with pytest.raises(ValueError, match="32"):
        SymbolicMatrixRequest.model_validate(
            {
                "matrix": {
                    "variables": ["a"],
                    "entries": [["a"] * 33],
                }
            }
        )


def test_symbolic_eigenvalues_returns_polynomial_for_unrepresentable() -> None:
    """Parameterized quintic companion matrix returns ROOTS_BY_POLYNOMIAL."""
    request = SymbolicMatrixRequest.model_validate(
        {
            "matrix": {
                "variables": ["a"],
                "entries": [
                    ["0", "0", "0", "0", "a"],
                    ["1", "0", "0", "0", "1"],
                    ["0", "1", "0", "0", "0"],
                    ["0", "0", "1", "0", "0"],
                    ["0", "0", "0", "1", "0"],
                ],
            }
        }
    )
    result = compute_symbolic_eigenvalues(request)
    assert result.representation == "ROOTS_BY_POLYNOMIAL"
    assert result.degree == 5
    assert result.characteristic_polynomial is not None
    assert result.eigenvalues is None


def test_symbolic_eigenvalues_explicit_for_representable() -> None:
    """Regular 2x2 matrix returns EXPLICIT_ROOTS."""
    request = _request([["1", "2"], ["3", "4"]], [])
    result = compute_symbolic_eigenvalues(request)
    assert result.representation == "EXPLICIT_ROOTS"
    assert result.eigenvalues is not None
    assert result.characteristic_polynomial is None
