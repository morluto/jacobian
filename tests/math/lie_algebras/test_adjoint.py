"""Tests for the native Lie-algebra adjoint projection."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras import lie_adjoint_matrices
from jacobian.math.lie_algebras._models import FiniteDimensionalLieAlgebra
from jacobian.math.lie_algebras.operations import lie_bracket
from jacobian.math.matrices.values import RationalMatrix


def _algebra(
    basis: tuple[str, ...],
    constants: tuple[tuple[int, int, int, int], ...],
) -> FiniteDimensionalLieAlgebra:
    return FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": list(basis),
            "structure_constants": [
                {"i": i, "j": j, "k": k, "coefficient": {"num": value, "den": 1}}
                for i, j, k, value in constants
            ],
        }
    )


def _entries(matrix: RationalMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(entry.as_fraction() for entry in row) for row in matrix.entries)


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
JACOBI_VIOLATOR = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, 1), (1, 2, 0, 1)))

ZERO = Fraction(0)


class TestAdjointMatrices:
    def test_sl2_adjoint_action(self) -> None:
        ad_e, ad_f, ad_h = (_entries(matrix) for matrix in lie_adjoint_matrices(SL2))
        assert ad_e == (
            (ZERO, ZERO, Fraction(-2)),
            (ZERO, ZERO, ZERO),
            (ZERO, Fraction(1), ZERO),
        )
        assert ad_f == (
            (ZERO, ZERO, ZERO),
            (ZERO, ZERO, Fraction(2)),
            (Fraction(-1), ZERO, ZERO),
        )
        assert ad_h == (
            (Fraction(2), ZERO, ZERO),
            (ZERO, Fraction(-2), ZERO),
            (ZERO, ZERO, ZERO),
        )

    def test_adjoint_columns_replay_brackets(self) -> None:
        """Column j of ad_i is [b_i, b_j]: composition with the bracket op."""
        from jacobian.math.lie_algebras._models import LieAlgebraElement

        matrices = lie_adjoint_matrices(SL2)
        basis = ("e", "f", "h")
        for i in range(3):
            for j in range(3):
                left = LieAlgebraElement.model_validate(
                    {
                        "basis": list(basis),
                        "coordinates": [
                            {"num": int(k == i), "den": 1} for k in range(3)
                        ],
                    }
                )
                right = LieAlgebraElement.model_validate(
                    {
                        "basis": list(basis),
                        "coordinates": [
                            {"num": int(k == j), "den": 1} for k in range(3)
                        ],
                    }
                )
                expected = tuple(
                    coordinate.as_fraction()
                    for coordinate in lie_bracket(SL2, left, right).bracket.coordinates
                )
                assert tuple(row[j] for row in _entries(matrices[i])) == expected

    def test_jacobi_violator_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_adjoint_matrices(JACOBI_VIOLATOR)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_public_api_exports_adjoint(self) -> None:
        from jacobian.math import lie_algebras

        assert "lie_adjoint_matrices" in lie_algebras.__all__
