"""Tests for the native Lie-algebra adjoint projection."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.lie_algebras import lie_adjoint, lie_adjoint_matrices
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
)
from jacobian.math.lie_algebras.operations import (
    lie_adjoint as native_lie_adjoint,
)
from jacobian.math.lie_algebras.operations import (
    lie_adjoint_representation,
    lie_bracket,
)
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

    def test_catalog_result_retains_source_and_exact_representation(self) -> None:
        result = lie_adjoint_representation(SL2)
        matrices = tuple(_entries(matrix) for matrix in result.matrices)
        assert result.algebra == SL2

        def multiply(
            left: tuple[tuple[Fraction, ...], ...],
            right: tuple[tuple[Fraction, ...], ...],
        ) -> tuple[tuple[Fraction, ...], ...]:
            return tuple(
                tuple(
                    sum(left[row][inner] * right[inner][column] for inner in range(3))
                    for column in range(3)
                )
                for row in range(3)
            )

        # Independently replay [ad(e), ad(f)] = ad([e, f]) = ad(h).
        commutator = multiply(matrices[0], matrices[1])
        reverse = multiply(matrices[1], matrices[0])
        assert (
            tuple(
                tuple(
                    commutator[row][column] - reverse[row][column]
                    for column in range(3)
                )
                for row in range(3)
            )
            == matrices[2]
        )

    def test_catalog_declares_and_runs_the_adjoint_operation(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        matches = tuple(
            tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.adjoint_representation.compute"
        )
        assert len(matches) == 1
        tool = matches[0]
        assert tool.examples
        request = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        result = tool.run(request)
        assert isinstance(result, tool.result_type)
        assert result.algebra == request.algebra
        assert len(result.matrices) == len(request.algebra.basis)
        assert tool.result_type.model_validate_json(result.model_dump_json()) == result

    def test_adjoint_result_rejects_wrong_matrix_count(self) -> None:
        from pydantic import ValidationError

        from jacobian.math.lie_algebras._models import LieAdjointRepresentationResult

        with pytest.raises(ValidationError):
            LieAdjointRepresentationResult(
                algebra=SL2,
                matrices=(
                    RationalMatrix.model_validate(
                        {
                            "domain": "QQ",
                            "row_count": 3,
                            "column_count": 3,
                            "entries": [[{"num": 0, "den": 1}] * 3 for _ in range(3)],
                        }
                    ),
                ),
            )


class TestSingleElementAdjoint:
    def _heisenberg(self) -> FiniteDimensionalLieAlgebra:
        return _algebra(("x", "y", "z"), ((0, 1, 2, 1),))

    def _element(self, coordinates: tuple[int, int, int]) -> LieAlgebraElement:
        return LieAlgebraElement.model_validate(
            {
                "basis": ["x", "y", "z"],
                "coordinates": [
                    {"num": coordinate, "den": 1} for coordinate in coordinates
                ],
            }
        )

    def test_adjoint_of_linear_combination_matches_independent_bracket_oracle(
        self,
    ) -> None:
        # In [x,y]=z, ad_(x+2y)(a*x+b*y+c*z) = (b-2a)z.
        result = lie_adjoint(self._heisenberg(), self._element((1, 2, 0)))
        assert result.algebra.basis == ("x", "y", "z")
        assert result.element.coordinates == self._element((1, 2, 0)).coordinates
        assert _entries(result.matrix) == (
            (ZERO, ZERO, ZERO),
            (ZERO, ZERO, ZERO),
            (Fraction(-2), Fraction(1), ZERO),
        )
        for column, vector in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            basis_element = self._element(vector)
            bracket = lie_bracket(
                self._heisenberg(), result.element, basis_element
            ).bracket
            assert tuple(row[column] for row in _entries(result.matrix)) == tuple(
                entry.as_fraction() for entry in bracket.coordinates
            )

    def test_single_adjoint_matches_linear_combination_of_basis_representation(
        self,
    ) -> None:
        algebra = self._heisenberg()
        element = self._element((3, -2, 5))
        result = native_lie_adjoint(algebra, element)
        basis_matrices = lie_adjoint_matrices(algebra)
        expected = tuple(
            tuple(
                sum(
                    element.coordinates[index].as_fraction()
                    * _entries(basis_matrices[index])[row][column]
                    for index in range(3)
                )
                for column in range(3)
            )
            for row in range(3)
        )
        assert _entries(result.matrix) == expected

    def test_jacobi_and_parent_requirements_are_enforced(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_adjoint(JACOBI_VIOLATOR, self._element((1, 0, 0)))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"
        wrong_axis = self._element((1, 0, 0)).model_copy(
            update={"basis": ("a", "b", "c")}
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_adjoint(self._heisenberg(), wrong_axis)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.adjoint_element_basis"

    def test_over_bound_element_is_rejected_before_matrix_expansion(self) -> None:
        coordinate_type = type(self._element((1, 0, 0)).coordinates[0])
        too_large = self._element((1, 0, 0)).model_copy(
            update={
                "coordinates": (
                    coordinate_type(num=10**64, den=1),
                    *self._element((1, 0, 0)).coordinates[1:],
                )
            }
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_adjoint(self._heisenberg(), too_large)
        assert "admission" in exc_info.value.errors()[0]["type"]

    def test_public_tool_round_trips_exact_result(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        (tool,) = (
            item for item in TOOLS if item.operation_id == "lie_algebra.adjoint.compute"
        )
        request = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        result = tool.run(request)
        assert result == lie_adjoint(request.algebra, request.element)
        assert tool.result_type.model_validate_json(result.model_dump_json()) == result
