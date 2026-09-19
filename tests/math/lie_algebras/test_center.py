"""Tests for the exact Lie-algebra center."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieAlgebraRequest,
    LieCenterResult,
)
from jacobian.math.lie_algebras.operations import (
    lie_bracket,
    lie_center,
)


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


def _rows(result: LieCenterResult) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.center.generators.entries
    )


def _element(basis: tuple[str, ...], coords: tuple[Fraction, ...]) -> LieAlgebraElement:
    from jacobian._exact import CanonicalRational

    return LieAlgebraElement.model_construct(
        basis=basis,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in coords),
    )


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
ABELIAN_3 = _algebra(("a", "b", "c"), ())
GL2 = _algebra(
    ("a", "b", "c", "d"),
    (
        (0, 1, 1, 1),
        (0, 2, 2, -1),
        (1, 2, 0, 1),
        (1, 2, 3, -1),
        (1, 3, 1, 1),
        (2, 3, 2, -1),
    ),
)
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))
LINE = _algebra(("t",), ())
JACOBI_VIOLATOR = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, 1), (1, 2, 0, 1)))


class TestCenterKnownAnswers:
    def test_sl2_center_is_zero(self) -> None:
        result = lie_center(SL2)
        assert _rows(result) == ()
        assert result.center.generators.row_count == 0
        assert result.center.generators.column_count == 3

    def test_heisenberg_center_is_span_of_z(self) -> None:
        assert _rows(lie_center(HEISENBERG)) == (
            (Fraction(0), Fraction(0), Fraction(1)),
        )

    def test_abelian_center_is_whole_space(self) -> None:
        assert _rows(lie_center(ABELIAN_3)) == (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )

    def test_gl2_center_is_scalars(self) -> None:
        """gl2's center is the non-basis-aligned scalar line."""
        assert _rows(lie_center(GL2)) == (
            (Fraction(1), Fraction(0), Fraction(0), Fraction(1)),
        )

    def test_affine_center_is_zero(self) -> None:
        assert _rows(lie_center(AFFINE)) == ()

    def test_line_center_is_whole_space(self) -> None:
        assert _rows(lie_center(LINE)) == ((Fraction(1),),)


class TestCenterDefiningInvariant:
    @pytest.mark.parametrize("algebra", [SL2, HEISENBERG, ABELIAN_3, GL2, AFFINE, LINE])
    def test_center_rows_bracket_to_zero(
        self, algebra: FiniteDimensionalLieAlgebra
    ) -> None:
        """Every retained generator brackets to zero against every basis
        element: independent replay through the bracket operation."""
        result = lie_center(algebra)
        dimension = len(algebra.basis)
        for row in _rows(result):
            element = _element(algebra.basis, row)
            for index in range(dimension):
                target = _element(
                    algebra.basis,
                    tuple(Fraction(int(i == index)) for i in range(dimension)),
                )
                bracket = lie_bracket(algebra, element, target).bracket
                assert all(
                    coordinate.as_fraction() == 0 for coordinate in bracket.coordinates
                )

    def test_center_binds_source_algebra(self) -> None:
        result = lie_center(GL2)
        assert result.center.basis == GL2.basis == ("a", "b", "c", "d")


class TestCenterRejections:
    def test_jacobi_violator_rejected_before_solve(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_center(JACOBI_VIOLATOR)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_forged_non_rref_generators_rejected(self) -> None:
        import json

        result = lie_center(HEISENBERG)
        forged = json.loads(result.model_dump_json())
        forged["center"]["generators"]["entries"][0][0] = {"num": "2", "den": "1"}
        with pytest.raises(ValidationError) as exc_info:
            LieCenterResult.model_validate_json(json.dumps(forged))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "lie_algebra.subspace_leading_one"
        )


class TestCenterComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = LieAlgebraRequest(algebra=HEISENBERG)
        assert lie_center(request.algebra) == lie_center(HEISENBERG)

    def test_serialized_result_round_trips(self) -> None:
        result = lie_center(GL2)
        assert LieCenterResult.model_validate_json(result.model_dump_json()) == result

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.center.compute"
        }
        assert set(tools) == {"lie_algebra.center.compute"}
        tool = tools["lie_algebra.center.compute"]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        assert tool.run(payload) == lie_center(payload.algebra)
