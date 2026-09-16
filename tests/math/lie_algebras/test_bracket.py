"""Tests for the exact finite-dimensional Lie bracket."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieBracketRequest,
)
from jacobian.math.lie_algebras.operations import lie_bracket


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


def _element(basis: tuple[str, ...], coords: tuple[int, ...]) -> LieAlgebraElement:
    return LieAlgebraElement.model_validate(
        {
            "basis": list(basis),
            "coordinates": [{"num": value, "den": 1} for value in coords],
        }
    )


def _coords(element: LieAlgebraElement) -> tuple[Fraction, ...]:
    return tuple(coordinate.as_fraction() for coordinate in element.coordinates)


SL2_BASIS = ("e", "f", "h")
SL2 = _algebra(SL2_BASIS, ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))

HEISENBERG_BASIS = ("x", "y", "z")
HEISENBERG = _algebra(HEISENBERG_BASIS, ((0, 1, 2, 1),))

AXB_BASIS = ("a", "b")
AXB = _algebra(AXB_BASIS, ((0, 1, 1, 1),))


class TestBracketKnownAnswers:
    def test_sl2_bracket_e_f(self) -> None:
        result = lie_bracket(
            SL2, _element(SL2_BASIS, (1, 0, 0)), _element(SL2_BASIS, (0, 1, 0))
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(0), Fraction(1))
        assert [(row.i, row.j) for row in result.ledger] == [(0, 1)]

    def test_sl2_bracket_h_e(self) -> None:
        result = lie_bracket(
            SL2, _element(SL2_BASIS, (0, 0, 1)), _element(SL2_BASIS, (1, 0, 0))
        )
        assert _coords(result.bracket) == (Fraction(2), Fraction(0), Fraction(0))

    def test_sl2_bracket_h_f(self) -> None:
        result = lie_bracket(
            SL2, _element(SL2_BASIS, (0, 0, 1)), _element(SL2_BASIS, (0, 1, 0))
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(-2), Fraction(0))

    def test_heisenberg_bracket(self) -> None:
        result = lie_bracket(
            HEISENBERG,
            _element(HEISENBERG_BASIS, (1, 0, 0)),
            _element(HEISENBERG_BASIS, (0, 1, 0)),
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(0), Fraction(1))

    def test_heisenberg_center_bracket_vanishes(self) -> None:
        result = lie_bracket(
            HEISENBERG,
            _element(HEISENBERG_BASIS, (0, 0, 1)),
            _element(HEISENBERG_BASIS, (1, 2, 3)),
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(0), Fraction(0))
        assert result.ledger == ()

    def test_affine_bracket(self) -> None:
        result = lie_bracket(
            AXB, _element(AXB_BASIS, (1, 0)), _element(AXB_BASIS, (0, 1))
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(1))

    def test_abelian_bracket_vanishes(self) -> None:
        abelian = _algebra(("u", "v"), ())
        result = lie_bracket(
            abelian, _element(("u", "v"), (3, -1)), _element(("u", "v"), (2, 5))
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(0))

    def test_zero_element_bracket_vanishes(self) -> None:
        result = lie_bracket(
            SL2, _element(SL2_BASIS, (0, 0, 0)), _element(SL2_BASIS, (1, 2, 3))
        )
        assert _coords(result.bracket) == (Fraction(0), Fraction(0), Fraction(0))


class TestBracketInvariants:
    @pytest.mark.parametrize(
        "algebra_basis",
        (
            (SL2, SL2_BASIS),
            (HEISENBERG, HEISENBERG_BASIS),
            (AXB, AXB_BASIS),
        ),
    )
    def test_antisymmetry(
        self, algebra_basis: tuple[FiniteDimensionalLieAlgebra, tuple[str, ...]]
    ) -> None:
        algebra, basis = algebra_basis
        dimension = len(basis)
        left = _element(basis, tuple(range(1, dimension + 1)))
        right = _element(basis, tuple(range(dimension, 0, -1)))
        forward = _coords(lie_bracket(algebra, left, right).bracket)
        backward = _coords(lie_bracket(algebra, right, left).bracket)
        assert forward == tuple(-value for value in backward)

    @pytest.mark.parametrize(
        "algebra_basis",
        (
            (SL2, SL2_BASIS),
            (HEISENBERG, HEISENBERG_BASIS),
            (AXB, AXB_BASIS),
        ),
    )
    def test_bilinearity(
        self, algebra_basis: tuple[FiniteDimensionalLieAlgebra, tuple[str, ...]]
    ) -> None:
        from jacobian.math.lie_algebras.operations import _bracket_table

        algebra, basis = algebra_basis
        dimension = len(basis)
        left = _element(basis, tuple(range(1, dimension + 1)))
        right = _element(basis, tuple(range(dimension, 0, -1)))
        other = _element(basis, tuple(-index for index in range(dimension)))
        assert _bracket_table(algebra) is not None
        combined_coords = tuple(
            2 * value + other_value
            for value, other_value in zip(_coords(left), _coords(other), strict=True)
        )
        combined = LieAlgebraElement.model_validate(
            {
                "basis": list(basis),
                "coordinates": [
                    {"num": value.numerator, "den": value.denominator}
                    for value in combined_coords
                ],
            }
        )
        direct = _coords(lie_bracket(algebra, combined, right).bracket)
        replay = tuple(
            2 * value + other_value
            for value, other_value in zip(
                _coords(lie_bracket(algebra, left, right).bracket),
                _coords(lie_bracket(algebra, other, right).bracket),
                strict=True,
            )
        )
        assert direct == replay

    @pytest.mark.parametrize(
        "algebra_basis",
        (
            (SL2, SL2_BASIS),
            (HEISENBERG, HEISENBERG_BASIS),
            (AXB, AXB_BASIS),
        ),
    )
    def test_jacobi_on_elements(
        self, algebra_basis: tuple[FiniteDimensionalLieAlgebra, tuple[str, ...]]
    ) -> None:
        algebra, basis = algebra_basis
        dimension = len(basis)
        vectors = [
            _element(basis, tuple(index + offset for index in range(dimension)))
            for offset in (1, 2, 3)
        ]
        totals = [Fraction(0)] * dimension
        for first, second, third in (
            (vectors[0], vectors[1], vectors[2]),
            (vectors[1], vectors[2], vectors[0]),
            (vectors[2], vectors[0], vectors[1]),
        ):
            inner = lie_bracket(algebra, first, second).bracket
            outer = _coords(lie_bracket(algebra, inner, third).bracket)
            totals = [total + value for total, value in zip(totals, outer, strict=True)]
        assert tuple(totals) == tuple(Fraction(0) for _ in range(dimension))

    def test_ledger_replays_bracket(self) -> None:
        left = _element(SL2_BASIS, (1, 2, 3))
        right = _element(SL2_BASIS, (3, -1, 2))
        result = lie_bracket(SL2, left, right)
        totals = [Fraction(0)] * 3
        for row in result.ledger:
            assert row.pair_coefficient.as_fraction() == (
                left.coordinates[row.i].as_fraction()
                * right.coordinates[row.j].as_fraction()
                - left.coordinates[row.j].as_fraction()
                * right.coordinates[row.i].as_fraction()
            )
            for term in row.terms:
                totals[term.k] += term.coefficient.as_fraction()
        assert tuple(totals) == _coords(result.bracket)


class TestBracketAdmission:
    def test_jacobi_violating_table_rejected(self) -> None:
        bad = _algebra(SL2_BASIS, ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, -2)))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_bracket(
                bad, _element(SL2_BASIS, (1, 0, 0)), _element(SL2_BASIS, (0, 1, 0))
            )
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_unordered_constant_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _algebra(SL2_BASIS, ((1, 0, 2, 1),))

    def test_basis_mismatch_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            lie_bracket(
                SL2,
                _element(("e", "f", "q"), (1, 0, 0)),
                _element(SL2_BASIS, (0, 1, 0)),
            )

    def test_request_model_rejects_basis_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            LieBracketRequest(
                algebra=SL2,
                left=_element(("e", "f", "q"), (1, 0, 0)),
                right=_element(SL2_BASIS, (0, 1, 0)),
            )

    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.lie_algebras._tools import TOOLS

        tool = next(
            tool for tool in TOOLS if tool.operation_id == "lie_algebra.bracket.compute"
        )
        request = LieBracketRequest(
            algebra=SL2,
            left=_element(SL2_BASIS, (1, 0, 0)),
            right=_element(SL2_BASIS, (0, 1, 0)),
        )
        assert tool.run(request) == lie_bracket(
            request.algebra, request.left, request.right
        )

    def test_published_example_validates(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        tool = next(
            tool for tool in TOOLS if tool.operation_id == "lie_algebra.bracket.compute"
        )
        example = tool.examples[0]
        request = tool.request_type.model_validate_json(
            encode_strict_json(example.input), strict=True
        )
        assert _coords(tool.run(request).bracket) == (
            Fraction(0),
            Fraction(0),
            Fraction(1),
        )
