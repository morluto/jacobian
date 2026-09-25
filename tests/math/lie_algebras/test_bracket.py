"""Tests for the exact finite-dimensional Lie bracket."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    MAX_ELEMENT_COEFFICIENT_DIGITS,
    MAX_LIE_DIMENSION,
    MAX_STRUCTURE_COEFFICIENT_DIGITS,
    MAX_STRUCTURE_NONZEROS,
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieAlgebraStructureConstant,
    LieBracketRequest,
    StructureConstant,
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
    def test_canonical_algebra_constructor_admits_jacobi_and_round_trips(self) -> None:
        restored = FiniteDimensionalLieAlgebra.model_validate_json(
            SL2.model_dump_json()
        )
        uncached = FiniteDimensionalLieAlgebra.model_construct(
            basis=SL2.basis, structure_constants=SL2.structure_constants
        )

        assert restored == SL2
        assert restored.model_dump(mode="json") == SL2.model_dump(mode="json")
        assert uncached == SL2
        assert hash(uncached) == hash(SL2)
        assert "_jacobi_admitted" not in restored.model_dump()

    def test_model_copy_cannot_retain_admission_after_bracket_mutation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SL2.model_copy(
                update={
                    "structure_constants": (
                        LieAlgebraStructureConstant.model_construct(
                            i=0,
                            j=1,
                            k=2,
                            coefficient=SL2.structure_constants[0].coefficient,
                        ),
                        LieAlgebraStructureConstant.model_construct(
                            i=0,
                            j=2,
                            k=0,
                            coefficient=SL2.structure_constants[
                                0
                            ].coefficient.from_integer_ratio(-2, 1),
                        ),
                        LieAlgebraStructureConstant.model_construct(
                            i=1,
                            j=2,
                            k=1,
                            coefficient=SL2.structure_constants[
                                0
                            ].coefficient.from_integer_ratio(-2, 1),
                        ),
                    )
                }
            )
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_nested_json_request_rejects_non_lie_structure_constants(self) -> None:
        payload = {
            "algebra": {
                "basis": list(SL2_BASIS),
                "structure_constants": [
                    {
                        "i": i,
                        "j": j,
                        "k": k,
                        "coefficient": {"num": str(value), "den": "1"},
                    }
                    for i, j, k, value in (
                        (0, 1, 2, 1),
                        (0, 2, 0, -2),
                        (1, 2, 1, -2),
                    )
                ],
            },
            "left": {
                "basis": list(SL2_BASIS),
                "coordinates": [{"num": "0", "den": "1"}] * len(SL2_BASIS),
            },
            "right": {
                "basis": list(SL2_BASIS),
                "coordinates": [{"num": "0", "den": "1"}] * len(SL2_BASIS),
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            LieBracketRequest.model_validate_json(json.dumps(payload))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_canonical_algebra_constructor_rejects_jacobi_violation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _algebra(
                SL2_BASIS,
                ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, -2)),
            )
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

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
    def test_boundary_coordinate_and_structure_heights_are_admitted(self) -> None:
        scale = (
            10 ** min(MAX_ELEMENT_COEFFICIENT_DIGITS, MAX_STRUCTURE_COEFFICIENT_DIGITS)
            - 1
        )
        algebra = _algebra(("x", "y"), ((0, 1, 1, scale),))
        result = lie_bracket(
            algebra,
            _element(("x", "y"), (scale, 0)),
            _element(("x", "y"), (0, scale)),
        )

        assert _coords(result.bracket) == (Fraction(0), Fraction(scale**3))
        assert result.ledger[0].pair_coefficient.as_fraction() == Fraction(scale**2)
        assert result.ledger[0].terms[0].coefficient.as_fraction() == Fraction(scale**3)

    def test_over_bound_element_coefficient_is_rejected_before_expansion(self) -> None:
        over_bound = 10**MAX_ELEMENT_COEFFICIENT_DIGITS
        with pytest.raises(OperationDomainValidationError, match="64-digit bound"):
            lie_bracket(
                AXB,
                _element(AXB_BASIS, (over_bound, 0)),
                _element(AXB_BASIS, (0, 1)),
            )

    def test_over_bound_structure_coefficient_is_rejected_before_jacobi(self) -> None:
        over_bound = 10**MAX_ELEMENT_COEFFICIENT_DIGITS
        with pytest.raises(ValidationError, match="structure constants must use"):
            _algebra(AXB_BASIS, ((0, 1, 1, over_bound),))

    def test_maximum_structure_coefficient_height_is_accepted(self) -> None:
        coefficient = 10**MAX_STRUCTURE_COEFFICIENT_DIGITS - 1
        algebra = _algebra(AXB_BASIS, ((0, 1, 1, coefficient),))

        assert algebra.structure_constants[0].coefficient.as_fraction() == coefficient

    def test_structure_coefficient_digit_bound_is_published_in_schema(self) -> None:
        schema = FiniteDimensionalLieAlgebra.model_json_schema()
        coefficient = schema["$defs"]["LieAlgebraStructureConstant"]["properties"][
            "coefficient"
        ]

        assert coefficient["properties"]["num"]["maxLength"] == (
            MAX_STRUCTURE_COEFFICIENT_DIGITS + 1
        )
        assert coefficient["properties"]["den"]["maxLength"] == (
            MAX_STRUCTURE_COEFFICIENT_DIGITS
        )

    def test_bracket_ledger_constant_keeps_full_canonical_rational_capacity(
        self,
    ) -> None:
        coefficient = 10 ** (MAX_STRUCTURE_COEFFICIENT_DIGITS + 1)
        term = StructureConstant.model_validate(
            {
                "i": 0,
                "j": 1,
                "k": 1,
                "coefficient": {"num": coefficient, "den": 1},
            }
        )

        assert term.coefficient.as_fraction() == coefficient

    def test_over_bound_dimension_is_rejected_before_jacobi(self) -> None:
        basis = tuple(f"e{index}" for index in range(MAX_LIE_DIMENSION + 1))
        algebra = FiniteDimensionalLieAlgebra.model_construct(
            basis=basis,
            structure_constants=(),
        )
        zero = _element(AXB_BASIS, (0, 0)).coordinates[0]
        element = LieAlgebraElement.model_construct(
            basis=basis,
            coordinates=(zero,) * len(basis),
        )

        with pytest.raises(OperationResourceAdmissionError, match="dimension"):
            lie_bracket(algebra, element, element)

    def test_dimension_limit_is_enforced_at_value_construction(self) -> None:
        basis = tuple(f"e{index}" for index in range(MAX_LIE_DIMENSION))
        value = FiniteDimensionalLieAlgebra.model_validate(
            {"basis": list(basis), "structure_constants": []}
        )
        assert len(value.basis) == MAX_LIE_DIMENSION
        with pytest.raises(ValidationError):
            FiniteDimensionalLieAlgebra.model_validate(
                {"basis": [*basis, "e8"], "structure_constants": []}
            )

    def test_over_bound_structure_table_is_rejected_before_jacobi(self) -> None:
        constant = StructureConstant.model_construct(
            i=0,
            j=1,
            k=0,
            coefficient=AXB.structure_constants[0].coefficient,
        )
        algebra = FiniteDimensionalLieAlgebra.model_construct(
            basis=("x", "y"),
            structure_constants=(constant,) * (MAX_STRUCTURE_NONZEROS + 1),
        )
        element = _element(("x", "y"), (1, 0))

        with pytest.raises(OperationResourceAdmissionError, match="structure-constant"):
            lie_bracket(algebra, element, element)

    def test_complete_ledger_keeps_every_nonzero_basis_pair(self) -> None:
        basis = tuple(f"e{index}" for index in range(7))
        constants = tuple(
            (first, second, 6, 1)
            for first in range(6)
            for second in range(first + 1, 6)
        )
        algebra = _algebra(basis, constants)
        result = lie_bracket(
            algebra,
            _element(basis, (1, 2, 3, 4, 5, 6, 7)),
            _element(basis, (7, 6, 5, 4, 3, 2, 1)),
        )

        assert [(row.i, row.j) for row in result.ledger] == [
            (first, second) for first in range(6) for second in range(first + 1, 6)
        ]
        assert all(len(row.terms) == 1 for row in result.ledger)
        replayed = [Fraction(0)] * len(basis)
        for row in result.ledger:
            for term in row.terms:
                replayed[term.k] += term.coefficient.as_fraction()
        assert tuple(replayed) == _coords(result.bracket)

    def test_jacobi_violating_table_rejected(self) -> None:
        bad_constants = tuple(
            StructureConstant.model_construct(
                i=i,
                j=j,
                k=k,
                coefficient=SL2.structure_constants[0].coefficient.from_integer_ratio(
                    coefficient, 1
                ),
            )
            for i, j, k, coefficient in (
                (0, 1, 2, 1),
                (0, 2, 0, -2),
                (1, 2, 1, -2),
            )
        )
        bad = FiniteDimensionalLieAlgebra.model_construct(
            basis=SL2_BASIS,
            structure_constants=bad_constants,
            _jacobi_admitted=True,
        )
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
