"""Exact characteristic-zero solvable radicals as source-bound Lie ideals."""

import json
from fractions import Fraction

from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieIdeal,
    LieKillingRadicalResult,
)
from jacobian.math.lie_algebras._tools import TOOLS
from jacobian.math.lie_algebras.operations import (
    check_ideal,
    lie_killing_form_radical,
    lie_quotient,
    lie_solvable_radical,
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


SL2 = _algebra(
    ("e", "f", "h"),
    ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)),
)
AFFINE = _algebra(("h", "e"), ((0, 1, 1, 1),))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
SL2_PLUS_ABELIAN_FIVE = _algebra(
    ("e", "f", "h", "a", "b", "c", "d", "z"),
    ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)),
)


def _rows(ideal: LieIdeal) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in ideal.generators.entries
    )


def test_solvable_radical_distinguishes_affine_algebra_from_killing_radical() -> None:
    radical = lie_solvable_radical(AFFINE)
    killing_radical: LieKillingRadicalResult = lie_killing_form_radical(AFFINE)

    assert _rows(radical) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    assert _rows(killing_radical.radical) == ((Fraction(0), Fraction(1)),)
    assert check_ideal(AFFINE, radical).is_ideal
    assert LieIdeal.model_validate_json(radical.model_dump_json()) == radical


def test_semisimple_solvable_and_nilpotent_fixtures() -> None:
    assert _rows(lie_solvable_radical(SL2)) == ()
    assert _rows(lie_solvable_radical(HEISENBERG)) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_eight_dimensional_result_composes_with_ideal_and_quotient_owners() -> None:
    radical = lie_solvable_radical(SL2_PLUS_ABELIAN_FIVE)

    assert _rows(radical) == tuple(
        tuple(Fraction(int(column == row + 3)) for column in range(8))
        for row in range(5)
    )
    assert check_ideal(SL2_PLUS_ABELIAN_FIVE, radical).is_ideal
    restored = LieIdeal.model_validate_json(radical.model_dump_json())
    quotient = lie_quotient(
        SL2_PLUS_ABELIAN_FIVE,
        restored,
        ("q_e", "q_f", "q_h"),
    )
    assert quotient.quotient.basis == ("q_e", "q_f", "q_h")
    assert tuple(
        (item.i, item.j, item.k, item.coefficient.as_fraction())
        for item in quotient.quotient.structure_constants
    ) == ((0, 1, 2, Fraction(1)), (0, 2, 0, Fraction(-2)), (1, 2, 1, Fraction(2)))


def test_manifest_declares_the_typed_radical_result_and_example() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "lie_algebra.solvable_radical.compute"
    )
    assert tool.result_type is LieIdeal
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert _rows(tool.run(request)) == _rows(lie_solvable_radical(AFFINE))
