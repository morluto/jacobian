from __future__ import annotations

from fractions import Fraction

from jacobian.math.lie_algebras._models import FiniteDimensionalLieAlgebra, LieIdeal
from jacobian.math.lie_algebras.operations import (
    check_ideal,
    lie_derived_subalgebra,
    lie_quotient,
)


def _algebra(basis: list[str], constants: list[tuple[int, int, int, int]]):
    return FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": basis,
            "structure_constants": [
                {
                    "i": i,
                    "j": j,
                    "k": k,
                    "coefficient": {"num": value, "den": 1},
                }
                for i, j, k, value in constants
            ],
        }
    )


def _fraction_rows(ideal: LieIdeal) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in ideal.generators.entries
    )


def test_heisenberg_derived_ideal_matches_bracket_span_oracle_and_quotient() -> None:
    algebra = _algebra(["x", "y", "z"], [(0, 1, 2, 1)])

    derived = lie_derived_subalgebra(algebra)

    # Independent basis-bracket table: the only nonzero bracket direction is z.
    assert _fraction_rows(derived) == ((Fraction(0), Fraction(0), Fraction(1)),)
    assert derived.algebra == algebra
    assert check_ideal(algebra, derived).is_ideal
    quotient = lie_quotient(algebra, derived, ("xbar", "ybar"))
    assert len(quotient.quotient.basis) == 2


def test_affine_and_abelian_derived_subalgebras() -> None:
    affine = _algebra(["x", "y"], [(0, 1, 1, 1)])
    abelian = _algebra(["a", "b"], [])

    assert _fraction_rows(lie_derived_subalgebra(affine)) == (
        (Fraction(0), Fraction(1)),
    )
    zero_derived = lie_derived_subalgebra(abelian)
    assert _fraction_rows(zero_derived) == ()
    assert zero_derived.algebra == abelian
