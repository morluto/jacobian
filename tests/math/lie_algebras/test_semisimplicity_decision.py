"""Exact Cartan semisimplicity decisions for finite-dimensional QQ algebras."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieSemisimplicityResult,
)
from jacobian.math.lie_algebras.operations import lie_algebra_is_semisimple


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


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
ABELIAN = _algebra(("a", "b"), ())
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))


def test_cartan_criterion_known_answers() -> None:
    # Hand-computed classical Killing forms: sl2 has determinant -128;
    # Heisenberg and abelian adjoints are nilpotent/zero; for [a,b]=b,
    # ad_a=diag(0,1) and ad_b is nilpotent, giving diag(1,0).
    assert lie_algebra_is_semisimple(SL2).is_semisimple
    assert not lie_algebra_is_semisimple(HEISENBERG).is_semisimple
    assert not lie_algebra_is_semisimple(ABELIAN).is_semisimple
    # Degenerate but nonzero Killing form distinguishes degeneracy from
    # vanishing as a decision case.
    assert not lie_algebra_is_semisimple(AFFINE).is_semisimple
    assert lie_algebra_is_semisimple(SL2).model_dump() == {
        "algebra": SL2.model_dump(),
        "is_semisimple": True,
    }


def test_decision_roundtrip_retains_source_algebra() -> None:
    decision = lie_algebra_is_semisimple(SL2)
    restored = LieSemisimplicityResult.model_validate_json(decision.model_dump_json())
    assert restored == decision
    assert restored.algebra == SL2
    assert restored.is_semisimple is True


def test_decision_wire_value_requires_a_json_boolean() -> None:
    with pytest.raises(ValidationError):
        LieSemisimplicityResult.model_validate_json('{"is_semisimple":1}')


def test_native_semisimplicity_projection_uses_exact_killing_radical() -> None:
    assert lie_algebra_is_semisimple(SL2).is_semisimple
    assert not lie_algebra_is_semisimple(HEISENBERG).is_semisimple


def test_native_semisimplicity_normalizes_malformed_algebra_mapping() -> None:
    malformed = {
        "basis": ["a", "a"],
        "structure_constants": [],
    }
    with pytest.raises(OperationDomainValidationError) as error:
        lie_algebra_is_semisimple(malformed)
    assert error.value.errors() == ({
        "loc": (),
        "type": "lie_algebra.input",
        "msg": "algebra must be a valid finite-dimensional Lie algebra",
    },)
