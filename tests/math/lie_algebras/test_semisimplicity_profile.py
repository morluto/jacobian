"""Exact Cartan semisimplicity profiles for finite-dimensional QQ algebras."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieSemisimplicityProfileResult,
)
from jacobian.math.lie_algebras.operations import lie_semisimplicity_profile


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


def _matrix_entries(
    profile: LieSemisimplicityProfileResult,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row)
        for row in profile.killing_radical.killing_result.killing_form.entries
    )


def test_cartan_criterion_known_answers() -> None:
    # Hand-computed classical Killing forms: sl2 has determinant -128;
    # Heisenberg and abelian adjoints are nilpotent/zero; for [a,b]=b,
    # ad_a=diag(0,1) and ad_b is nilpotent, giving diag(1,0).
    sl2 = lie_semisimplicity_profile(SL2)
    assert sl2.is_semisimple
    assert sl2.killing_radical.radical.generators.row_count == 0
    assert _matrix_entries(sl2) == (
        (Fraction(0), Fraction(4), Fraction(0)),
        (Fraction(4), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(8)),
    )

    for algebra in (HEISENBERG, ABELIAN):
        profile = lie_semisimplicity_profile(algebra)
        assert not profile.is_semisimple
        assert profile.killing_radical.radical.generators.row_count == len(
            algebra.basis
        )
        assert all(value == 0 for row in _matrix_entries(profile) for value in row)

    # Degenerate but nonzero Killing form guards against treating vanishing
    # and degeneracy as interchangeable.
    affine = lie_semisimplicity_profile(AFFINE)
    assert not affine.is_semisimple
    assert affine.killing_radical.radical.generators.row_count == 1
    assert _matrix_entries(affine) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )


def test_profile_retains_exact_source_through_strict_json_roundtrip() -> None:
    profile = lie_semisimplicity_profile(HEISENBERG)
    restored = LieSemisimplicityProfileResult.model_validate_json(
        profile.model_dump_json()
    )
    assert restored == profile
    assert restored.algebra == HEISENBERG
    assert not restored.is_semisimple
    assert restored.killing_radical.radical.basis == HEISENBERG.basis


def test_profile_decode_rejects_status_inconsistent_with_radical() -> None:
    payload = json.loads(lie_semisimplicity_profile(HEISENBERG).model_dump_json())
    payload["is_semisimple"] = True
    with pytest.raises(ValidationError) as exc_info:
        LieSemisimplicityProfileResult.model_validate_json(json.dumps(payload))
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.semisimplicity_profile_status"
    )


def test_catalog_exposes_one_profile_operation_with_exact_example() -> None:
    tool = Catalog.open().operation("lie_algebra.semisimplicity.profile.compute")
    assert tool is not None
    assert tool.result_type is LieSemisimplicityProfileResult
    assert tool.examples[0].name == "sl2_is_semisimple"
    request = tool.request_type.model_validate_json(
        json.dumps({"algebra": SL2.model_dump(mode="json")})
    )
    assert tool.run(request).is_semisimple
