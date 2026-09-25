from __future__ import annotations

import math

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_mixed_jacobi_sum,
)


def _principal_character(modulus: int):
    group = character_group(modulus)
    return dirichlet_character(group, (0,) * len(group.generator_orders))


def test_three_character_jacobi_sum_matches_direct_integer_oracle() -> None:
    # The principal character is 1 on units and 0 elsewhere; this oracle uses
    # the defining sum directly and does not use the operation's cyclotomic
    # exponent encoding or polynomial reduction.
    character = _principal_character(5)
    result = dirichlet_character_mixed_jacobi_sum((character,) * 3)
    oracle = sum(
        1
        for first in range(5)
        for second in range(5)
        if math.gcd(first, 5) == 1
        and math.gcd(second, 5) == 1
        and math.gcd(1 - first - second, 5) == 1
    )
    assert oracle == 13
    assert result.characters == (character,) * 3
    assert result.value.field.order == 4
    assert tuple(
        (value.num, value.den) for value in result.value.coefficients_ascending
    ) == (
        (13, 1),
        (0, 1),
    )


def test_mixed_jacobi_sum_requires_three_characters_in_same_parent() -> None:
    character = _principal_character(5)
    with pytest.raises(OperationDomainValidationError) as count_error:
        dirichlet_character_mixed_jacobi_sum((character, character))  # type: ignore[arg-type]
    assert count_error.value.errors()[0]["type"] == (
        "dirichlet_character.mixed_jacobi_sum.character_count"
    )
    other = _principal_character(3)
    with pytest.raises(OperationDomainValidationError) as parent_error:
        dirichlet_character_mixed_jacobi_sum((character, character, other))
    assert parent_error.value.errors()[0]["type"] == (
        "dirichlet_character.mixed_jacobi_sum.parent_mismatch"
    )


def test_mixed_jacobi_sum_rejects_quadratic_residue_work_before_expansion() -> None:
    character = _principal_character(1024)
    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_mixed_jacobi_sum((character,) * 3)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.mixed_jacobi_sum.work_bound"
    )


def test_three_character_jacobi_sum_matches_gaussian_integer_oracle() -> None:
    group = character_group(5)
    characters = (
        dirichlet_character(group, (1,)),
        dirichlet_character(group, (3,)),
        dirichlet_character(group, (2,)),
    )
    # With 2 as generator modulo 5, the coordinates 1, 3, and 2
    # have values 1, i, and -1 at that generator, respectively.
    values = (
        {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (-1, 0)},
        {0: (0, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (-1, 0)},
        {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (-1, 0), 4: (1, 0)},
    )
    oracle_real = 0
    oracle_imaginary = 0
    for first in range(5):
        for second in range(5):
            third = (1 - first - second) % 5
            a, b, c = values[0][first], values[1][second], values[2][third]
            ab = (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
            oracle_real += ab[0] * c[0] - ab[1] * c[1]
            oracle_imaginary += ab[0] * c[1] + ab[1] * c[0]

    result = dirichlet_character_mixed_jacobi_sum(characters)
    assert result.value.field.order == 4
    assert tuple(
        (value.num, value.den) for value in result.value.coefficients_ascending
    ) == (
        (oracle_real, 1),
        (oracle_imaginary, 1),
    )


def test_serialized_mixed_jacobi_result_binds_source_parent_and_field() -> None:
    from pydantic import ValidationError

    from jacobian._exact import CanonicalRational
    from jacobian.math.matrices.cyclic_linear import (
        RationalCyclotomicElement,
        RationalCyclotomicField,
    )
    from jacobian.math.number_theory.characters._models import (
        DirichletCharacterMixedJacobiSumResult,
    )

    character = _principal_character(5)
    result = dirichlet_character_mixed_jacobi_sum((character,) * 3)
    payload = result.model_dump()
    payload["characters"] = (character, character, _principal_character(3))
    with pytest.raises(ValidationError, match="parent_mismatch"):
        DirichletCharacterMixedJacobiSumResult.model_validate(payload)

    payload = result.model_dump()
    payload["value"] = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=2),
        coefficients_ascending=(CanonicalRational(num=13, den=1),),
    )
    with pytest.raises(ValidationError, match="field_parent_mismatch"):
        DirichletCharacterMixedJacobiSumResult.model_validate(payload)
