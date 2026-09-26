"""Exact multidimensional Hecke matrices on character-valued forms."""

from __future__ import annotations

from itertools import product
from math import gcd

import pytest
from cypari import pari
from pydantic import ValidationError

from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms._pari_basis_worker import (
    _character_vector,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke._tools import (
    TOOLS,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.models import (
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeMatrixResult,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.operations import (
    modular_character_hecke_matrix_multidimensional,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
    _pari_character_request,
    pari_character_basis,
)


def _request(character_coordinate: int = 6, index: int = 2):
    payload = dict(TOOLS[0].examples[0].input)
    space = dict(payload["space"])
    character = dict(space["character"])
    character["coordinates"] = [character_coordinate]
    space["character"] = character
    payload["space"] = space
    payload["index"] = index
    return ModularCharacterHeckeMatrixRequest.model_validate(payload)


def _rational(field, value: int) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            {"num": value if index == 0 else 0, "den": 1}
            for index in range(field.degree)
        ),
    )


def test_multidimensional_t2_matrix_matches_independent_pari_oracle():
    request = _request()
    result = modular_character_hecke_matrix_multidimensional(
        request.space, request.index
    )
    assert len(result.entries) == 2
    assert result.basis_id == "gamma0-cyclotomic-character-sturm-rref-v1"
    assert result.model_validate(result.model_dump(mode="python")) == result

    # PARI's mfheckemat uses its own basis; its exact characteristic polynomial
    # is basis invariant and independently checks the returned matrix.
    character_request = _pari_character_request(request.space)
    pari_group, pari_character = _character_vector(
        pari, character_request["character"], 6
    )
    pari_space = pari.mfinit(
        [request.space.level, request.space.weight, [pari_group, pari_character]], 4
    )
    pari_matrix = pari.mfheckemat(pari_space, request.index)
    assert pari.charpoly(pari_matrix) == pari("x^2 - 1")

    a, b = result.entries[0]
    c, d = result.entries[1]
    trace = cyclotomic.add(a, d)
    determinant = cyclotomic.subtract(
        cyclotomic.multiply(a, d), cyclotomic.multiply(b, c)
    )
    field = request.space.coefficient_domain
    assert trace == _rational(field, 0)
    assert determinant == _rational(field, -1)


def test_order_three_field_transport_matches_pari_hecke_matrix():
    request = _request(character_coordinate=4)
    result = modular_character_hecke_matrix_multidimensional(
        request.space, request.index
    )

    # The helper returns coefficients in Jacobian's Q(x), x^2-x+1=0,
    # transporting PARI's q^2 coefficient t+2 via t=x-1.
    raw = pari_character_basis(
        request.space,
        5,
        len(result.entries),
        character_request=_pari_character_request(request.space),
    )
    assert raw[0][2] == (1, 1)

    # In PARI's native basis T_2 is diagonal with eigenvalues t+2 and 2t+1.
    character_request = _pari_character_request(request.space)
    pari_group, pari_character = _character_vector(
        pari, character_request["character"], 6
    )
    pari_space = pari.mfinit(
        [request.space.level, request.space.weight, [pari_group, pari_character]], 4
    )
    pari_matrix = pari.mfheckemat(pari_space, request.index)
    assert pari_matrix[0, 0] == pari("Mod(t+2,t^2+t+1)")
    assert pari_matrix[1, 1] == pari("Mod(2*t+1,t^2+t+1)")
    trace = cyclotomic.add(result.entries[0][0], result.entries[1][1])
    determinant = cyclotomic.subtract(
        cyclotomic.multiply(result.entries[0][0], result.entries[1][1]),
        cyclotomic.multiply(result.entries[0][1], result.entries[1][0]),
    )
    assert trace.coefficients_ascending[0].num == 0
    assert trace.coefficients_ascending[1].num == 3
    assert determinant.coefficients_ascending[0].num == -3
    assert determinant.coefficients_ascending[1].num == 3

    with pytest.raises(ValidationError):
        _request(character_coordinate=6, index=13)


def test_public_example_runs_through_its_typed_contract():
    request = ModularCharacterHeckeMatrixRequest.model_validate(
        TOOLS[0].examples[0].input
    )
    result = TOOLS[0].run(request)
    assert isinstance(result, ModularCharacterHeckeMatrixResult)
    assert (
        result.entries
        == modular_character_hecke_matrix_multidimensional(
            request.space, request.index
        ).entries
    )


def test_precision_81_boundary_is_admitted_for_level39_t8():
    # For Γ0(39), B=11, so T_8 requires n(B-1)+1=81 source coefficients.
    source_character = dirichlet_character(character_group(13), (6,))
    target_group = character_group(39)
    target_character = None
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        candidate = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source_character, residue).value
            for residue in range(39)
            if gcd(residue, 39) == 1
        ):
            target_character = candidate
            break
    assert target_character is not None
    space_payload = dict(TOOLS[0].examples[0].input["space"])
    space_payload["level"] = 39
    space_payload["character"] = target_character
    from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

    space = ModularFormSpace.model_validate(space_payload)
    result = modular_character_hecke_matrix_multidimensional(space, 8)
    assert len(result.entries) == 6
    assert result.index == 8
    assert result.labels == tuple(f"q^{index}" for index in range(6))
