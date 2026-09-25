"""Character Hecke images retain their exact source/target space parent."""

from __future__ import annotations

import pytest
from cypari import pari
from pydantic import TypeAdapter

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms._pari_basis_worker import (
    _as_cyclotomic_coordinates,
    _character_vector,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_coordinates_hecke,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterHeckeRequest,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
    _pari_character_request,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_FIELD = RationalCyclotomicField(order=6)


def _space() -> ModularFormSpace:
    """The inflated even order-three character has a one-dimensional cusp space."""
    return ModularFormSpace(
        level=26,
        weight=2,
        kind="S",
        character=dirichlet_character(character_group(26), (4,)),
        coefficient_domain=_FIELD,
    )


def _element(value: object) -> RationalCyclotomicElement:
    coordinates = _as_cyclotomic_coordinates(pari, value, 6)
    return RationalCyclotomicElement(
        field=_FIELD,
        coefficients_ascending=tuple(
            {"num": int(numerator), "den": int(denominator)}
            for numerator, denominator in coordinates
        ),
    )


def _form() -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=_space(),
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(
            RationalCyclotomicElement(
                field=_FIELD,
                coefficients_ascending=(
                    {"num": 1, "den": 1},
                    {"num": 0, "den": 1},
                ),
            ),
        ),
    )


def test_character_hecke_action_preserves_rref_target_and_matches_pari() -> None:
    form = _form()
    operation = Catalog.open().operation(
        "modular_form.character_coordinates.hecke.apply"
    )
    assert operation is not None
    result = operation.run(ModularCharacterHeckeRequest(form=form, index=5))

    assert result.space == form.space
    assert result.basis_id == form.basis_id
    assert len(result.coordinates) == 1
    restored = TypeAdapter(ModularFormCoordinates).validate_json(
        result.model_dump_json()
    )
    assert restored == result

    # PARI's modular-form Hecke operator supplies an independent exact
    # q-expansion oracle. In this one-dimensional space, the T_5 eigenvalue is
    # the ratio at any nonzero source coefficient; the q-Sturm reconstruction
    # in the operation returns that same scalar in Jacobian's RREF basis.
    character_request = _pari_character_request(form.space)
    pari_group, pari_character = _character_vector(
        pari, character_request["character"], 6
    )
    pari_space = pari.mfinit([26, 2, [pari_group, pari_character]], 1)
    pari_form = pari.mfbasis(pari_space)[0]
    pari_image = pari.mfhecke(pari_space, pari_form, 5)
    source_coefficients = tuple(pari.mfcoef(pari_form, index) for index in range(8))
    pivot = next(index for index, value in enumerate(source_coefficients) if value)
    expected = pari.mfcoef(pari_image, pivot) / source_coefficients[pivot]
    assert result.coordinates[0] == _element(expected)


def test_character_hecke_rejects_bad_prime_for_its_exact_level() -> None:
    with pytest.raises(OperationDomainValidationError, match="gcd"):
        modular_character_coordinates_hecke(_form(), 2)
