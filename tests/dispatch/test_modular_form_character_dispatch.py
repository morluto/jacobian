"""Catalog and dispatch boundaries for modular-form character operations.

The mathematical contracts of these operations are owned by the
``tests/math/number_theory/modular_forms`` modules. Booting the complete
product boundary is forbidden there, so the published catalog and dispatch
surfaces are exercised from this owner instead.
"""

from __future__ import annotations

from pydantic import TypeAdapter
from tests.math.number_theory.modular_forms.test_character_coordinates import (
    _element,
    _form,
    _space,
)
from tests.math.number_theory.modular_forms.test_character_hecke_target import (
    _form as _hecke_form,
)

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterHeckeRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
)


def test_equal_check_operation_executes_on_restored_character_coordinates() -> None:
    space = _space()
    first = _form(space, _element(1), _element(0))
    second = _form(space, _element(0), _element(1))
    adapter = TypeAdapter(ModularFormCoordinates)
    restored_first = adapter.validate_json(first.model_dump_json())
    restored_second = adapter.validate_json(second.model_dump_json())

    tool = Catalog.open().operation("modular_form.equal.check")
    assert tool is not None
    result = tool.run(tool.request_type(left=restored_first, right=restored_second))
    assert result.equal is False
    assert tool.run(tool.request_type(left=restored_first, right=restored_first)).equal


def test_character_hecke_operation_is_published_and_executes() -> None:
    form = _hecke_form()
    tool = Catalog.open().operation("modular_form.character_coordinates.hecke.apply")
    assert tool is not None
    result = tool.run(ModularCharacterHeckeRequest(form=form, index=5))

    assert result.space == form.space
    assert result.basis_id == form.basis_id
    assert (
        TypeAdapter(ModularFormCoordinates).validate_json(result.model_dump_json())
        == result
    )
