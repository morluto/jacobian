"""Exact scalar extension and q-expansion over an explicit cyclotomic parent."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    BASIS_ID,
    modular_form_coordinates_equal,
)
from jacobian.math.number_theory.modular_forms.field_coordinates import (
    modular_form_coordinates_extend_field,
    modular_form_field_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)


def _rational_form() -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=4, kind="M"),
        basis_id=BASIS_ID,
        coordinates=(CanonicalRational(num=2, den=1),),
    )


def _field_coordinates(value: RationalCyclotomicElement) -> tuple[Fraction, ...]:
    return tuple(
        coordinate.as_fraction() for coordinate in value.coefficients_ascending
    )


def test_scalar_extension_and_q_expansion_preserve_exact_parent() -> None:
    field = RationalCyclotomicField(order=6)
    extended = modular_form_coordinates_extend_field(_rational_form(), field)

    assert extended.space == ModularFormSpace(
        level=1,
        weight=4,
        kind="M",
        coefficient_domain=field,
    )
    assert extended.basis_id == BASIS_ID
    assert _field_coordinates(extended.coordinates[0]) == (Fraction(2), Fraction(0))
    assert (
        ModularFormCoordinates.model_validate_json(extended.model_dump_json())
        == extended
    )
    restored = ModularFormCoordinates.model_validate_json(extended.model_dump_json())
    assert modular_form_coordinates_equal(extended, restored)
    malformed = ModularFormCoordinates.model_construct(
        space=extended.space,
        basis_id=extended.basis_id,
        coordinates=None,
    )
    with pytest.raises(OperationDomainValidationError, match="basis and shape"):
        modular_form_coordinates_equal(malformed, extended)
    with pytest.raises(OperationDomainValidationError, match="basis and shape"):
        modular_form_field_coordinates_q_expansion(malformed, 3)

    expansion = modular_form_field_coordinates_q_expansion(extended, 3)

    assert expansion.space == extended.space
    assert tuple(_field_coordinates(value) for value in expansion.coefficients) == (
        (Fraction(2), Fraction(0)),
        (Fraction(480), Fraction(0)),
        (Fraction(4_320), Fraction(0)),
    )
    assert type(expansion).model_validate_json(expansion.model_dump_json()) == expansion


def test_scalar_extension_rejects_unadmitted_field() -> None:
    with pytest.raises(OperationDomainValidationError, match=r"exactly Q\(zeta_6\)"):
        modular_form_coordinates_extend_field(
            _rational_form(), RationalCyclotomicField(order=5)
        )


def test_field_coordinate_operations_are_published_and_examples_execute() -> None:
    catalog = Catalog.open()
    expected = {
        "modular_form.coordinates.extend_field.compute",
        "modular_form.field_coordinates.q_expansion.compute",
        "modular_form.character_coordinates.product.compute",
    }
    assert expected <= {tool.operation_id for tool in TOOLS}
    assert all(catalog.operation(operation_id) is not None for operation_id in expected)
    for tool in TOOLS:
        if tool.operation_id not in expected:
            continue
        for example in tool.examples:
            request = tool.request_type.model_validate_json(
                json.dumps(example.input), strict=True
            )
            assert tool.run(request) is not None
