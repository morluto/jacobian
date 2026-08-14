from __future__ import annotations

import pytest

from jacobian.operation_catalog import OperationCatalogError
from jacobian.operation_locators import (
    FamilyLocator,
    ModuleLocator,
    decode_locator,
    encode_locator,
)


def test_family_locator_round_trips() -> None:
    locator = FamilyLocator(family="graph")
    encoded = encode_locator(locator)
    assert encoded == '{"family":"graph","kind":"family"}'
    assert decode_locator(encoded) == locator


def test_module_locator_round_trips_with_symbol() -> None:
    locator = ModuleLocator(
        module="jacobian.domains.matrix_lattice.operation_declarations",
        symbol="MATRIX_DETERMINANT_COMPUTE",
    )
    assert decode_locator(encode_locator(locator)) == locator


def test_bare_module_path_decodes_as_module_locator() -> None:
    assert decode_locator("jacobian.domains.matrix_lattice.operation_declarations") == (
        ModuleLocator(module="jacobian.domains.matrix_lattice.operation_declarations")
    )


def test_legacy_family_prefix_fails_closed() -> None:
    with pytest.raises(OperationCatalogError, match="jacobian update"):
        decode_locator("family:graph")
