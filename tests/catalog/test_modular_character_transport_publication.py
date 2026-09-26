"""Character transport declarations remain published by the owner manifest."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.number_theory.modular_forms.character_basis_tools import TOOLS

_OPERATION_IDS = {
    "modular_form.character_coordinates.transport.compute",
    "modular_form.character.equal.check",
}


def test_modular_character_transport_operations_are_published() -> None:
    catalog = Catalog.open()
    declarations = {
        tool.operation_id: tool for tool in TOOLS if tool.operation_id in _OPERATION_IDS
    }

    assert declarations.keys() == _OPERATION_IDS
    for operation_id, declaration in declarations.items():
        assert catalog.operation(operation_id) is declaration
