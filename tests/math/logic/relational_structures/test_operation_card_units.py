"""Published operation cards must name the unit admission actually uses."""

from __future__ import annotations

import pytest

from jacobian.math.logic.relational_structures._tools import TOOLS

# Cards whose runtime admission counts retained structural entries.
CELL_ENVELOPE_OPERATIONS = (
    "relational.polymorphisms.arity.enumerate",
    "relation.closure_under_operations.compute",
)


@pytest.mark.parametrize("operation_id", CELL_ENVELOPE_OPERATIONS)
def test_no_card_promises_an_encoded_byte_ceiling(operation_id: str) -> None:
    """A cell envelope is not a byte ceiling, and the card must not imply one.

    ``admit_polymorphism_family`` and the closure admission count retained
    entries and report cells. Encoded byte size can differ substantially, since
    a result repeats its register and table context across rows, so claiming a
    serialized ceiling gives a schema-driven client a guarantee the native
    boundary does not make.
    """
    tool = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    assert "bytes" not in tool.description
    assert "cell" in tool.description
