"""Public dispatch serializes owner aliases from its advertised output schema."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_dispatch_pauli_output_uses_wire_aliases_and_composes_unchanged() -> None:
    catalog = Catalog.open()
    operation_id = "quantum.pauli.qubit.from_labels.compute"
    payload = {
        "register": {"qubit_ids": ["left", "right"]},
        "labels": ["I", "Y"],
        "phase": 2,
    }

    result = invoke_operation(operation_id, payload, catalog)

    descriptor = catalog.inspect(operation_id)
    assert descriptor is not None
    Draft202012Validator(descriptor.output_schema).validate(result.output)
    assert result.output["source"] == payload
    pauli = result.output["pauli"]
    assert pauli == {
        "phase_free": {
            "register": payload["register"],
            "x_bits": [0, 1],
            "z_bits": [0, 1],
        },
        "phase": 3,
    }
    recovered = invoke_operation(
        "quantum.pauli.qubit.to_labels.compute", {"pauli": pauli}, catalog
    )
    assert recovered.output["labels"] == payload["labels"]
    assert recovered.output["phase"] == payload["phase"]
