"""Serialized canonical complexes compose through declared structural schemas."""

import copy

import pytest
from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


@pytest.mark.parametrize(
    "suffix",
    [
        "barycentric_subdivision.compute",
        "deletion.compute",
        "elementary_collapse.check",
        "f_vector.compute",
        "join.compute",
        "link.compute",
        "pseudomanifold.decide",
        "shelling.check",
        "skeleton.compute",
        "star.compute",
    ],
)
def test_canonicalized_outputs_feed_structural_operations_unchanged(
    suffix: str,
) -> None:
    operation_id = f"topology.simplicial_complex.{suffix}"
    catalog = Catalog.open()
    operation = catalog.inspect(operation_id)
    assert operation is not None
    payload = copy.deepcopy(operation.examples[0].input)
    expected = invoke_operation(operation_id, payload, catalog).output
    for name in ("complex", "complex_a", "complex_b"):
        if name in payload:
            payload[name] = invoke_operation(
                "topology.simplicial_complex.canonicalize", payload[name], catalog
            ).output["complex"]
    Draft202012Validator(operation.input_schema).validate(payload)
    assert invoke_operation(operation_id, payload, catalog).output == expected
