"""Catalog adapter coverage for mathematical operations."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


@pytest.mark.parametrize(
    "operation_id",
    (
        "algebra.trigonometric_rational.normalize",
        "matroid.intersection.maximum_weight.compute",
        "weyl_group.element.act_on_root.compute",
        "weyl_group.element.act_on_weight.compute",
        "topology.simplicial_set.quotient.compute",
        "topology.simplicial_set.map.induced_chain_map.compute",
    ),
)
def test_published_math_operation_example_executes_through_catalog(
    operation_id: str,
) -> None:
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None and operation.examples

    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    operation.result_type.model_validate_json(json.dumps(result.output))
