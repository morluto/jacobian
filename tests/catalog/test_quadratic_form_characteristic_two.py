"""Catalog dispatch for the published characteristic-two quadratic examples."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_characteristic_two_examples_execute_through_catalog() -> None:
    catalog = Catalog.open()
    for operation_id, expected in (
        ("quadratic_form.characteristic_two.evaluate.compute", ["0"]),
        ("quadratic_form.characteristic_two.polar_pairing.compute", ["1"]),
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None
        example = operation.examples[0]

        result = invoke_operation(operation_id, example.input, catalog)

        assert result.output["value"]["coordinates"] == expected
