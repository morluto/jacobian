"""Catalog projection checks for finite periodic congruence unions."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_periodic_congruence_examples_execute_through_catalog() -> None:
    catalog = Catalog.open()
    for operation_id in (
        "congruence.periodic_union.measure.compute",
        "congruence.periodic_union.profile.compute",
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert operation.examples
        for invocation_example in operation.examples:
            result = invoke_operation(operation_id, invocation_example.input, catalog)
            validated = operation.result_type.model_validate(result.output)
            assert validated.model_dump(mode="json") == result.output
