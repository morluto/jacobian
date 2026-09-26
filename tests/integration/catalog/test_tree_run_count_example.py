"""Check the published run-count example through the complete dispatch boundary."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_run_count_catalog_example_executes_and_serializes() -> None:
    operation_id = "tree_automaton.nondeterministic.run_counts.compute"
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    assert operation.examples

    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    validated = operation.result_type.model_validate_json(json.dumps(result.output))
    assert validated.run_counts_by_size == (1, 2, 2)
