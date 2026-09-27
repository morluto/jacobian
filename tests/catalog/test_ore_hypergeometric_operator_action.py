"""The published proper-hypergeometric operator action composes through the catalog."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_published_shift_action_example_executes_with_source_bound_result():
    catalog = Catalog.open()
    operation_id = "ore.proper_hypergeometric.apply_shift_operator.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    assert operation.examples

    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    validated = operation.result_type.model_validate_json(
        json.dumps(result.output), strict=True
    )
    assert validated.operator.model_dump(mode="json") == result.output["operator"]
    assert validated.term.model_dump(mode="json") == result.output["term"]
    assert validated.relative_multiplier.variables == ("n", "k")
