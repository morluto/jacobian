"""Publication check for the polymorphism-invariant relation closure.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import RelationalInvariantClosure


def test_source_bound_value_roundtrips_and_catalog_call_agrees() -> None:
    operation_id = "relation.closure_under_operations.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    direct = tool.run(tool.request_type.model_validate(tool.examples[0].input))
    restored = RelationalInvariantClosure.model_validate_json(direct.model_dump_json())
    assert restored == direct
    assert direct.tuples == ((0,), (1,))

    output = invoke_operation(
        operation_id,
        json.loads(
            tool.request_type.model_validate(tool.examples[0].input).model_dump_json()
        ),
        Catalog.open(),
    ).output
    assert RelationalInvariantClosure.model_validate(output) == direct
