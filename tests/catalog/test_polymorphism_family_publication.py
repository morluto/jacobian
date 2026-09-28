"""Publication check for the polymorphism family operation.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    RelationalPolymorphismFamily,
)
from jacobian.math.logic.relational_structures._models import (
    RelationalPolymorphismEnumerationRequest,
)


def test_operation_catalog_example_and_serialized_call() -> None:
    operation_id = "relational.polymorphisms.arity.enumerate"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = RelationalPolymorphismEnumerationRequest.model_validate(
        tool.examples[0].input
    )
    direct = tool.run(request)
    assert direct.operation_tables == ((0, 0), (0, 1))

    output = invoke_operation(
        operation_id,
        json.loads(request.model_dump_json()),
        Catalog.open(),
    ).output
    decoded = RelationalPolymorphismFamily.model_validate(output)
    assert decoded == direct
