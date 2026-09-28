"""Publication check for the primitive-positive formula operation.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    PPDefinedRelation,
)
from jacobian.math.logic.relational_structures._models import (
    PPFormulaEvaluationRequest,
)


def test_operation_is_discoverable_and_serialized_consumer_value_is_exact() -> None:
    operation_id = "pp_formula.evaluate_relation.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = PPFormulaEvaluationRequest.model_validate(tool.examples[0].input)
    direct = tool.run(request)
    assert direct.tuples == ((0, 2),)

    wire_input = json.loads(request.model_dump_json())
    output = invoke_operation(operation_id, wire_input, Catalog.open()).output
    assert output["tuples"] == [[0, 2]]
    assert PPDefinedRelation.model_validate(output).tuples == direct.tuples
